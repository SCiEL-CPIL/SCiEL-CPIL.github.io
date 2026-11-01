"""Optimal packing of 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

# Best known radius from Huang, Ye, Cantrell (2010)
KNOWN_RADIUS = 0.036612798904

def _score(positions):
    """Compute the minimum radius for given positions."""
    # Check boundaries
    min_boundary = np.min(np.minimum(0.5 - np.abs(positions[:, 0]), 
                                      0.5 - np.abs(positions[:, 1])))
    if min_boundary <= 0:
        return -1.0
    
    # Compute all pairwise distances
    n = len(positions)
    # Use vectorized computation for speed
    dx = positions[:, 0:1] - positions[0:1, 0]
    dy = positions[:, 1:2] - positions[0:1, 1]
    
    # Only compute upper triangle to save time
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((positions[i, 0] - positions[j, 0])**2 + 
                        (positions[i, 1] - positions[j, 1])**2)
            if d < min_dist:
                min_dist = d
                if min_dist < 2 * min_boundary:
                    break
        if min_dist < 2 * min_boundary:
            break
    
    return min(min_boundary, min_dist / 2)

def _hexagonal_lattice(n_circles):
    """Generate initial positions using hexagonal lattice."""
    positions = []
    # Hexagonal lattice spacing
    # For n circles, we need roughly sqrt(n) rows
    rows = int(np.ceil(np.sqrt(n_circles * 2 / np.sqrt(3))))
    
    for row in range(rows):
        for col in range(rows + 1):
            if len(positions) >= n_circles:
                break
            x = (col - row / 2) * 1.0
            y = row * np.sqrt(3) / 2
            # Scale to fit in [-0.5, 0.5]^2
            x = x * 0.9 / max(rows, 1)
            y = y * 0.9 / max(rows, 1)
            # Center in the square
            x = x - np.mean([p[0] for p in positions]) if positions else x
            positions.append([x, y])
    
    # If we don't have enough, add random points
    while len(positions) < n_circles:
        positions.append([np.random.uniform(-0.45, 0.45), 
                         np.random.uniform(-0.45, 0.45)])
    
    return np.array(positions[:n_circles])

def _optimize_from_initial(initial_pos, max_iter=1000, tol=1e-12):
    """Optimize positions starting from initial configuration."""
    n = len(initial_pos)
    
    def objective(x):
        """Minimize the negative of the minimum radius."""
        pos = x.reshape(n, 2)
        
        # Boundary constraint
        boundary = np.min(np.minimum(0.5 - np.abs(pos[:, 0]), 
                                     0.5 - np.abs(pos[:, 1])))
        
        # Pairwise distances
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + 
                           (pos[i, 1] - pos[j, 1])**2)
                if d < min_dist:
                    min_dist = d
        
        # We want to maximize min(boundary, min_dist/2)
        # Equivalent to minimizing -min(boundary, min_dist/2)
        # Use a smooth approximation
        r = min(boundary, min_dist / 2)
        return -r
    
    def gradient(x):
        """Compute gradient of the objective."""
        pos = x.reshape(n, 2)
        grad = np.zeros_like(x)
        
        # Boundary gradient
        boundary = np.min(np.minimum(0.5 - np.abs(pos[:, 0]), 
                                     0.5 - np.abs(pos[:, 1])))
        
        # Find which boundary constraint is active
        for i in range(n):
            for k in range(2):
                val = 0.5 - np.abs(pos[i, k])
                if val <= boundary + 1e-10:
                    # Gradient of -val with respect to pos[i,k]
                    if pos[i, k] > 0:
                        grad[i * 2 + k] = 1.0
                    else:
                        grad[i * 2 + k] = -1.0
        
        # Pairwise distance gradient
        min_dist = np.inf
        min_pair = None
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + 
                           (pos[i, 1] - pos[j, 1])**2)
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        
        if min_pair is not None:
            i, j = min_pair
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx**2 + dy**2)
            if d > 1e-10:
                # Gradient of -d/2
                grad[i * 2] = -dx / d
                grad[i * 2 + 1] = -dy / d
                grad[j * 2] = dx / d
                grad[j * 2 + 1] = dy / d
        
        return grad
    
    # Use L-BFGS-B
    result = minimize(objective, initial_pos.flatten(), 
                     method='L-BFGS-B',
                     jac=gradient,
                     options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
    
    return result.x.reshape(n, 2)

def construct():
    """Construct optimal packing of 200 circles in unit square."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Generate multiple initial configurations
    best_pos = None
    best_score = -1.0
    
    # Try different random seeds for diversity
    np.random.seed(42)
    
    # Strategy 1: Hexagonal lattice with perturbations
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Create hexagonal lattice
        pos = _hexagonal_lattice(N)
        
        # Add small random perturbations
        pos += np.random.normal(0, 0.01, size=pos.shape)
        
        # Optimize
        try:
            optimized = _optimize_from_initial(pos, max_iter=500)
            score = _score(optimized)
            
            if score > best_score:
                best_score = score
                best_pos = optimized.copy()
        except Exception:
            continue
    
    # Strategy 2: Grid with perturbations
    if time.time() < deadline:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pos = grid[:N].copy()
        
        # Add perturbations
        pos += np.random.normal(0, 0.02, size=pos.shape)
        
        try:
            optimized = _optimize_from_initial(pos, max_iter=500)
            score = _score(optimized)
            
            if score > best_score:
                best_score = score
                best_pos = optimized.copy()
        except Exception:
            pass
    
    # Strategy 3: Pure random initialization
    if time.time() < deadline:
        pos = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        try:
            optimized = _optimize_from_initial(pos, max_iter=300)
            score = _score(optimized)
            
            if score > best_score:
                best_score = score
                best_pos = optimized.copy()
        except Exception:
            pass
    
    # Final polish: try to improve the best solution found
    if best_pos is not None and time.time() < deadline - 20:
        try:
            polished = _optimize_from_initial(best_pos, max_iter=1000, tol=1e-14)
            score = _score(polished)
            
            if score > best_score:
                best_score = score
                best_pos = polished.copy()
        except Exception:
            pass
    
    # Ensure we have a valid result
    if best_pos is None:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Verify and clip to valid range
    best_pos = np.clip(best_pos, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_pos
