import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
DEADLINE = 145.0  # Safety margin before 150s kill

def _validity_check(pos, radii, tol=1e-8):
    """Check if configuration is valid."""
    for i in range(N):
        if pos[i][0] - radii[i] < -tol or pos[i][0] + radii[i] > 1 + tol:
            return False
        if pos[i][1] - radii[i] < -tol or pos[i][1] + radii[i] > 1 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _enforce_validity(pos, radii):
    """Make configuration valid by shrinking radii if necessary."""
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp positions
    pos = np.clip(pos, 0, 1)
    
    # Clamp radii to wall distances
    for i in range(N):
        wall_dist = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        radii[i] = min(radii[i], max(wall_dist, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(500):
        overlap_found = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                overlap = radii[i] + radii[j] - d
                if overlap > 1e-10:
                    overlap_found = True
                    # Shrink the smaller circle
                    if radii[i] <= radii[j]:
                        shrink = min(overlap, radii[i] - 1e-6)
                        radii[i] -= shrink
                        rem = overlap - shrink
                        if rem > 0:
                            radii[j] -= min(rem, radii[j] - 1e-6)
                    else:
                        shrink = min(overlap, radii[j] - 1e-6)
                        radii[j] -= shrink
                        rem = overlap - shrink
                        if rem > 0:
                            radii[i] -= min(rem, radii[i] - 1e-6)
        if not overlap_found:
            break
    
    # Final wall check
    for i in range(N):
        wall_dist = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        radii[i] = min(radii[i], max(wall_dist, 1e-6))
    
    return pos, radii

def _grow_and_relax(pos, radii, deadline, max_iter=2000):
    """
    Grow radii, and if invalid, use gradient-based relaxation to fix overlaps.
    Uses Delaunay edges for efficient constraint identification.
    """
    current_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = current_score
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Try to grow radii
        growth_factor = 1.0005
        new_radii = radii * growth_factor
        
        if _validity_check(pos, new_radii):
            radii = new_radii
            current_score = np.sum(radii)
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                best_radii = radii.copy()
            continue
        
        # If invalid, relax positions to resolve overlaps
        # Use penalty method with gradient
        penalty_weight = 1000.0
        
        # Compute penalty and gradient
        penalty = 0.0
        grad = np.zeros_like(pos)
        
        # Pairwise overlaps
        for i in range(N):
            for j in range(i+1, N):
                d_vec = pos[i] - pos[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    d = 1e-10
                    d_vec = np.array([1e-10, 0])
                overlap = radii[i] + radii[j] - d
                if overlap > 0:
                    penalty += overlap**2
                    force = 2 * overlap * (d_vec / d)
                    grad[i] += force
                    grad[j] -= force
        
        # Wall violations
        for i in range(N):
            # Left
            pen = radii[i] - pos[i][0]
            if pen > 0:
                penalty += pen**2
                grad[i][0] += 2 * pen
            # Right
            pen = pos[i][0] + radii[i] - 1
            if pen > 0:
                penalty += pen**2
                grad[i][0] -= 2 * pen
            # Bottom
            pen = radii[i] - pos[i][1]
            if pen > 0:
                penalty += pen**2
                grad[i][1] += 2 * pen
            # Top
            pen = pos[i][1] + radii[i] - 1
            if pen > 0:
                penalty += pen**2
                grad[i][1] -= 2 * pen
        
        # Move in direction of gradient descent
        step_size = 0.005
        pos = pos - step_size * grad
        pos = np.clip(pos, 0, 1)
        
        # Check if we improved
        current_score = np.sum(radii)
        if _validity_check(pos, radii):
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                best_radii = radii.copy()
    
    return best_pos, best_radii, best_score

def _slsqp_refine(pos, radii, deadline, max_iter=500):
    """
    Refine using SLSQP with Delaunay edge constraints.
    This exploits the sparsity of the contact graph.
    """
    # Get Delaunay triangulation to identify likely contact pairs
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
    except Exception:
        edges = set()
        for i in range(N):
            for j in range(i+1, N):
                edges.add((i,j))
    
    # Convert to list
    edge_list = list(edges)
    
    def objective(params):
        p = params.reshape(N, 2)
        # Maximize sum of radii, so minimize negative
        return -np.sum(radii)
    
    def constraints():
        cons = []
        # Wall constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*2] - radii[i], 'jac': lambda p, i=i: _wall_jac_x(i, p)})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: (1 - p[i*2]) - radii[i], 'jac': lambda p, i=i: _wall_jac_x_neg(i, p)})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*2+1] - radii[i], 'jac': lambda p, i=i: _wall_jac_y(i, p)})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: (1 - p[i*2+1]) - radii[i], 'jac': lambda p, i=i: _wall_jac_y_neg(i, p)})
        
        # Pairwise constraints (only for Delaunay edges)
        for i, j in edge_list:
            cons.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: np.sqrt((p[i*2]-p[j*2])**2 + (p[i*2+1]-p[j*2+1])**2) - (radii[i] + radii[j]),
                'jac': lambda p, i=i, j=j: _pair_jac(i, j, p)
            })
        
        return cons
    
    def _wall_jac_x(i, p):
        g = np.zeros(2*N)
        g[i*2] = 1.0
        return g
    
    def _wall_jac_x_neg(i, p):
        g = np.zeros(2*N)
        g[i*2] = -1.0
        return g
    
    def _wall_jac_y(i, p):
        g = np.zeros(2*N)
        g[i*2+1] = 1.0
        return g
    
    def _wall_jac_y_neg(i, p):
        g = np.zeros(2*N)
        g[i*2+1] = -1.0
        return g
    
    def _pair_jac(i, j, p):
        g = np.zeros(2*N)
        dx = p[i*2] - p[j*2]
        dy = p[i*2+1] - p[j*2+1]
        d = np.sqrt(dx*dx + dy*dy)
        if d < 1e-10:
            d = 1e-10
        g[i*2] = dx/d
        g[i*2+1] = dy/d
        g[j*2] = -dx/d
        g[j*2+1] = -dy/d
        return g
    
    # SLSQP optimization
    x0 = pos.flatten()
    bounds = [(0, 1)] * (2*N)
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints(),
            options={'maxiter': max_iter, 'ftol': 1e-8}
        )
        
        if res.success:
            new_pos = res.x.reshape(N, 2)
            new_pos, new_radii = _enforce_validity(new_pos, radii)
            return new_pos, new_radii, np.sum(new_radii)
        
    except Exception:
        pass
    
    return pos, radii, np.sum(radii)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Multi-start with different initializations
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Seed values for reproducible randomness
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        rng = np.random.RandomState(seed)
        
        # Initialize with hexagonal-like packing
        # Try different grid configurations
        configs = [
            (6, 5),  # 6 cols, 5 rows
            (5, 6),
            (7, 4),
            (4, 7),
        ]
        
        for cols, rows in configs:
            if time.time() > deadline - 5:
                break
            
            # Create grid
            xs = np.linspace(0.5/cols, 1 - 0.5/cols, cols)
            ys = np.linspace(0.5/rows, 1 - 0.5/rows, rows)
            
            centers = []
            for x in xs:
                for y in ys:
                    centers.append([x, y])
            centers = np.array(centers)
            
            # Select N most central points
            d2c = np.sum((centers - 0.5)**2, axis=1)
            idx = np.argsort(d2c)[:N]
            pos = centers[idx].copy()
            
            # Add small jitter
            pos += rng.uniform(-0.01, 0.01, size=pos.shape)
            pos = np.clip(pos, 0.05, 0.95)
            
            # Initial radii
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, pos[i][0], 1-pos[i][0], pos[i][1], 1-pos[i][1])
            
            radii = np.full(N, min_dist * 0.9)
            
            # Ensure validity
            pos, radii = _enforce_validity(pos, radii)
            
            current_score = np.sum(radii)
            
            # Grow and relax
            pos, radii, score1 = _grow_and_relax(pos, radii, deadline, max_iter=1000)
            
            # SLSQP refinement
            pos, radii, score2 = _slsqp_refine(pos, radii, deadline, max_iter=300)
            
            if score2 > best_score:
                best_score = score2
                best_pos = pos.copy()
                best_radii = radii.copy()
    
    # Final validity check
    if best_pos is None or not _validity_check(best_pos, best_radii):
        # Fallback: tiny circles
        pos = np.random.RandomState(42).uniform(0.1, 0.9, size=(N, 2))
        radii = np.full(N, 0.01)
        best_pos, best_radii = _enforce_validity(pos, radii)
    
    result = np.column_stack((best_pos, best_radii))
    return result
