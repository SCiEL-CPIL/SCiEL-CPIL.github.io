"""Rewritten program: circle packing in perimeter-4 rectangle.

Strategy:
- Perimeter 4 means 2*(w+h)=4 => w+h=2.
- We want to maximize sum of radii for 21 non-overlapping circles inside [0,w]x[0,h].
- Use a local optimization approach:
  1. Initialize circles on a hexagonal-like grid within the rectangle.
  2. Iteratively expand each circle as much as possible given constraints:
     - Circle must stay within rectangle: r <= min(x, w-x, y, h-y)
     - Circles must not overlap: r_i + r_j <= dist(i,j)
  3. Use multiple restarts with different initial configurations and aspect ratios.
  4. Final polish with gradient-based optimization using scipy.

We use a deadline guard to ensure we finish within 150s.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 140.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    best_value = 0.0
    best_result = None
    
    # Try different aspect ratios w/h where w+h=2
    # w in (0, 2), h = 2 - w
    # Reasonable aspect ratios: from very wide to very tall
    aspect_ratios = []
    for w_frac in np.linspace(0.3, 1.7, 15):
        w = w_frac
        h = 2.0 - w
        if w > 0.05 and h > 0.05:
            aspect_ratios.append((w, h))
    
    for w, h in aspect_ratios:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Try multiple initial configurations
        for restart in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            result = _solve_for_aspect(w, h, restart, start_time)
            if result is not None:
                value = np.sum(result[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = result.copy()
    
    # Final polish on best result if time permits
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        polished = _polish(best_result, start_time)
        if polished is not None:
            value = np.sum(polished[:, 2])
            if value > best_value:
                best_value = value
                best_result = polished
    
    if best_result is None:
        # Fallback: simple grid
        return _fallback()
    
    return best_result


def _fallback():
    """Simple grid fallback."""
    w, h = 1.0, 1.0
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side - 1e-4
            cells.append([x, y, r])
    return np.array(cells)


def _solve_for_aspect(w, h, restart, start_time):
    """Solve packing for given aspect ratio w x h (w+h=2)."""
    
    # Initialize positions on a hexagonal grid
    positions, radii = _initialize_hex(w, h, restart)
    
    if len(positions) < N:
        return None
    
    # Iterative expansion
    for iteration in range(200):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        improved = False
        for i in range(N):
            # Max radius based on walls
            r_wall = min(positions[i][0], w - positions[i][0], 
                        positions[i][1], h - positions[i][1])
            
            # Max radius based on neighbors
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                # r_i + r_j <= dist => r_i <= dist - r_j
                candidate = dist - radii[j]
                if candidate < r_max:
                    r_max = candidate
            
            r_max = max(r_max, 1e-6)
            
            if r_max > radii[i] + 1e-8:
                radii[i] = r_max
                improved = True
        
        if not improved:
            break
    
    # Local optimization to improve positions
    result = _local_optimize(positions, radii, w, h, start_time)
    
    if result is not None:
        # Validate
        if _is_valid(result, w, h):
            return result
    
    return None


def _initialize_hex(w, h, restart):
    """Initialize circles on a hexagonal grid."""
    positions = []
    radii = []
    
    # Number of rows and columns to roughly fill N circles
    # Use a hexagonal packing pattern
    n_cols = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    n_rows = int(np.ceil(N / n_cols))
    
    # Spacing
    dx = w / (n_cols + 0.5)
    dy = h / (n_rows + 0.5)
    
    # Hexagonal offset
    offset = 0.5
    
    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            x = (col + 0.5 + (offset if row % 2 else 0)) * dx
            y = (row + 0.5) * dy
            
            # Add some randomness for different restarts
            if restart > 0:
                x += np.random.uniform(-0.1, 0.1) * dx
                y += np.random.uniform(-0.1, 0.1) * dy
            
            # Clamp to bounds
            x = np.clip(x, 0.01, w - 0.01)
            y = np.clip(y, 0.01, h - 0.01)
            
            positions.append([x, y])
            # Initial radius: small
            r = 0.5 * min(dx, dy) * 0.4
            radii.append(r)
            count += 1
    
    # If we don't have enough, add more
    while len(positions) < N:
        x = np.random.uniform(0.05, w - 0.05)
        y = np.random.uniform(0.05, h - 0.05)
        positions.append([x, y])
        radii.append(0.05)
    
    return np.array(positions[:N]), np.array(radii[:N])


def _local_optimize(positions, radii, w, h, start_time):
    """Local optimization to maximize sum of radii."""
    
    def objective(params):
        """Negative sum of radii (we minimize)."""
        pos = params[:2*N].reshape(N, 2)
        rads = params[2*N:]
        
        # Penalty for going out of bounds
        penalty = 0.0
        for i in range(N):
            x, y = pos[i]
            r = rads[i]
            if x < r or x > w - r or y < r or y > h - r:
                penalty += 1000 * (max(0, r - x) + max(0, r - (w - x)) + 
                                  max(0, r - y) + max(0, r - (h - y)))
        
        # Penalty for overlap
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                overlap = rads[i] + rads[j] - dist
                if overlap > 0:
                    penalty += 1000 * overlap**2
        
        return -np.sum(rads) + penalty
    
    # Initial parameters
    x0 = np.concatenate([positions.flatten(), radii])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w))  # x
        bounds.append((0, h))  # y
        bounds.append((1e-6, min(w, h)/2))  # r
    
    # Try to optimize
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if res.success or res.fun < 0:
            pos = res.x[:2*N].reshape(N, 2)
            rads = res.x[2*N:]
            
            # Check validity
            result = np.column_stack([pos, rads])
            if _is_valid(result, w, h):
                return result
    except Exception:
        pass
    
    return None


def _is_valid(result, w, h, tol=1e-6):
    """Check if the packing is valid."""
    for i in range(len(result)):
        x, y, r = result[i]
        if r < 1e-8:
            return False
        if x < -tol or x > w + tol or y < -tol or y > h + tol:
            return False
        if x < r - tol or x > w - r + tol or y < r - tol or y > h - r + tol:
            return False
    
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            dist = np.linalg.norm(result[i, :2] - result[j, :2])
            if dist < result[i, 2] + result[j, 2] - tol:
                return False
    
    return True


def _polish(result, start_time):
    """Final high-precision polish."""
    w = result[:, 0].max() + result[:, 2].max()
    h = result[:, 1].max() + result[:, 2].max()
    
    # Better estimate of bounding box
    w = max(result[:, 0] + result[:, 2])
    h = max(result[:, 1] + result[:, 2])
    
    # Ensure perimeter constraint
    if 2*(w + h) > 4 + 1e-6:
        scale = 2.0 / (w + h)
        result = result * scale
    
    return result
