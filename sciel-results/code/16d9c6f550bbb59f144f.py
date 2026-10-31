"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 to maximize sum of radii.
    Uses a combination of grid initialization and gradient-based optimization.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Perimeter constraint: 2*(W + H) = 4 => W + H = 2
    # We'll optimize over W, H, and circle positions/radii
    # Variables: [W, H, x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
    # Total: 2 + 3*21 = 65 variables
    
    def objective(params):
        """Negative sum of radii (we minimize negative sum)."""
        W = params[0]
        H = params[1]
        
        # Enforce perimeter constraint softly
        if W <= 0 or H <= 0:
            return 1e10
        if abs(2*(W + H) - 4) > 1e-6:
            # Penalize deviation from perimeter constraint
            penalty = 100 * (2*(W + H) - 4)**2
        else:
            penalty = 0
        
        # Extract circles
        circles = params[2:].reshape(N, 3)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Penalty for negative radii
        neg_r_penalty = np.sum(np.maximum(0, -rs)) * 100
        
        # Penalty for circles going outside rectangle
        out_penalty = 0
        for i in range(N):
            if xs[i] < rs[i] or xs[i] > W - rs[i]:
                out_penalty += max(0, rs[i] - xs[i]) + max(0, xs[i] - (W - rs[i]))
            if ys[i] < rs[i] or ys[i] > H - rs[i]:
                out_penalty += max(0, rs[i] - ys[i]) + max(0, ys[i] - (H - rs[i]))
        out_penalty *= 100
        
        # Penalty for overlapping circles
        overlap_penalty = 0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist)**2 * 50
        
        total_penalty = penalty + neg_r_penalty + out_penalty + overlap_penalty
        
        return -np.sum(rs) + total_penalty
    
    def gradient(params):
        """Numerical gradient."""
        eps = 1e-7
        grad = np.zeros_like(params)
        f0 = objective(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective(params_plus)
            params_minus = params.copy()
            params_minus[i] -= eps
            f_minus = objective(params_minus)
            grad[i] = (f_plus - f_minus) / (2*eps)
        return grad
    
    # Multiple random restarts
    best_params = None
    best_value = -1e10
    
    # Initial guess: grid-like arrangement
    # Try aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # W + H = 2, W/H = ar => H = 2/(1+ar), W = 2*ar/(1+ar)
        H_init = 2.0 / (1.0 + ar)
        W_init = 2.0 * ar / (1.0 + ar)
        
        # Grid initialization
        n_cols = int(np.round(np.sqrt(N * W_init / H_init)))
        n_rows = int(np.ceil(N / n_cols))
        
        params = np.zeros(2 + 3*N)
        params[0] = W_init
        params[1] = H_init
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            x = (col + 0.5) * W_init / n_cols
            y = (row + 0.5) * H_init / n_rows
            r = 0.5 * min(W_init / n_cols, H_init / n_rows) * 0.9
            params[2 + 3*i] = x
            params[3 + 3*i] = y
            params[4 + 3*i] = r
        
        # Add some randomness
        for _ in range(3):
            if time.time() > deadline:
                break
            
            # Perturb
            perturbed = params.copy()
            perturbed[2:] += np.random.normal(0, 0.05, size=3*N)
            
            # Optimize
            try:
                result = minimize(
                    objective, 
                    perturbed, 
                    method='BFGS', 
                    jac=None,  # Let it use numerical gradient
                    options={'maxiter': 200, 'gtol': 1e-5}
                )
                
                if result.success or result.fun < objective(params):
                    # Check if valid
                    W = result.x[0]
                    H = result.x[1]
                    if W > 0 and H > 0 and abs(2*(W+H) - 4) < 0.01:
                        val = -result.fun
                        # Verify no overlaps
                        circles = result.x[2:].reshape(N, 3)
                        valid = True
                        for i in range(N):
                            for j in range(i+1, N):
                                dist = np.sqrt((circles[i,0]-circles[j,0])**2 + (circles[i,1]-circles[j,1])**2)
                                if dist < circles[i,2] + circles[j,2] - 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and val > best_value:
                            best_value = val
                            best_params = result.x.copy()
            except Exception:
                pass
    
    # If we found a good solution, use it
    if best_params is not None:
        W = best_params[0]
        H = best_params[1]
        circles = best_params[2:].reshape(N, 3)
        
        # Final polish with tighter tolerance
        if time.time() < deadline - 10:
            try:
                result = minimize(
                    objective,
                    best_params,
                    method='L-BFGS-B',
                    bounds=[(0.1, None)] * 2 + [(0, W_init), (0, H_init), (0.01, 0.5)] * N,
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if result.fun < objective(best_params):
                    best_params = result.x
                    W = best_params[0]
                    H = best_params[1]
                    circles = best_params[2:].reshape(N, 3)
            except Exception:
                pass
        
        # Ensure validity: clip radii and positions
        for i in range(N):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], W - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], H - circles[i, 2])
            circles[i, 2] = max(0.001, circles[i, 2])
        
        return circles
    
    # Fallback: simple grid
    W = 1.0
    H = 1.0
    n_cols = 5
    n_rows = 5
    cells = []
    for i in range(N):
        col = i % n_cols
        row = i // n_cols
        x = (col + 0.5) * W / n_cols
        y = (row + 0.5) * H / n_rows
        r = 0.5 * min(W / n_cols, H / n_rows) * 0.95
        cells.append([x, y, r])
    
    return np.array(cells)
# EVOLVE-BLOCK-END
