"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    def objective(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Maximize sum of radii -> Minimize negative
        score = -np.sum(r)
        
        # Boundary penalties
        # Violation if x < r or x > 1-r
        # We use squared penalty
        px_neg = np.maximum(0, r - x)
        px_pos = np.maximum(0, r - (1 - x))
        py_neg = np.maximum(0, r - y)
        py_pos = np.maximum(0, r - (1 - y))
        
        boundary_pen = np.sum(px_neg**2) + np.sum(px_pos**2) + np.sum(py_neg**2) + np.sum(py_pos**2)
        
        # Overlap penalties
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx**2 + dy**2
        # Avoid sqrt if possible? No, distance is needed for linear violation
        d = np.sqrt(d2)
        
        r_sum = r[:, None] + r[None, :]
        
        # Only i < j
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        r_pairs = r_sum[iu]
        
        violations = r_pairs - d_pairs
        pos_viol = np.maximum(0, violations)
        overlap_pen = np.sum(pos_viol**2)
        
        # Weight for penalty
        # If weight is too high, it might prevent growth.
        # If too low, it allows overlap.
        # We want to maximize r, so penalty should be strong enough to keep valid.
        return score + 10.0 * (boundary_pen + overlap_pen)

    def is_valid(x, y, r):
        # Check bounds
        if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5):
            return False
        if np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5):
            return False
        if np.any(x < -1e-5) or np.any(x > 1 + 1e-5) or np.any(y < -1e-5) or np.any(y > 1 + 1e-5):
            return False
        if np.any(r < 0):
            return False
            
        # Check overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        r_sum = r[:, None] + r[None, :]
        
        # Check i < j
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        r_pairs = r_sum[iu]
        
        if np.any(d_pairs < r_pairs - 1e-5):
            return False
            
        return True

    best_result = None
    best_score = -np.inf

    # Initial guess: Grid
    def get_initial_guess():
        # 5x5 grid
        xs = np.linspace(0.1, 0.9, 5)
        ys = np.linspace(0.1, 0.9, 5)
        x, y = np.meshgrid(xs, ys)
        x = x.flatten()
        y = y.flatten()
        x = np.append(x, 0.5)
        y = np.append(y, 0.5)
        r = np.full(26, 0.08)
        return np.concatenate([x, y, r])

    p0 = get_initial_guess()
    bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N

    # Run optimization
    # We can try multiple restarts with random perturbations if the first one fails or is suboptimal
    # But the grid start is very good.
    # Let's just run one long optimization.
    
    res = minimize(objective, p0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'maxfun': 50000})
    
    x, y, r = res.x[:N], res.x[N:2*N], res.x[2*N:3*N]
    
    if is_valid(x, y, r):
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_result = np.column_stack((x, y, r))
    else:
        # If invalid, try to fix by shrinking radii?
        # Or just fallback to seed.
        # But let's try a few random restarts if grid fails.
        for i in range(5):
            if time.time() > deadline:
                break
            x0 = np.random.rand(N) * 0.8 + 0.1
            y0 = np.random.rand(N) * 0.8 + 0.1
            r0 = np.full(N, 0.05)
            p0_r = np.concatenate([x0, y0, r0])
            res_r = minimize(objective, p0_r, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'maxfun': 20000})
            xr, yr, rr = res_r.x[:N], res_r.x[N:2*N], res_r.x[2*N:3*N]
            if is_valid(xr, yr, rr):
                score_r = np.sum(rr)
                if score_r > best_score:
                    best_score = score_r
                    best_result = np.column_stack((xr, yr, rr))

    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
