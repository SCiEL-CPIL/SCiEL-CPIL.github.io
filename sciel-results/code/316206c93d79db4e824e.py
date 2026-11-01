import numpy as np
import time
from scipy.optimize import minimize

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = 0.0
    best_solution = None
    
    # Penalty weight
    lam = 1000.0
    
    def unpack(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return x, y, r

    def soft_max_smooth(violations, temperature=0.01):
        """Smooth approximation of max(0, v) using softplus-like function"""
        # Use a smooth approximation: for v > 0, ~v; for v < 0, ~0
        # softplus(x) = log(1 + exp(x)) is smooth, but we want max(0, v)
        # Use: 0.5 * (v + sqrt(v^2 + eps)) which is smooth approximation of max(0, v)
        eps = temperature ** 2
        return 0.5 * (violations + np.sqrt(violations**2 + eps))

    def objective(v):
        x, y, r = unpack(v)
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Boundary penalties - use smooth max
        viol_x = r - x
        viol_1mx = r - (1 - x)
        viol_y = r - y
        viol_1my = r - (1 - y)
        
        pen_x = np.sum(soft_max_smooth(viol_x)**2)
        pen_1mx = np.sum(soft_max_smooth(viol_1mx)**2)
        pen_y = np.sum(soft_max_smooth(viol_y)**2)
        pen_1my = np.sum(soft_max_smooth(viol_1my)**2)
        pen_bound = pen_x + pen_1mx + pen_y + pen_1my
        
        # Overlap penalties - use smooth max
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        r_sum_sq = r_sum**2
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        violations = r_sum_sq[mask] - dist_sq[mask]
        smooth_viol = soft_max_smooth(violations)
        pen_overlap = np.sum(smooth_viol**2)
        
        total_penalty = pen_bound + pen_overlap
        
        return -sum_r + lam * total_penalty

    def is_valid(v):
        x, y, r = unpack(v)
        # Check boundaries
        if np.any(x < r - 1e-9) or np.any(1 - x < r - 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(1 - y < r - 1e-9):
            return False
        if np.any(r < -1e-9):
            return False
            
        # Check overlaps
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # dist >= r_sum for i != j
        dist[np.diag_indices(N)] = np.inf
        if np.any(dist < r_sum - 1e-9):
            return False
        return True

    def get_initial_guesses():
        guesses = []
        
        # Random
        for _ in range(3):
            x = np.random.rand(N)
            y = np.random.rand(N)
            r = np.random.rand(N) * 0.08 + 0.02
            v = np.concatenate([x, y, r])
            guesses.append(v)
            
        # Grid - 6x6 = 36, use first 32
        side = 6
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()[:N]
        grid_y = grid_y.flatten()[:N]
        grid_r = np.full(N, 0.5 / side * 0.95)
        v_grid = np.concatenate([grid_x, grid_y, grid_r])
        guesses.append(v_grid)
        
        # Perturbed Grid
        v_pert = v_grid + np.random.randn(3*N) * 0.005
        guesses.append(v_pert)
        
        # Another perturbed grid with different seed
        np.random.seed(42)
        v_pert2 = v_grid + np.random.randn(3*N) * 0.01
        np.random.seed()
        guesses.append(v_pert2)
        
        return guesses

    guesses = get_initial_guesses()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
    for i in range(N):
        bounds.append((0, 1)) # y
    for i in range(N):
        bounds.append((0, 0.5)) # r

    # First pass: moderate optimization
    for v0 in guesses:
        if time.time() > deadline - 30:  # Reserve 30s for polish
            break
            
        try:
            res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 1500, 'ftol': 1e-9, 'gtol': 1e-7, 'maxfun': 5000})
            
            if time.time() > deadline - 30:
                break
                
            if res.success or res.fun < 1e-5:
                v_opt = res.x
                v_opt = np.clip(v_opt, [b[0] for b in bounds], [b[1] for b in bounds])
                
                if is_valid(v_opt):
                    score = np.sum(v_opt[2*N:3*N])
                    if score > best_score:
                        best_score = score
                        best_solution = v_opt
        except Exception:
            pass

    # If no valid solution found, return a trivial valid one
    if best_solution is None:
        side = 6
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Second pass: high-precision polish on the best solution
    if time.time() < deadline - 10:
        try:
            # Use tighter tolerances for polish
            res_polish = minimize(objective, best_solution, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000})
            
            if time.time() < deadline - 5:
                v_polished = res_polish.x
                v_polished = np.clip(v_polished, [b[0] for b in bounds], [b[1] for b in bounds])
                
                if is_valid(v_polished):
                    score_polished = np.sum(v_polished[2*N:3*N])
                    if score_polished > best_score:
                        best_score = score_polished
                        best_solution = v_polished
        except Exception:
            pass

    x, y, r = unpack(best_solution)
    result = np.column_stack((x, y, r))
    return result
