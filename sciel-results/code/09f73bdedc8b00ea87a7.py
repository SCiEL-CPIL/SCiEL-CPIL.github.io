import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0

def construct():
    def soft_min(values, k):
        if len(values) == 0:
            return 0.0
        m = np.min(values)
        shifted = values - m
        # Clip shifted to avoid overflow in exp if k is huge and shifted is negative?
        # shifted >= 0 always.
        exp_terms = np.exp(-k * shifted)
        sum_exp = np.sum(exp_terms)
        return m - (1.0/k) * np.log(sum_exp)

    def objective_soft(x, k):
        c = x.reshape(N, 2)
        
        # Wall distances
        wall_dists = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists = dists[mask] / 2.0
        
        values = np.append(wall_dists, pair_dists)
        
        sm = soft_min(values, k)
        return -sm

    def get_score(coords):
        wall_dists = np.minimum(0.5 - np.abs(coords[:, 0]), 0.5 - np.abs(coords[:, 1]))
        min_wall = np.min(wall_dists)
        
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists = dists[mask] / 2.0
        min_circle = np.min(pair_dists) if len(pair_dists) > 0 else 0
        
        return np.min([min_wall, min_circle])

    def initial_guess(seed):
        rng = np.random.default_rng(seed)
        # 7x7 grid
        m = 7
        # Spread points slightly less than 0.5 to allow movement
        x_coords = np.linspace(-0.45, 0.45, m)
        y_coords = np.linspace(-0.45, 0.45, m)
        grid_points = np.array([(x, y) for y in y_coords for x in x_coords])
        grid_points += rng.uniform(-0.02, 0.02, grid_points.shape)
        
        # Add 50th point
        # Place it randomly
        p50 = rng.uniform(-0.45, 0.45, 2)
        
        points = np.vstack([grid_points, p50])
        return points.flatten()

    best_r = 0.0
    best_x = None

    # Parameters
    num_restarts = 8
    k_steps = 5
    k_start = 10.0
    k_end = 200.0

    for i in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT:
            break

        x0 = initial_guess(i)
        x_current = x0

        for step in range(k_steps):
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
            # Interpolate k
            k = k_start + (k_end - k_start) * (step / max(1, k_steps - 1))
            
            try:
                res = minimize(
                    objective_soft, 
                    x_current, 
                    args=(k,), 
                    method='SLSQP', 
                    bounds=[(-0.5, 0.5)] * (2*N), 
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                
                if res.success:
                    x_current = res.x
                else:
                    # If it fails, we might try to continue with current x
                    pass
            except Exception:
                pass

        # Calculate score
        r = get_score(x_current.reshape(N, 2))
        if r > best_r:
            best_r = r
            best_x = x_current.copy()

    # Final polish with very high k on the best solution
    if best_x is not None and time.time() - START_TIME < TIME_LIMIT - 10:
        try:
            k_final = 1000.0
            res = minimize(
                objective_soft, 
                best_x, 
                args=(k_final,), 
                method='SLSQP', 
                bounds=[(-0.5, 0.5)] * (2*N), 
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.success:
                r_final = get_score(res.x.reshape(N, 2))
                if r_final > best_r:
                    best_x = res.x
        except Exception:
            pass

    return best_x.reshape(N, 2)
