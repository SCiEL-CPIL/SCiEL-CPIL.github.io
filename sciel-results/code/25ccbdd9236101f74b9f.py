import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill

    def _radius(c):
        """Compute the maximum radius for given centers."""
        d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        bx = 0.5 - np.abs(c[:, 0])
        by = 0.5 - np.abs(c[:, 1])
        min_b = np.min(np.minimum(bx, by))
        return min(min_d / 2, min_b)

    def _penalty_smooth(c, r):
        """
        Smooth penalty function for L-BFGS-B.
        Uses softplus approximation of max(0, x) for differentiability.
        """
        # Pairwise distances
        d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Softplus: log(1 + exp(x/beta)) approximates max(0, x)
        beta = 0.01
        viol_d = 2 * r - d
        E_d = np.sum(np.log1p(np.exp(viol_d / beta)) * beta)
        
        # Boundary
        bx = 0.5 - np.abs(c[:, 0])
        by = 0.5 - np.abs(c[:, 1])
        b = np.minimum(bx, by)
        viol_b = r - b
        E_b = np.sum(np.log1p(np.exp(viol_b / beta)) * beta)
        
        return E_d + E_b

    def _grad_penalty_smooth(c, r):
        """Gradient of the smooth penalty function."""
        beta = 0.01
        grad = np.zeros_like(c)
        
        # Pairwise
        d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        
        viol_d = 2 * r - d
        sigmoid_d = 1.0 / (1.0 + np.exp(-viol_d / beta))
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = c[i] - c[j]
                d_ij = d[i, j]
                if d_ij < 1e-12:
                    continue
                grad[i] += (2 * sigmoid_d[i, j] / d_ij) * diff
        
        # Boundary
        bx = 0.5 - np.abs(c[:, 0])
        by = 0.5 - np.abs(c[:, 1])
        b = np.minimum(bx, by)
        viol_b = r - b
        sigmoid_b = 1.0 / (1.0 + np.exp(-viol_b / beta))
        
        for i in range(N):
            # x boundary
            if c[i, 0] > 0:
                grad[i, 0] -= sigmoid_b[i] * (1 if b[i] == bx[i] else 0)
            else:
                grad[i, 0] += sigmoid_b[i] * (1 if b[i] == bx[i] else 0)
            # y boundary
            if c[i, 1] > 0:
                grad[i, 1] -= sigmoid_b[i] * (1 if b[i] == by[i] else 0)
            else:
                grad[i, 1] += sigmoid_b[i] * (1 if b[i] == by[i] else 0)
        
        return grad

    def _init_grid(seed=42):
        """Generate initial grid configuration."""
        rng = np.random.RandomState(seed)
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        c = grid[:N].copy()
        c += rng.normal(0, 0.003, c.shape)
        c = np.clip(c, -0.49, 0.49)
        return c

    def try_radius(r, c0, max_iter=2000):
        """
        Try to find a feasible packing for radius r using L-BFGS-B.
        Returns (success, best_c, best_radius).
        """
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        def obj(x):
            c = x.reshape(N, 2)
            return _penalty_smooth(c, r)
        
        def grad(x):
            c = x.reshape(N, 2)
            return _grad_penalty_smooth(c, r).ravel()
        
        x0 = c0.ravel()
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                jac=grad, bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            return False, c0, _radius(c0)
        
        c_opt = result.x.reshape(N, 2)
        r_found = _radius(c_opt)
        
        # Check feasibility: penalty should be near zero
        penalty = _penalty_smooth(c_opt, r)
        success = penalty < 1e-6 and r_found >= r - 1e-5
        
        return success, c_opt, r_found

    # Bisection on radius
    r_low = 0.065
    r_high = 0.072
    
    best_c = _init_grid(42)
    best_r = _radius(best_c)
    
    # First, get a good starting point at a moderate radius
    c_start = _init_grid(42)
    success, c_start, r_start = try_radius(0.068, c_start, max_iter=3000)
    if r_start > best_r:
        best_r = r_start
        best_c = c_start
    
    # Bisection: ~10 iterations to narrow down
    for iteration in range(12):
        if time.time() - start_time > timeout - 10:
            break
        
        r_mid = (r_low + r_high) / 2
        
        # Try multiple initializations for robustness
        found = False
        for seed in [42, 123, 456, 789]:
            if time.time() - start_time > timeout - 10:
                break
            
            c0 = _init_grid(seed)
            # Warm start from best if available
            if best_c is not None and iteration > 0:
                c0 = best_c + np.random.RandomState(seed).normal(0, 0.002, best_c.shape)
                c0 = np.clip(c0, -0.49, 0.49)
            
            success, c_try, r_try = try_radius(r_mid, c0, max_iter=1500)
            
            if success:
                found = True
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try
                r_low = r_mid
                break
        
        if not found:
            r_high = r_mid
    
    # Final refinement: try to push slightly above r_low
    if time.time() - start_time < timeout - 5:
        for r_try in [r_low + 0.0001, r_low + 0.0002, r_low + 0.0003]:
            if time.time() - start_time > timeout - 5:
                break
            c0 = best_c.copy()
            success, c_try, r_try_found = try_radius(r_try, c0, max_iter=2000)
            if success and r_try_found > best_r:
                best_r = r_try_found
                best_c = c_try
    
    # Ensure validity
    if not np.all(np.isfinite(best_c)) or np.any(np.abs(best_c) > 0.5 + 1e-10):
        best_c = _init_grid(42)
        best_r = _radius(best_c)
    
    return best_c
