import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    # Precompute A = a[:, None] * y[None, :]
    # L = x[:, None] + A
    A = a[:, None] * y[None, :]
    
    def calc_area_and_grad(x_vals):
        # x_vals: shape (n,)
        L = x_vals[:, None] + A  # Shape (n, M)
        
        # Sort L along axis 0
        I = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, I, axis=0)
        
        # Gaps
        d = L_sorted[1:, :] - L_sorted[:-1, :]  # Shape (n-1, M)
        
        # Determine where gaps are less than w (overlap)
        overlap_mask = d < w[None, :]
        
        # Gradient calculation
        grad_L_sorted = np.zeros_like(L)
        
        for k in range(n - 1):
            mask = overlap_mask[k, :]
            grad_L_sorted[k, mask] -= 1.0
            grad_L_sorted[k + 1, mask] += 1.0
            
        # Accumulate gradients back to original indices
        grad_x = np.zeros(n)
        np.add.at(grad_x, I, grad_L_sorted)
        
        # Union length and area
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        area = np.mean(union_lens)
        
        # Gradient of area is mean of gradients of union lengths
        grad_x /= M
        
        return area, grad_x

    def obj_func(x_vals):
        area, _ = calc_area_and_grad(x_vals)
        return area

    def grad_func(x_vals):
        _, grad = calc_area_and_grad(x_vals)
        return grad

    # Initial guess: Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x_best = np.array(positions, dtype=np.float64)
    area_best = obj_func(x_best)
    
    start_time = time.time()
    timeout = 630
    safety_margin = 10
    
    # Optimization function wrapper
    def run_optimization(x_start, max_iter=500):
        try:
            res = scipy.optimize.minimize(
                obj_func,
                x_start,
                method='L-BFGS-B',
                jac=grad_func,
                bounds=[(-1.0, 2.0) for _ in range(n)],
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            return res.x, res.fun
        except Exception:
            return x_start, obj_func(x_start)

    # 1. Initial optimization
    x_opt, area_opt = run_optimization(x_best, max_iter=1000)
    if area_opt < area_best:
        x_best = x_opt
        area_best = area_opt

    # 2. Stochastic Local Search / Perturbation
    # We will perturb the current best solution and try to optimize from there.
    # This helps escape local minima caused by permutation swaps.
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < timeout - safety_margin:
        # Create a perturbed version of the best solution
        # The noise scale should be small enough to stay in the same "region" 
        # or cross a nearby boundary, but large enough to potentially find a better basin.
        noise_scale = 0.01
        x_perturbed = x_best + rng.normal(0, noise_scale, n)
        
        # Run a shorter optimization from the perturbed point
        x_cand, area_cand = run_optimization(x_perturbed, max_iter=200)
        
        if area_cand < area_best:
            x_best = x_cand
            area_best = area_cand
            # If we found a significantly better solution, we might want to 
            # increase the noise or just continue. 
            # For now, we just update the base for future perturbations.
            
    return x_best
