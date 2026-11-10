import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # Precompute constants
    a = (np.arange(n) + 1) / n  # shape (n,)
    n_heights = 20000
    y = (np.arange(n_heights) + 0.5) / n_heights  # shape (n_heights,)
    w_const = 1.0 / n
    
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin before 600s kill
    
    def compute_area(x):
        """Exact nonsmooth area computation."""
        # L shape (n_heights, n)
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        diffs = np.diff(L_sorted, axis=1)  # shape (n_heights, n-1)
        w = (1.0 - y) / n  # shape (n_heights,)
        union_lengths = w + np.sum(np.minimum(diffs, w[:, None]), axis=1)
        return np.mean(union_lengths)
    
    def compute_area_smooth(x, tau=0.01):
        """Smooth surrogate using log-sum-exp soft-min."""
        L = x[None, :] + a[None, :] * y[:, None]  # (n_heights, n)
        L_sorted = np.sort(L, axis=1)
        diffs = np.diff(L_sorted, axis=1)  # (n_heights, n-1)
        w = (1.0 - y) / n  # (n_heights,)
        
        # min(diffs, w) element-wise, then soft-sum over axis=1
        # We want sum_j min(diffs[:,j], w) approximated smoothly.
        # min(d, w) = -softmax(-d, -w) * ... actually simpler:
        # min(d, w) = d * sigmoid((w-d)/(2*tau)) + w * sigmoid((d-w)/(2*tau))
        # But that's not quite right either. Let's use:
        # min(d, w) = -softmin(-d, -w) where softmin(a,b) = -logsumexp(-a/tau, -b/tau)*tau
        # Actually, for the sum over j of min(d_j, w), we can just compute
        # min(d_j, w) exactly (it's piecewise linear in d_j) and the non-smoothness
        # is in the sort. The sort is the main issue.
        # Alternative: use the exact min but smooth the sort via a differentiable
        # relaxation. That's complex.
        
        # Simpler approach: just use the exact formula but accept that the
        # gradient is subgradient. L-BFGS-B can handle this with numerical
        # gradients or we can provide an analytical subgradient.
        # For now, let's just use the exact compute_area and let L-BFGS-B
        # use numerical gradients. It's slower per eval but more robust.
        return compute_area(x)
    
    # Keich baseline
    j = np.arange(1, k + 1)
    x0 = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x_best = np.array(x0, dtype=np.float64)
    area_best = compute_area(x_best)
    
    # Phase 1: L-BFGS-B with numerical gradients
    # Use a limited number of iterations to stay within time budget
    try:
        result = minimize(
            compute_area, x_best, method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if result.fun < area_best:
            x_best = result.x
            area_best = result.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        return x_best
    
    # Phase 2: Coordinate descent polishing with exact objective
    # Multiple passes, shrinking window
    for pass_num in range(5):
        if time.time() > deadline:
            break
        improved = False
        window = 0.5 / (pass_num + 1)
        for i in range(n):
            if time.time() > deadline:
                break
            def f_1d(xi):
                x_temp = x_best.copy()
                x_temp[i] = xi
                return compute_area(x_temp)
            
            lower = x_best[i] - window
            upper = x_best[i] + window
            res_i = minimize_scalar(f_1d, bounds=(lower, upper), method='bounded',
                                    options={'xatol': 1e-9})
            if res_i.success:
                if abs(res_i.x - x_best[i]) > 1e-12:
                    x_best[i] = res_i.x
                    improved = True
                if res_i.fun < area_best:
                    area_best = res_i.fun
    
    if time.time() > deadline:
        return x_best
    
    # Phase 3: Final hill-climbing with adaptive step size
    rng = np.random.default_rng(42)
    step_size = 1e-3
    for _ in range(30000):
        if time.time() > deadline:
            break
        i = rng.integers(0, n)
        delta = rng.normal(0, step_size)
        x_new = x_best.copy()
        x_new[i] += delta
        new_area = compute_area(x_new)
        if new_area < area_best:
            x_best = x_new
            area_best = new_area
            step_size *= 0.98
        else:
            step_size *= 0.995
            if step_size < 1e-6:
                step_size = 1e-3  # Reset occasionally
    
    return x_best
