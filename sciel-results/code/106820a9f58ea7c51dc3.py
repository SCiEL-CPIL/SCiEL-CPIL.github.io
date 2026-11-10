import numpy as np
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Initial guess: Keich construction
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    # Precompute constants
    a = (np.arange(n) + 1) / n
    M = 20000
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n  # Shape (M,)
    
    # Precompute A = a[:, None] * y[None, :] for L calculation
    A = a[:, None] * y[None, :]
    
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin
    
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
        
        for k_idx in range(n - 1):
            mask = overlap_mask[k_idx, :]
            grad_L_sorted[k_idx, mask] -= 1.0
            grad_L_sorted[k_idx + 1, mask] += 1.0
        
        # Scatter gradients back to original indices
        grad_x = np.zeros(n)
        np.add.at(grad_x, I, grad_L_sorted)
        
        # The area is the mean of union lengths.
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        area = np.mean(union_lens)
        
        # The gradient of the area is the mean of the gradients of the union lengths.
        grad_x /= M
        
        return area, grad_x

    def obj_func(x_vals):
        area, _ = calc_area_and_grad(x_vals)
        return area

    def grad_func(x_vals):
        _, grad = calc_area_and_grad(x_vals)
        return grad

    best_x = x.copy()
    best_area = obj_func(x)
    
    # Try L-BFGS-B optimization
    try:
        res = scipy.optimize.minimize(
            obj_func,
            x,
            method='L-BFGS-B',
            jac=grad_func,
            bounds=[(-1.0, 2.0) for _ in range(n)],
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if res.fun < best_area:
            best_area = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 60.0:
        return best_x
    
    # Try Nelder-Mead for further refinement
    try:
        res2 = scipy.optimize.minimize(
            obj_func,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
        )
        
        if res2.fun < best_area:
            best_area = res2.fun
            best_x = res2.x.copy()
    except Exception:
        pass
    
    # Check time again
    if time.time() > deadline - 30.0:
        return best_x
    
    # Try a few random restarts with local optimization if time permits
    rng = np.random.RandomState(42)
    for _ in range(5):
        if time.time() > deadline - 60.0:
            break
        
        # Perturb the best solution
        x_try = best_x + rng.randn(n) * 0.1
        
        try:
            res3 = scipy.optimize.minimize(
                obj_func,
                x_try,
                method='L-BFGS-B',
                jac=grad_func,
                bounds=[(-1.0, 2.0) for _ in range(n)],
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if res3.fun < best_area:
                best_area = res3.fun
                best_x = res3.x.copy()
        except Exception:
            continue
    
    return best_x

import scipy.optimize
