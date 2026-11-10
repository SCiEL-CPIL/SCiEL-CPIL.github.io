# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _exact_area_and_subgrad(x, n=64, M=20000):
    """
    Compute the exact area of the union of triangles and a subgradient.
    The area is piecewise-linear in x, so we return a valid subgradient.
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    # L shape: (M, n)
    L = x[None, :] + a[None, :] * y[:, None]
    
    # Sort L at each height
    idx = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, idx, axis=1)
    
    # Gaps D_k = L_sorted[:, k+1] - L_sorted[:, k]
    D = L_sorted[:, 1:] - L_sorted[:, :-1]
    
    # Exact min(D, w)
    min_vals = np.minimum(D, w[:, None])
    sum_min = np.sum(min_vals, axis=1)
    
    U = w + sum_min
    area = np.mean(U)
    
    # Subgradient: for each gap D_k, the subgradient of min(D_k, w) w.r.t D_k is:
    # 1 if D_k < w, 0 if D_k > w, [0,1] if D_k == w
    # We use 0.5 at ties for a symmetric subgradient.
    subgrad_D = np.where(D < w[:, None], 1.0, np.where(D > w[:, None], 0.0, 0.5))
    
    # g_sorted: contribution to each sorted position
    g_sorted = np.zeros_like(L_sorted)
    g_sorted[:, :-1] -= subgrad_D
    g_sorted[:, 1:] += subgrad_D
    
    # Map back to x
    G = np.zeros_like(L)
    np.put_along_axis(G, idx, g_sorted, axis=1)
    subgrad_x = np.sum(G, axis=0) * a / M
    
    return area, subgrad_x

def _smoothed_area_and_grad(x, eps, n=64, M=20000):
    """
    Compute the smoothed area of the union of triangles and its gradient.
    Uses a log-sum-exp based smooth approximation for min(D, w).
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    L = x[None, :] + a[None, :] * y[:, None]
    
    idx = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, idx, axis=1)
    
    D = L_sorted[:, 1:] - L_sorted[:, :-1]
    
    diff = D - w[:, None]
    z = diff / eps
    sig = np.zeros_like(z)
    mask_pos = z >= 0
    mask_neg = ~mask_pos
    
    sig[mask_pos] = 1.0 / (1.0 + np.exp(z[mask_pos]))
    exp_neg_z = np.exp(-z[mask_neg])
    sig[mask_neg] = exp_neg_z / (1.0 + exp_neg_z)
    
    log_terms = np.logaddexp(-D / eps, -w[:, None] / eps)
    f_vals = -eps * log_terms
    sum_f = np.sum(f_vals, axis=1)
    
    U = w + sum_f
    area = np.mean(U)
    
    g_sorted = np.zeros_like(L_sorted)
    g_sorted[:, :-1] -= sig
    g_sorted[:, 1:] += sig
    
    G = np.zeros_like(L)
    np.put_along_axis(G, idx, g_sorted, axis=1)
    grad_x = np.sum(G, axis=0) * a / M
    
    return area, grad_x

def _keich_init(n):
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def _coordinate_descent_polish(x, n=64, M=20000, max_rounds=5, tol=1e-10):
    """
    Coordinate descent on the exact piecewise-linear area function.
    For each coordinate, find the best value by scanning candidate breakpoints.
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    best_x = x.copy()
    best_area = _exact_area_and_subgrad(best_x, n, M)[0]
    
    for round_num in range(max_rounds):
        improved = False
        for i in range(n):
            # For coordinate i, L_i(y) = x_i + a_i * y
            # The area as a function of x_i is piecewise linear with breakpoints
            # where L_i(y) crosses other L_j(y) or where min(D, w) switches.
            # We can find the optimal x_i by ternary search or by evaluating
            # at candidate breakpoints.
            
            # Simple approach: use the subgradient to determine direction,
            # then do a line search.
            area, subgrad = _exact_area_and_subgrad(best_x, n, M)
            
            if abs(subgrad[i]) < tol:
                continue
            
            # Line search along coordinate i
            # Try a few step sizes
            best_val = best_x[i]
            best_local_area = area
            
            # Determine search direction
            direction = -np.sign(subgrad[i])
            
            # Try decreasing step sizes
            step = 0.1
            for _ in range(20):
                candidate = best_x.copy()
                candidate[i] = best_x[i] + direction * step
                
                cand_area = _exact_area_and_subgrad(candidate, n, M)[0]
                
                if cand_area < best_local_area - 1e-12:
                    best_local_area = cand_area
                    best_val = candidate[i]
                    step *= 0.5
                else:
                    step *= 0.5
                    if step < 1e-12:
                        break
            
            if abs(best_val - best_x[i]) > 1e-12:
                best_x[i] = best_val
                improved = True
        
        if not improved:
            break
    
    return best_x

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    x = _keich_init(n)
    
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin (630s limit, 10s buffer)
    
    # Phase 1: Smoothed optimization with annealing
    eps_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for eps in eps_schedule:
        if time.time() > deadline - 120:
            break
            
        def func_and_grad(x_curr):
            return _smoothed_area_and_grad(x_curr, eps)
        
        try:
            res = minimize(
                func_and_grad, 
                x, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 300, 'maxfun': 800, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            
            if np.all(np.isfinite(res.x)):
                x = res.x
        except Exception:
            pass
    
    # Phase 2: Exact subgradient polishing with small steps
    if time.time() < deadline - 60:
        # Use the exact area with subgradient for final refinement
        best_x = x.copy()
        best_area = _exact_area_and_subgrad(best_x, n, M=20000)[0]
        
        # Simple subgradient descent with decreasing step size
        step = 0.01
        for it in range(200):
            if time.time() > deadline - 30:
                break
            
            area, subgrad = _exact_area_and_subgrad(best_x, n, M=20000)
            
            if area < best_area:
                best_area = area
            
            # Normalize subgradient
            sg_norm = np.linalg.norm(subgrad)
            if sg_norm < 1e-12:
                break
            
            # Update
            new_x = best_x - step * subgrad / sg_norm
            
            # Project: keep values finite and in reasonable range
            new_x = np.clip(new_x, -2.0, 2.0)
            
            new_area = _exact_area_and_subgrad(new_x, n, M=20000)[0]
            
            if new_area < best_area:
                best_area = new_area
                best_x = new_x
            else:
                # Reduce step size
                step *= 0.7
                if step < 1e-8:
                    break
        
        x = best_x
    
    # Phase 3: Coordinate descent polish on exact function
    if time.time() < deadline - 10:
        x = _coordinate_descent_polish(x, n, M=20000, max_rounds=3, tol=1e-11)
    
    # Ensure finite
    if not np.all(np.isfinite(x)):
        return _keich_init(n)
        
    return x

# EVOLVE-BLOCK-END
