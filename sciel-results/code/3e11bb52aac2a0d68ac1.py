# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _calc_area_and_grad_smooth(x, n, y_vals, w_vals, a_vals, beta):
    """
    Calculate smooth area and gradient using log-sum-exp for min(d, w).
    beta controls smoothness: large beta -> closer to true min.
    """
    M = len(y_vals)
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    
    sorted_L = np.sort(L, axis=0)
    idx = np.argsort(L, axis=0)
    
    d = np.diff(sorted_L, axis=0)  # shape (n-1, M)
    w_b = w_vals[None, :]  # shape (1, M)
    
    # Smooth min(d, w) via log-sum-exp:
    # min(d, w) ≈ -1/beta * log(exp(-beta*d) + exp(-beta*w))
    # For numerical stability, use the max trick
    # Let a = -beta*d, b = -beta*w
    # log(exp(a) + exp(b)) = max(a,b) + log(exp(a-max) + exp(b-max))
    
    a = -beta * d
    b = -beta * w_b
    m = np.maximum(a, b)  # shape (n-1, M)
    # Avoid overflow in exp: exp(a - m) <= 1
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    logsum = m + np.log(ea + eb)
    smooth_min = -logsum / beta  # shape (n-1, M)
    
    union_lengths = w_vals + np.sum(smooth_min, axis=0)
    area = np.mean(union_lengths)
    
    # Gradient of smooth_min w.r.t. d:
    # d/d_d [-1/beta * log(exp(-beta*d) + exp(-beta*w))]
    # = -1/beta * (-beta * exp(-beta*d)) / (exp(-beta*d) + exp(-beta*w))
    # = exp(-beta*d) / (exp(-beta*d) + exp(-beta*w))
    # = 1 / (1 + exp(-beta*(w - d)))
    # = sigmoid(beta*(w - d))
    # This is between 0 and 1, approaching 1 when d << w (active) and 0 when d >> w (inactive)
    
    # grad_smooth_min w.r.t. d = sigmoid(beta*(w - d))
    # Use numerically stable sigmoid
    z = beta * (w_b - d)  # shape (n-1, M)
    # sigmoid(z) = 1/(1+exp(-z))
    # For z > 0: 1/(1+exp(-z)), for z <= 0: exp(z)/(1+exp(z))
    grad_d = np.empty_like(d)
    pos = z > 0
    grad_d[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    grad_d[~pos] = exp_z / (1.0 + exp_z)
    
    # Now build gradient w.r.t. sorted_L
    grad_L = np.zeros_like(sorted_L)  # shape (n, M)
    # grad_L[j+1, m] += grad_d[j, m]
    # grad_L[j, m] -= grad_d[j, m]
    grad_L[1:, :] += grad_d
    grad_L[:-1, :] -= grad_d
    
    # Map back to x using inverse permutation
    # For each m, grad_x[idx[m, :], m] = grad_L[:, m]
    # We can use take_along_axis: 
    # Create an index array that maps from original to sorted position
    # Actually, we need: for each original index i and each m, find j such that idx[m, j] == i
    # Then grad_x[i, m] = grad_L[j, m]
    # 
    # Alternative: use the fact that idx is a permutation for each m.
    # Create inv_idx where inv_idx[m, i] = j such that idx[m, j] = i
    # Then grad_x[i, m] = grad_L[inv_idx[m, i], m]
    
    # Build inverse index: for each m, inv_idx[m, idx[m, j]] = j
    inv_idx = np.zeros_like(idx)
    rows = np.arange(M)[None, :]  # shape (1, M)
    np.put_along_axis(inv_idx, idx, np.arange(n)[:, None], axis=0)
    
    # Now grad_x[i, m] = grad_L[inv_idx[m, i], m]
    # We need to gather: for each i, m, take grad_L[inv_idx[m, i], m]
    # Reshape for easier indexing
    # inv_idx shape (M, n), grad_L shape (n, M)
    # We want grad_x shape (n, M) where grad_x[i, m] = grad_L[inv_idx[m, i], m]
    
    # Transpose inv_idx to (n, M): inv_idx_T[i, m] = inv_idx[m, i]
    inv_idx_T = inv_idx.T  # shape (n, M)
    # Now gather: grad_x[i, m] = grad_L[inv_idx_T[i, m], m]
    # This is a 2D gather along axis 0 of grad_L
    grad_x = np.take_along_axis(grad_L, inv_idx_T, axis=0)  # shape (n, M)
    
    grad_x_mean = np.mean(grad_x, axis=1)
    
    return area, grad_x_mean

def _calc_area_and_grad(x, n, y_vals, w_vals, a_vals):
    """
    Calculate the exact area and its gradient with respect to x.
    """
    M = len(y_vals)
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    
    sorted_L = np.sort(L, axis=0)
    idx = np.argsort(L, axis=0)
    
    d = np.diff(sorted_L, axis=0)  # shape (n-1, M)
    w_b = w_vals[None, :]
    
    active = d < w_b
    
    min_terms = np.where(active, d, w_b)
    union_lengths = w_vals + np.sum(min_terms, axis=0)
    area = np.mean(union_lengths)
    
    grad_L = np.zeros_like(sorted_L)
    grad_L[1:, :] += active.astype(np.float64)
    grad_L[:-1, :] -= active.astype(np.float64)
    
    # Use inverse permutation for efficient scatter
    inv_idx = np.zeros_like(idx)
    np.put_along_axis(inv_idx, idx, np.arange(n)[:, None], axis=0)
    inv_idx_T = inv_idx.T
    grad_x = np.take_along_axis(grad_L, inv_idx_T, axis=0)
    
    grad_x_mean = np.mean(grad_x, axis=1)
    
    return area, grad_x_mean

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # 1. Initial guess: Keich construction
    j = np.arange(1, k + 1)
    positions_keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions_keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions_keich, dtype=np.float64)
    
    # 2. Setup quadrature points
    M = 20000
    y_vals = (np.arange(M) + 0.5) / M
    w_vals = (1 - y_vals) / n
    a_vals = (np.arange(n) + 1) / n
    
    # 3. Optimization
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin
    
    best_x = x0.copy()
    best_area, _ = _calc_area_and_grad(best_x, n, y_vals, w_vals, a_vals)
    
    # Phase 1: Smooth surrogate optimization with decreasing beta
    # Start with moderate beta, decrease to get closer to true objective
    beta_schedule = [50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_x = x0.copy()
    for beta in beta_schedule:
        if time.time() > deadline - 30.0:
            break
            
        def smooth_obj_and_grad(x):
            area, grad = _calc_area_and_grad_smooth(x, n, y_vals, w_vals, a_vals, beta)
            return area, grad
        
        try:
            res = minimize(
                smooth_obj_and_grad,
                current_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if res.fun < best_area:
                # Verify with exact area
                exact_area, _ = _calc_area_and_grad(res.x, n, y_vals, w_vals, a_vals)
                if exact_area < best_area:
                    best_area = exact_area
                    best_x = res.x.copy()
                    current_x = res.x.copy()
                else:
                    current_x = res.x.copy()
            else:
                current_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Exact objective polish with tight tolerances
    if time.time() < deadline - 10.0:
        def exact_obj_and_grad(x):
            area, grad = _calc_area_and_grad(x, n, y_vals, w_vals, a_vals)
            return area, grad
        
        try:
            res = minimize(
                exact_obj_and_grad,
                current_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            if res.fun < best_area:
                best_area = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Try a few random restarts if time allows
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(42)
        for restart in range(3):
            if time.time() > deadline - 15.0:
                break
            
            # Perturb from best_x
            perturbation = rng.normal(0, 0.01, size=n)
            x_start = best_x + perturbation
            
            def smooth_obj_and_grad_r(x, beta=200.0):
                area, grad = _calc_area_and_grad_smooth(x, n, y_vals, w_vals, a_vals, beta)
                return area, grad
            
            try:
                res = minimize(
                    smooth_obj_and_grad_r,
                    x_start,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                exact_area, _ = _calc_area_and_grad(res.x, n, y_vals, w_vals, a_vals)
                if exact_area < best_area:
                    best_area = exact_area
                    best_x = res.x.copy()
            except Exception:
                pass
    
    if time.time() > deadline:
        return best_x
        
    return best_x

# EVOLVE-BLOCK-END
