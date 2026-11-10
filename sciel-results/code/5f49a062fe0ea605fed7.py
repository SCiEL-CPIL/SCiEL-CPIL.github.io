import numpy as np
import time
from scipy.optimize import minimize

def _calc_area_fast(x, n, y_vals, w_vals, a_vals):
    """
    Calculate area using vectorized operations.
    x: shape (n,)
    """
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    sorted_L = np.sort(L, axis=0)
    d = np.diff(sorted_L, axis=0)
    w_b = w_vals[None, :]
    union_lengths = w_vals + np.sum(np.minimum(d, w_b), axis=0)
    area = np.mean(union_lengths)
    return area

def _calc_area_and_grad(x, n, y_vals, w_vals, a_vals):
    """
    Calculate the area and its gradient with respect to x.
    """
    M = len(y_vals)
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    
    sorted_L = np.sort(L, axis=0)
    idx = np.argsort(L, axis=0)
    
    d = np.diff(sorted_L, axis=0)
    w_b = w_vals[None, :]
    active = d < w_b
    
    min_terms = np.where(active, d, w_b)
    union_lengths = w_vals + np.sum(min_terms, axis=0)
    area = np.mean(union_lengths)
    
    grad_L = np.zeros_like(sorted_L)
    grad_L[1:, :] += active
    grad_L[:-1, :] -= active
    
    grad_x = np.zeros_like(L)
    np.add.at(grad_x, (idx, np.arange(M)[None, :]), grad_L)
    grad_x_mean = np.mean(grad_x, axis=1)
    
    return area, grad_x_mean

def _calc_area_smooth(x, n, y_vals, w_vals, a_vals, epsilon=0.01):
    """
    Calculate a smooth approximation of the union area using soft-min.
    """
    M = len(y_vals)
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    sorted_L = np.sort(L, axis=0)
    d = np.diff(sorted_L, axis=0)
    w_b = w_vals[None, :]
    
    # Soft-min: min(d, w) ≈ -epsilon * log(exp(-d/epsilon) + exp(-w/epsilon))
    a_term = d / epsilon
    b_term = w_b / epsilon
    
    c = np.maximum(a_term, b_term)
    soft_min = -epsilon * (c + np.log(np.exp(-a_term - c) + np.exp(-b_term - c)))
    
    union_lengths = w_vals + soft_min.sum(axis=0)
    area = union_lengths.mean()
    return area

def _calc_area_smooth_grad(x, n, y_vals, w_vals, a_vals, epsilon=0.01):
    """
    Calculate smooth area and its gradient.
    """
    M = len(y_vals)
    L = x[:, None] + a_vals[:, None] * y_vals[None, :]
    sorted_L = np.sort(L, axis=0)
    idx = np.argsort(L, axis=0)
    d = np.diff(sorted_L, axis=0)
    w_b = w_vals[None, :]
    
    a_term = d / epsilon
    b_term = w_b / epsilon
    c = np.maximum(a_term, b_term)
    
    # soft_min = -epsilon * (c + log(exp(-a-c) + exp(-b-c)))
    # d(soft_min)/da = -epsilon * (dc/da - exp(-a-c)/(exp(-a-c)+exp(-b-c)))
    # dc/da = 1 if a > b else 0
    # Let p = exp(-a-c)/(exp(-a-c)+exp(-b-c))
    
    ea = np.exp(-a_term - c)
    eb = np.exp(-b_term - c)
    denom = ea + eb
    p_a = ea / denom  # derivative contribution from a_term
    
    # d(soft_min)/d(a_term) = -epsilon * (1[a>b] - p_a)
    # d(soft_min)/d(b_term) = -epsilon * (1[b>a] - p_b) where p_b = 1 - p_a
    
    # Since d = sorted_L[j+1] - sorted_L[j]
    # d(d)/d(sorted_L[j+1]) = 1, d(d)/d(sorted_L[j]) = -1
    
    # grad_soft_min_wrt_a_term[j,m] = -epsilon * (1[a>b] - p_a[j,m])
    # grad_soft_min_wrt_b_term[j,m] = -epsilon * (1[b>a] - (1-p_a[j,m])) = -epsilon * (p_a[j,m] - 1[b>a])
    
    # But we need grad w.r.t. sorted_L
    # soft_min depends on d and w
    # d(soft_min)/d(sorted_L[j+1]) = d(soft_min)/da * da/d(sorted_L[j+1]) = d(soft_min)/da * 1
    # d(soft_min)/d(sorted_L[j]) = d(soft_min)/da * da/d(sorted_L[j]) = d(soft_min)/da * (-1)
    
    # d(soft_min)/da = -epsilon * (1[a>b] - p_a)
    
    a_b_mask = a_term > b_term
    grad_da = -epsilon * (a_b_mask.astype(float) - p_a)
    
    # Now map back to sorted_L
    grad_L = np.zeros_like(sorted_L)
    grad_L[1:, :] += grad_da
    grad_L[:-1, :] -= grad_da
    
    # Map back to x
    grad_x = np.zeros_like(L)
    np.add.at(grad_x, (idx, np.arange(M)[None, :]), grad_L)
    grad_x_mean = np.mean(grad_x, axis=1)
    
    # Compute area
    soft_min = -epsilon * (c + np.log(denom))
    union_lengths = w_vals + soft_min.sum(axis=0)
    area = union_lengths.mean()
    
    return area, grad_x_mean

def _keich_positions(n=64):
    """Generate Keich construction positions."""
    k = int(np.log2(n))
    positions = np.zeros(n)
    for i in range(n):
        bits = format(i, f'0{k}b')
        eps = np.array([int(b) for b in bits], dtype=float)
        j = np.arange(1, k + 1)
        positions[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return positions

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # 600s limit, use 580 as hard stop
    
    n = 64
    a_vals = (np.arange(n) + 1) / n
    
    # Full quadrature for final evaluation
    M_full = 20000
    y_full = (np.arange(M_full) + 0.5) / M_full
    w_full = (1 - y_full) / n
    
    # Coarser quadrature for optimization (2000 points)
    M_coarse = 2000
    y_coarse = (np.arange(M_coarse) + 0.5) / M_coarse
    w_coarse = (1 - y_coarse) / n
    
    # Initial guess: Keich construction
    x0 = _keich_positions(n)
    
    best_x = x0.copy()
    best_area_full = _calc_area_fast(x0, n, y_full, w_full, a_vals)
    
    # Phase 1: Coarse optimization with gradient-based method
    np.random.seed(42)
    best_coarse_x = x0.copy()
    best_coarse_area = _calc_area_fast(x0, n, y_coarse, w_coarse, a_vals)
    
    n_restarts = 6
    for restart in range(n_restarts):
        if time.time() > deadline - 120:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.normal(0, 0.05, n)
        
        try:
            res = minimize(
                lambda x: _calc_area_and_grad(x, n, y_coarse, w_coarse, a_vals),
                x_start,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.fun < best_coarse_area:
                best_coarse_area = res.fun
                best_coarse_x = res.x.copy()
        except Exception:
            pass
    
    # Evaluate best coarse solution on full grid
    area_full = _calc_area_fast(best_coarse_x, n, y_full, w_full, a_vals)
    if area_full < best_area_full:
        best_x = best_coarse_x
        best_area_full = area_full
    
    # Phase 2: Smooth surrogate optimization to escape local minima
    if time.time() < deadline - 60:
        epsilons = [0.02, 0.01, 0.005, 0.002]
        for eps in epsilons:
            if time.time() > deadline - 60:
                break
            try:
                res = minimize(
                    lambda x: _calc_area_smooth_grad(x, n, y_coarse, w_coarse, a_vals, epsilon=eps),
                    best_x,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.fun < best_area_full:
                    # Verify on full grid
                    area_full = _calc_area_fast(res.x, n, y_full, w_full, a_vals)
                    if area_full < best_area_full:
                        best_x = res.x.copy()
                        best_area_full = area_full
            except Exception:
                pass
    
    # Phase 3: Fine-tune on full grid with exact gradient
    if time.time() < deadline - 30:
        try:
            res = minimize(
                lambda x: _calc_area_and_grad(x, n, y_full, w_full, a_vals),
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_area_full:
                best_x = res.x
                best_area_full = res.fun
        except Exception:
            pass
    
    # Phase 4: Local refinement with small perturbations
    if time.time() < deadline - 15:
        current_x = best_x.copy()
        current_area = best_area_full
        step_size = 0.005
        
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            
            perturbation = np.random.normal(0, step_size, n)
            new_x = current_x + perturbation
            new_area = _calc_area_fast(new_x, n, y_full, w_full, a_vals)
            
            if new_area < current_area:
                current_x = new_x
                current_area = new_area
                step_size *= 0.95
            else:
                step_size *= 0.85
            
            if step_size < 1e-7:
                break
        
        if current_area < best_area_full:
            best_x = current_x
            best_area_full = current_area
    
    return np.array(best_x, dtype=np.float64)
