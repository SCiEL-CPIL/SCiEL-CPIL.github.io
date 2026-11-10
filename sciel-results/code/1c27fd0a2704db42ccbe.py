# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_length_grad(starts, widths):
    """
    Compute union length of 1-D intervals [starts[i], starts[i]+widths[i]]
    and its subgradient w.r.t. starts.
    """
    n = len(starts)
    if n == 0:
        return 0.0, np.zeros(0)
    
    order = np.argsort(starts)
    s = starts[order]
    e = s + widths[order]
    
    # Compute union length by merging
    total = 0.0
    cs, ce = s[0], e[0]
    for i in range(1, n):
        if s[i] > ce:
            total += ce - cs
            cs, ce = s[i], e[i]
        else:
            ce = max(ce, e[i])
    total += ce - cs
    
    # Compute subgradient
    prefix_max_e = np.maximum.accumulate(e)
    
    # Determine merge group boundaries
    group_starts = [0]
    cur_end = e[0]
    for i in range(1, n):
        if s[i] > cur_end:
            group_starts.append(i)
            cur_end = e[i]
        else:
            cur_end = max(cur_end, e[i])
    
    grad_sorted = np.zeros(n)
    for i in range(n):
        g = group_starts[0]
        for gs in group_starts:
            if gs <= i:
                g = gs
            else:
                break
        
        if i == g:
            left_exp = True
        else:
            left_exp = s[i] > np.max(e[g:i])
        
        if i == 0:
            right_exp = True
        else:
            right_exp = e[i] > prefix_max_e[i-1]
        
        grad_sorted[i] = (-1.0 if left_exp else 0.0) + (1.0 if right_exp else 0.0)
    
    grad = np.zeros(n)
    grad[order] = grad_sorted
    
    return total, grad


def _exact_volume(pos_x, pos_y, N=8, n_samples=200):
    """Exact union volume via product structure: integral of Lx(c)*Ly(c) dc."""
    c_samples = np.linspace(0, 1, n_samples)
    vol = 0.0
    invN = 1.0 / N
    arange_N = np.arange(N)
    widths = np.full(N, invN)
    for c in c_samples:
        x_starts = pos_x + c * arange_N * invN
        y_starts = pos_y + c * arange_N * invN
        lx, _ = _union_length_grad(x_starts, widths)
        ly, _ = _union_length_grad(y_starts, widths)
        vol += lx * ly
    return vol * (1.0 / n_samples)


def _exact_volume_grad(pos_x, pos_y, N=8, n_samples=200):
    """Union volume and gradient w.r.t. pos_x and pos_y."""
    c_samples = np.linspace(0, 1, n_samples)
    vol = 0.0
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    invN = 1.0 / N
    arange_N = np.arange(N)
    widths = np.full(N, invN)
    
    for c in c_samples:
        x_starts = pos_x + c * arange_N * invN
        y_starts = pos_y + c * arange_N * invN
        
        lx, gx = _union_length_grad(x_starts, widths)
        ly, gy = _union_length_grad(y_starts, widths)
        
        vol += lx * ly
        grad_x += gx * ly
        grad_y += lx * gy
    
    vol *= (1.0 / n_samples)
    grad_x *= (1.0 / n_samples)
    grad_y *= (1.0 / n_samples)
    
    return vol, grad_x, grad_y


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 580.0
    
    # Keich binary construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    best_x = keich.copy()
    best_y = keich.copy()
    best_vol = _exact_volume(best_x, best_y, N)
    
    # Gradient-based simulated annealing
    rng = np.random.default_rng(12345)
    x = best_x.copy()
    y = best_y.copy()
    vol = best_vol
    lr = 0.02
    temperature = 0.03
    min_temp = 0.0005
    
    while time.time() < deadline - 180:
        vol_cur, gx, gy = _exact_volume_grad(x, y, N)
        
        x_new = x - lr * gx + rng.normal(0, temperature, N)
        y_new = y - lr * gy + rng.normal(0, temperature, N)
        
        vol_new = _exact_volume(x_new, y_new, N)
        
        if vol_new < vol_cur or (temperature > min_temp and np.random.random() < np.exp(-(vol_new - vol_cur) / temperature)):
            x = x_new
            y = y_new
            vol_cur = vol_new
            if vol_cur < best_vol:
                best_vol = vol_cur
                best_x = x.copy()
                best_y = y.copy()
        
        lr *= 0.9995
        temperature *= 0.9995
        
        if temperature < min_temp:
            break
    
    # Phase 2: Nelder-Mead polish
    def objective(params):
        if time.time() > deadline - 30:
            return best_vol
        return _exact_volume(params[:N], params[N:], N)
    
    options = {'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13}
    
    n_restarts = 0
    while time.time() < deadline - 60:
        scale = max(0.003, 0.15 * best_vol ** 0.5)
        init = np.concatenate([best_x, best_y]) + rng.normal(0, scale, 2 * N)
        result = minimize(objective, init, method='Nelder-Mead', options=options)
        if result.fun < best_vol:
            best_vol = result.fun
            best_x = result.x[:N].copy()
            best_y = result.x[N:].copy()
        n_restarts += 1
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
