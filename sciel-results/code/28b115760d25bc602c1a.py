"""Second autocorrelation inequality - DCT-based parametric optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _h_from_dct(coeffs, num_coeffs=15):
    """Convert DCT coefficients to nonnegative normalized heights."""
    # coeffs: array of length num_coeffs
    # We use a modified DCT-II to reconstruct h
    # h[i] = sum_k coeffs[k] * cos(pi * (2i+1) * k / (2N))
    # Then take absolute value and normalize
    i = np.arange(N)
    k = np.arange(num_coeffs)
    # Build the DCT matrix
    cos_mat = np.cos(np.pi * np.outer(2*i + 1, k) / (2*N))
    h_raw = cos_mat @ coeffs
    # Enforce non-negativity
    h = np.abs(h_raw)
    s = h.sum()
    if s < 1e-300:
        return np.ones(N) / N
    return h / s

def _dct_coeffs_from_h(h, num_coeffs=15):
    """Extract DCT coefficients from heights."""
    i = np.arange(N)
    k = np.arange(num_coeffs)
    cos_mat = np.cos(np.pi * np.outer(2*i + 1, k) / (2*N))
    # Least squares solve
    coeffs, _, _, _ = np.linalg.lstsq(cos_mat, h, rcond=None)
    return coeffs

def _neg_value_dct(coeffs):
    h = _h_from_dct(coeffs)
    return -_compute_value(h)

def _neg_value_dct_grad(coeffs):
    """Objective with numerical gradient."""
    h = _h_from_dct(coeffs)
    val = -_compute_value(h)
    # Numerical gradient
    grad = np.zeros_like(coeffs)
    eps = 1e-7
    for j in range(len(coeffs)):
        c_plus = coeffs.copy()
        c_plus[j] += eps
        c_minus = coeffs.copy()
        c_minus[j] -= eps
        grad[j] = (_neg_value_dct(c_plus) - _neg_value_dct(c_minus)) / (2*eps)
    return val, grad

def _coordinate_descent_dct(coeffs, max_sweeps=20, deadline=None):
    """Coordinate descent in DCT coefficient space."""
    coeffs = np.array(coeffs, dtype=float)
    best_val = -_neg_value_dct(coeffs)
    num_coeffs = len(coeffs)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 3:
            break
        improved = False
        for j in range(num_coeffs):
            if deadline is not None and time.time() > deadline - 3:
                break
            old_val = -_neg_value_dct(coeffs)
            
            # Golden section search for this coefficient
            lo = coeffs[j] - 2.0
            hi = coeffs[j] + 2.0
            
            gr = (np.sqrt(5) - 1) / 2
            a = lo + (1 - gr) * (hi - lo)
            b = lo + gr * (hi - lo)
            
            c_a = coeffs.copy()
            c_b = coeffs.copy()
            c_a[j] = a
            c_b[j] = b
            fa = _neg_value_dct(c_a)
            fb = _neg_value_dct(c_b)
            
            for _ in range(30):
                if deadline is not None and time.time() > deadline - 3:
                    break
                if fa < fb:
                    hi = b
                    b = a
                    fb = fa
                    a = lo + (1 - gr) * (hi - lo)
                    c_a[j] = a
                    fa = _neg_value_dct(c_a)
                else:
                    lo = a
                    a = b
                    fa = fb
                    b = lo + gr * (hi - lo)
                    c_b[j] = b
                    fb = _neg_value_dct(c_b)
            
            best_j = (a + b) / 2
            c_new = coeffs.copy()
            c_new[j] = best_j
            val_new = -_neg_value_dct(c_new)
            
            if val_new > old_val + 1e-12:
                coeffs = c_new
                improved = True
                best_val = val_new
        
        if not improved:
            break
    
    return coeffs, best_val

def _generate_dct_candidates(rng, num_coeffs=15):
    """Generate diverse DCT coefficient initializations."""
    candidates = []
    
    # All zeros (uniform)
    candidates.append(np.zeros(num_coeffs))
    
    # Single dominant coefficient
    for k in range(min(8, num_coeffs)):
        for amp in [0.5, 1.0, 2.0, 5.0]:
            c = np.zeros(num_coeffs)
            c[k] = amp
            candidates.append(c.copy())
            c[k] = -amp
            candidates.append(c.copy())
    
    # Gaussian-like in coefficient space
    for width in [0.5, 1.0, 2.0]:
        for center in [0, 2, 5, 10]:
            for amp in [0.5, 1.0, 2.0]:
                c = np.zeros(num_coeffs)
                for k in range(num_coeffs):
                    c[k] = amp * np.exp(-((k - center) / width)**2)
                candidates.append(c.copy())
    
    # Random
    for _ in range(50):
        c = rng.normal(0, 1.0, num_coeffs)
        candidates.append(c)
    
    # From known good shapes
    xh = np.linspace(-0.5, 0.5, N)
    idx = np.arange(N)
    center = N // 2
    
    shapes = [
        np.exp(-xh**2 * 4),
        np.exp(-xh**2 * 16),
        np.exp(-xh**2 * 64),
        np.maximum(0, 1 - 2*np.abs(xh)),
        np.maximum(0, np.cos(np.pi * xh)),
        np.exp(-((idx - center)**2) / 10),
        np.exp(-((idx - center)**2) / 40),
        1.0 / (1 + (4*xh)**2),
        1.0 / np.cosh(4*xh),
        np.abs(np.sin(2*np.pi*xh)),
        np.abs(np.sin(4*np.pi*xh)),
    ]
    
    for shape in shapes:
        shape = np.maximum(shape, 1e-8)
        shape = shape / shape.sum()
        coeffs = _dct_coeffs_from_h(shape, num_coeffs)
        candidates.append(coeffs)
    
    # Asymmetric
    for asym in [0.3, 0.7, 1.0, 1.5, 2.0]:
        shape = np.exp(-xh**2 * (4 + asym * xh * 4))
        shape = np.maximum(shape, 1e-8)
        shape = shape / shape.sum()
        coeffs = _dct_coeffs_from_h(shape, num_coeffs)
        candidates.append(coeffs)
    
    # Multi-lobe
    for nlobes in [2, 3, 4]:
        for width in [0.1, 0.2, 0.3]:
            h_w = np.zeros(N)
            for k in range(nlobes):
                c = -1 + 2*k/(nlobes-1)
                h_w += np.exp(-(xh-c)**2/(2*width**2))
            h_w = np.maximum(h_w, 1e-8)
            h_w = h_w / h_w.sum()
            coeffs = _dct_coeffs_from_h(h_w, num_coeffs)
            candidates.append(coeffs)
    
    return candidates

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    num_coeffs = 15
    best_coeffs = np.zeros(num_coeffs)
    best_val = _compute_value(_h_from_dct(best_coeffs))
    
    def try_optimize(coeffs0, method='L-BFGS-B', maxiter=500, **kwargs):
        nonlocal best_coeffs, best_val
        if time.time() > deadline - 5:
            return
        try:
            if method == 'L-BFGS-B':
                opts = {'maxiter': maxiter, 'gtol': 1e-8, 'maxfun': 3000}
            elif method == 'Nelder-Mead':
                opts = {'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            elif method == 'Powell':
                opts = {'maxiter': maxiter, 'xtol': 1e-10, 'ftol': 1e-13}
            else:
                opts = {}
            res = minimize(_neg_value_dct, coeffs0, method=method, options=opts)
            if res.fun < -best_val + 1e-15:
                best_val = -res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    def try_coordinate(coeffs0, max_sweeps=20):
        nonlocal best_coeffs, best_val
        if time.time() > deadline - 3:
            return
        try:
            c_new, val_new = _coordinate_descent_dct(coeffs0, max_sweeps=max_sweeps, deadline=deadline)
            if val_new > best_val + 1e-15:
                best_val = val_new
                best_coeffs = c_new
        except Exception:
            pass
    
    rng = np.random.default_rng(50)
    
    # Generate candidates
    candidates = _generate_dct_candidates(rng, num_coeffs)
    
    # Phase 1: L-BFGS-B from diverse candidates
    phase1_deadline = t0 + 45
    for c0 in candidates:
        if time.time() > phase1_deadline:
            break
        try_optimize(c0, 'L-BFGS-B', maxiter=400)
    
    # Phase 2: Coordinate descent on best
    try_coordinate(best_coeffs, max_sweeps=20)
    
    # Phase 3: Powell refinement
    if time.time() < deadline - 30:
        try_optimize(best_coeffs, 'Powell', maxiter=400)
    
    # Phase 4: More random restarts
    phase4_deadline = t0 + 85
    for seed in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]:
        if time.time() > phase4_deadline:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > phase4_deadline:
                break
            c0 = rng2.normal(0, 1.5, num_coeffs)
            try_optimize(c0, 'L-BFGS-B', maxiter=300)
    
    # Phase 5: Coordinate descent again
    try_coordinate(best_coeffs, max_sweeps=20)
    
    # Phase 6: Nelder-Mead polish
    if time.time() < deadline - 20:
        try_optimize(best_coeffs, 'Nelder-Mead', maxiter=500)
    
    # Phase 7: Try different number of coefficients
    phase7_deadline = t0 + 125
    for nc in [10, 12, 18, 20]:
        if time.time() > phase7_deadline:
            break
        rng3 = np.random.default_rng(nc * 100)
        c0 = rng3.normal(0, 1.0, nc)
        local_best_coeffs = c0.copy()
        local_best_val = _compute_value(_h_from_dct(c0, nc))
        
        def _neg_value_dct_nc(c, nc=nc):
            h = _h_from_dct(c, nc)
            return -_compute_value(h)
        
        try:
            res = minimize(_neg_value_dct_nc, c0, method='L-BFGS-B',
                          options={'maxiter': 300, 'gtol': 1e-8, 'maxfun': 2000})
            if res.fun < -local_best_val + 1e-15:
                local_best_val = -res.fun
                local_best_coeffs = res.x.copy()
        except Exception:
            pass
        
        # Convert to num_coeffs if better
        if local_best_val > best_val + 1e-15:
            # Truncate or pad to num_coeffs
            c_mapped = np.zeros(num_coeffs)
            min_len = min(num_coeffs, nc)
            c_mapped[:min_len] = local_best_coeffs[:min_len]
            val_mapped = _compute_value(_h_from_dct(c_mapped, num_coeffs))
            if val_mapped > best_val + 1e-15:
                best_val = val_mapped
                best_coeffs = c_mapped
    
    # Phase 8: Final coordinate descent + Powell
    try_coordinate(best_coeffs, max_sweeps=20)
    if time.time() < deadline - 10:
        try_optimize(best_coeffs, 'Powell', maxiter=300)
    
    # Phase 9: Final NM
    if time.time() < deadline - 5:
        try_optimize(best_coeffs, 'Nelder-Mead', maxiter=400)
    
    # Final coordinate descent
    try_coordinate(best_coeffs, max_sweeps=15)
    
    # Convert to heights
    best_h = _h_from_dct(best_coeffs, num_coeffs)
    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
