import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 50
DELTA = 1.0 / N

# Precompute the autoconvolution overlap matrices
# M[k, i, j] = overlap length of interval [i, i+1] and [k-j, k-j+1] in units of DELTA
def _build_overlap_matrices():
    num_lags = 2 * N - 1
    M = np.zeros((num_lags, N, N))
    for k in range(num_lags):
        for i in range(N):
            for j in range(N):
                lo = max(i, k - j)
                hi = min(i + 1, k - j + 1)
                M[k, i, j] = max(0.0, hi - lo)
    return M

_M_matrices = _build_overlap_matrices()
_NUM_LAGS = 2 * N - 1

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution using precomputed matrices."""
    h = np.maximum(h, 0)
    ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
    return np.max(ff_vals)

def _autoconv_all(h):
    """Compute all autoconvolution values."""
    h = np.maximum(h, 0)
    ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
    return ff_vals

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _smooth_autoconv_sup(h, temperature=0.01):
    """Smooth approximation of max using log-sum-exp."""
    h = np.maximum(h, 0)
    ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
    max_val = np.max(ff_vals)
    if max_val < 1e-30:
        return 0.0
    shifted = ff_vals - max_val
    log_sum = max_val + temperature * np.log(np.sum(np.exp(shifted / temperature)))
    return log_sum

def _smooth_ratio(h, temperature=0.01):
    """Smooth version of the ratio for gradient-based optimization."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _smooth_autoconv_sup(h, temperature)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def generate_mian_chowla(n, max_size=None):
    """Generate a Mian-Chowla (Sidon) sequence."""
    if max_size is None:
        max_size = n
    seq = [1]
    used_sums = {2}
    for candidate in range(2, n + 1):
        valid = True
        for i in seq:
            if i + candidate in used_sums:
                valid = False
                break
        if valid:
            seq.append(candidate)
            used_sums.add(candidate + candidate)
            for i in seq[:-1]:
                used_sums.add(i + candidate)
        if len(seq) >= max_size:
            break
    return [x - 1 for x in seq]

def _multiplicative_update(h, max_iter=500, tol=1e-8, temperature=0.05):
    """Sinkhorn-Knopp style multiplicative updates for the min-max game."""
    h = np.maximum(h, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    p = np.ones(_NUM_LAGS) / _NUM_LAGS
    
    for iteration in range(max_iter):
        Q = np.array([h @ _M_matrices[k] @ h for k in range(_NUM_LAGS)])
        
        max_Q = np.max(Q)
        exp_Q = np.exp((Q - max_Q) / temperature)
        p = exp_Q / np.sum(exp_Q)
        
        M_eff = np.sum([p[k] * _M_matrices[k] for k in range(_NUM_LAGS)], axis=0)
        
        Mh = M_eff @ h
        Mh = np.maximum(Mh, 1e-15)
        h_new = h / np.sqrt(Mh)
        s_new = np.sum(h_new)
        if s_new < 1e-15:
            break
        h_new = h_new / s_new
        
        if np.max(np.abs(h_new - h)) < tol:
            h = h_new
            break
        h = h_new
    
    return h

def _epigraph_optimize(h0, max_iter=300, temperature=0.01):
    """Epigraph formulation using scipy minimize with SLSQP on the smooth ratio."""
    h = np.maximum(h0, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    def neg_smooth_ratio(x):
        x = np.maximum(x, 0)
        s = np.sum(x)
        if s < 1e-15:
            return 0.0
        return -_smooth_ratio(x / s, temperature)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(np.maximum(x, 0)) - 1.0}]
    bounds = [(0, None)] * N
    
    try:
        result = minimize(neg_smooth_ratio, h.copy(), method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': max_iter, 'ftol': 1e-12})
        h_candidate = np.maximum(result.x, 0)
        s = np.sum(h_candidate)
        if s > 1e-15:
            h_candidate = h_candidate / s
        return h_candidate
    except Exception:
        return h

def _coordinate_descent_polish(h, max_iter=1000, tol=1e-10):
    """Coordinate descent with exact line search for each coordinate."""
    h = np.maximum(h, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            h_old_i = h[i]
            best_ratio = _ratio(h)
            
            lo = 0.0
            hi = max(1.0, 3.0 * h_old_i)
            gr = (np.sqrt(5) - 1) / 2
            
            for _ in range(20):
                x1 = hi - gr * (hi - lo)
                x2 = lo + gr * (hi - lo)
                
                h[i] = x1
                r1 = _ratio(h)
                h[i] = x2
                r2 = _ratio(h)
                
                if r1 > r2:
                    hi = x2
                else:
                    lo = x1
            
            best_h_i = (lo + hi) / 2
            h[i] = best_h_i
            r_new = _ratio(h)
            
            if r_new > best_ratio + 1e-12:
                improved = True
            else:
                h[i] = h_old_i
        
        if not improved:
            break
    
    s = np.sum(h)
    if s > 1e-15:
        h = h / s
    return h

def _iterative_reweighting(h, max_iter=200, temperature=0.05):
    """Iterative reweighting: alternate between computing the worst lag and adjusting heights."""
    h = np.maximum(h, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    for iteration in range(max_iter):
        ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
        
        max_ff = np.max(ff_vals)
        if max_ff < 1e-15:
            break
        
        worst_k = np.argmax(ff_vals)
        M_worst = _M_matrices[worst_k]
        pressure = M_worst @ h
        pressure = np.maximum(pressure, 1e-15)
        
        h_new = h / np.sqrt(pressure)
        s_new = np.sum(h_new)
        if s_new < 1e-15:
            break
        h_new = h_new / s_new
        
        if np.max(np.abs(h_new - h)) < 1e-10:
            h = h_new
            break
        h = h_new
    
    return h

def _gaussian_parametric_optimize(h_init, num_gaussians=3, max_iter=200, temperature=0.02):
    """
    Model the height vector as the pointwise maximum of a small set of shifted, 
    scaled Gaussian basis functions. Treat centers and variances as continuous 
    parameters to be optimized. This parametric representation naturally enforces 
    non-negativity and smoothness, allowing standard gradient-based optimizers 
    to navigate the shape space.
    """
    h_init = np.maximum(h_init, 0).copy()
    s = np.sum(h_init)
    if s < 1e-15:
        return h_init
    h_init = h_init / s
    
    x = np.linspace(0, N - 1, N)
    
    # Initialize parameters: for each Gaussian, (amplitude, center, width)
    # We'll use a small number of Gaussians
    # Initial centers spread across the domain, widths moderate
    centers = np.linspace(0.2 * (N - 1), 0.8 * (N - 1), num_gaussians)
    widths = np.full(num_gaussians, 3.0)
    amplitudes = np.full(num_gaussians, 0.5)
    
    # Flatten parameters
    def params_to_h(params):
        h_gauss = np.zeros(N)
        for i in range(num_gaussians):
            amp = params[3*i]
            center = params[3*i + 1]
            width = params[3*i + 2]
            # Ensure width is positive
            width = max(abs(width), 0.5)
            g = amp * np.exp(-((x - center)**2) / (2 * width**2))
            h_gauss = np.maximum(h_gauss, g)
        return h_gauss
    
    def neg_ratio_params(params):
        h = params_to_h(params)
        s = np.sum(h)
        if s < 1e-15:
            return 0.0
        h_norm = h / s
        return -_smooth_ratio(h_norm, temperature)
    
    # Initial parameter vector
    init_params = np.zeros(num_gaussians * 3)
    for i in range(num_gaussians):
        init_params[3*i] = amplitudes[i]
        init_params[3*i + 1] = centers[i]
        init_params[3*i + 2] = widths[i]
    
    bounds = []
    for i in range(num_gaussians):
        bounds.append((0.01, 5.0))  # amplitude
        bounds.append((0.0, N - 1))  # center
        bounds.append((0.5, 10.0))   # width
    
    try:
        result = minimize(neg_ratio_params, init_params, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        h_candidate = params_to_h(result.x)
        s = np.sum(h_candidate)
        if s > 1e-15:
            h_candidate = h_candidate / s
            return h_candidate
        else:
            return h_init
    except Exception:
        return h_init

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    np.random.seed(42)
    
    # Generate initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Mian-Chowla / Sidon sets
    for size in [8, 10, 12, 14, 16]:
        mc = generate_mian_chowla(N, max_size=size)
        h_mc = np.zeros(N)
        for idx in mc:
            if 0 <= idx < N:
                h_mc[idx] = 1.0
        if np.sum(h_mc) > 0:
            initializations.append(h_mc)
    
    # 3. Sparse random
    for trial in range(15):
        h_sparse = np.zeros(N)
        num_ones = np.random.randint(5, 18)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sparse[indices] = np.random.uniform(0.3, 2.0, num_ones)
        initializations.append(h_sparse)
    
    # 4. Structured patterns
    t = np.linspace(-1, 1, N)
    initializations.append(np.maximum(1 - t**2, 0.01))
    initializations.append(np.maximum(t**2 + 0.1, 0.01))
    
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)
    
    h_sp3 = np.zeros(N)
    h_sp3[::3] = 1.0
    initializations.append(h_sp3)
    
    initializations.append(np.exp(-t**2 / 0.3))
    initializations.append(np.maximum(1.0 + 2.0 * np.exp(-((t - 0.6)**2) / 0.1) + 2.0 * np.exp(-((t + 0.6)**2) / 0.1), 0.05))
    
    # 5. Single spike
    h_spike = np.zeros(N)
    h_spike[N // 2] = 1.0
    initializations.append(h_spike)
    
    # 6. Two spikes
    for offset in [5, 10, 15]:
        h_two = np.zeros(N)
        h_two[N // 2 - offset] = 1.0
        h_two[N // 2 + offset] = 1.0
        initializations.append(h_two)
    
    # Optimize each initialization
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        # Step 1: Multiplicative updates (Sinkhorn-Knopp style)
        h_mu = _multiplicative_update(h0.copy(), max_iter=300, tol=1e-9, temperature=0.05)
        sc = _ratio(h_mu)
        if sc > best_score:
            best_score = sc
            best_h = h_mu.copy()
        
        if time.time() - start_time > DEADLINE - 10:
            continue
        
        # Step 2: Iterative reweighting
        h_ir = _iterative_reweighting(h_mu.copy(), max_iter=200)
        sc = _ratio(h_ir)
        if sc > best_score:
            best_score = sc
            best_h = h_ir.copy()
        
        if time.time() - start_time > DEADLINE - 10:
            continue
        
        # Step 3: Smooth epigraph optimization
        h_ep = _epigraph_optimize(h_ir.copy(), max_iter=200, temperature=0.02)
        sc = _ratio(h_ep)
        if sc > best_score:
            best_score = sc
            best_h = h_ep.copy()
        
        if time.time() - start_time > DEADLINE - 10:
            continue
        
        # Step 4: Gaussian parametric optimization (NEW)
        h_gauss = _gaussian_parametric_optimize(h_ep.copy(), num_gaussians=3, max_iter=150, temperature=0.02)
        sc = _ratio(h_gauss)
        if sc > best_score:
            best_score = sc
            best_h = h_gauss.copy()
        
        if time.time() - start_time > DEADLINE - 10:
            continue
        
        # Step 5: Coordinate descent polish
        h_cd = _coordinate_descent_polish(h_gauss.copy(), max_iter=500, tol=1e-10)
        sc = _ratio(h_cd)
        if sc > best_score:
            best_score = sc
            best_h = h_cd.copy()
    
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 15:
        try:
            h_final = _coordinate_descent_polish(best_h.copy(), max_iter=2000, tol=1e-12)
            sc = _ratio(h_final)
            if sc > best_score:
                best_score = sc
                best_h = h_final.copy()
        except Exception:
            pass
    
    # One more multiplicative update pass on the best
    if time.time() - start_time < DEADLINE - 10:
        try:
            h_mu2 = _multiplicative_update(best_h.copy(), max_iter=500, tol=1e-10, temperature=0.03)
            sc = _ratio(h_mu2)
            if sc > best_score:
                best_score = sc
                best_h = h_mu2.copy()
            
            h_cd2 = _coordinate_descent_polish(best_h.copy(), max_iter=1000, tol=1e-12)
            sc = _ratio(h_cd2)
            if sc > best_score:
                best_score = sc
                best_h = h_cd2.copy()
        except Exception:
            pass
    
    return best_h.tolist()
