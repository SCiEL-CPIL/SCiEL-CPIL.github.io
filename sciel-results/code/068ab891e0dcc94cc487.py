"""Combined program: Fourier/chirp subspace search + full-dimensional coordinate descent polish."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

N = 400
_DEADLINE = 145.0  # safety margin under 150s wall-clock

def _value(h):
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.fft import dct, idct, fft, ifft

N = 400
_DEADLINE = 145.0

def _value_from_h(h):
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def _smooth_max_value(h, beta=50.0):
    """Smooth surrogate for max(convolve(h,h)) using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    shifted = beta * (conv - max_c)
    log_sum = max_c + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    return abs(2.0 * N * log_sum / (s * s))

def _dct_basis_h(coeffs, K):
    """Construct h from first K DCT-II basis vectors (normalized)."""
    h = np.zeros(N)
    for k in range(min(len(coeffs), K)):
        c = coeffs[k]
        if abs(c) > 1e-12:
            h += c * np.cos(np.pi * k * (np.arange(N) + 0.5) / N)
    return h

def _dft_basis_h(p):
    """Represent h as sum of low-frequency DFT basis functions.
    p = [dc, a1, b1, a2, b2, a3, b3, a4, b4] (9 params)"""
    i = np.arange(N, dtype=float)
    h = np.zeros(N)
    h += p[0]  # DC
    for k in range(1, 5):
        h += p[2*k-1] * np.cos(2*np.pi*k*i/N)
        h += p[2*k] * np.sin(2*np.pi*k*i/N)
    return h

def _chirp(p):
    phase, rate, ac, as_, base = p
    i = np.arange(N, dtype=float)
    t = i / N
    freq = rate * t
    ang = phase + 2.0 * np.pi * np.cumsum(freq)
    h = base + ac * np.cos(ang) + as_ * np.sin(ang)
    return h

def _cubic_chirp(p):
    """Cubic phase chirp: h = base + ac*cos(phase + 2pi*(a*t + b*t^2 + c*t^3)) + as_*sin(...)
    p = [phase, a, b, c, ac, as_, base] (7 params)"""
    phase, a, b, c, ac, as_, base = p
    i = np.arange(N, dtype=float)
    t = i / N
    ang = phase + 2.0 * np.pi * (a * t + b * t**2 + c * t**3)
    h = base + ac * np.cos(ang) + as_ * np.sin(ang)
    return h

def _zadoff_chu_h(p):
    phase, u, base = p
    i = np.arange(N, dtype=float)
    ang = phase - np.pi * u * i * (i + 1) / N
    h = base + np.cos(ang)
    return h

def _two_level(p):
    a, b, frac, offset = p
    k = int(round(frac * N))
    h = np.empty(N)
    h[:k] = a
    h[k:] = b
    h = h + offset * (np.arange(N, dtype=float) / N)
    return h

def _random_h(rng):
    kind = rng.integers(0, 7)
    i = np.arange(N, dtype=float)
    if kind == 0:
        f = rng.uniform(1, 20)
        h = np.cos(2 * np.pi * f * i / N + rng.uniform(0, 2*np.pi))
        h = h + rng.uniform(-0.5, 0.5)
    elif kind == 1:
        a = rng.uniform(-2, 2); b = rng.uniform(-2, 2)
        k = rng.integers(1, N)
        h = np.where(np.arange(N) < k, a, b)
    elif kind == 2:
        rate = rng.uniform(0, 50)
        ang = 2 * np.pi * np.cumsum(rate * i / N) + rng.uniform(0, 2*np.pi)
        h = np.cos(ang) + rng.uniform(-0.3, 0.3)
    elif kind == 3:
        u = rng.integers(1, N)
        phase = rng.uniform(0, 2*np.pi)
        ang = phase - np.pi * u * i * (i + 1) / N
        h = np.cos(ang) + rng.uniform(-0.2, 0.2)
    elif kind == 4:
        h = rng.normal(0, 1, N)
        k = max(1, int(N // 50))
        ker = np.ones(k) / k
        h = np.convolve(h, ker, mode='same')
    elif kind == 5:
        p = rng.uniform(-1, 1, 9)
        h = _dft_basis_h(p)
    else:
        # Cubic chirp random init
        phase = rng.uniform(0, 2*np.pi)
        a = rng.uniform(-10, 10)
        b = rng.uniform(-20, 20)
        c = rng.uniform(-50, 50)
        ac = rng.uniform(0.5, 2.0)
        as_ = rng.uniform(-1.0, 1.0)
        base = rng.uniform(-0.5, 0.5)
        h = _cubic_chirp([phase, a, b, c, ac, as_, base])
    s = np.sum(h)
    if abs(s) < 1e-6:
        h = h + 0.1
    return h

def _surrogate_objective_full(h, tau=0.5):
    """Smooth surrogate for full-dimensional optimization with sum(h)=N normalization."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    h_norm = h * N / s
    conv = np.convolve(h_norm, h_norm)
    m = np.max(conv)
    if m > 1e10:
        return m
    shifted = (conv - m) / tau
    return m + tau * np.log(np.mean(np.exp(shifted)))

def _coordinate_update(h, idx, tau=0.5, max_iter=30):
    """Find optimal value for h[idx] given all other coordinates fixed."""
    h_old = h[idx]
    
    def obj(delta):
        h_new = h.copy()
        h_new[idx] = h_old + delta
        return _surrogate_objective_full(h_new, tau)
    
    try:
        res = minimize_scalar(obj, bounds=(-2.0, 2.0), method='bounded',
                              options={'maxiter': max_iter})
        if res.success:
            return h_old + res.x
    except Exception:
        pass
    return h_old

def construct():
    start = time.time()
    best_h = np.ones(N)
    best_v = _value_from_h(best_h)

    rng = np.random.default_rng(12345)

    def deadline_ok():
        return (time.time() - start) < _DEADLINE

    # Phase 1: random structured search
    iters = 0
    while deadline_ok() and iters < 2000:
        iters += 1
        h = _random_h(rng)
        v = _value_from_h(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    # Phase 2: DCT basis optimization with projected gradient descent
    K = 12  # number of DCT basis functions to use
    
    def _dct_objective(coeffs, beta=50.0):
        h = _dct_basis_h(coeffs, K)
        return _smooth_max_value(h, beta)
    
    def _dct_objective_exact(coeffs):
        h = _dct_basis_h(coeffs, K)
        return _value_from_h(h)
    
    # Initialize DCT coefficients from best_h
    def _get_dct_coeffs(h, K):
        coeffs = np.zeros(K)
        for k in range(K):
            coeffs[k] = 2.0 / N * np.sum(h * np.cos(np.pi * k * (np.arange(N) + 0.5) / N))
        return coeffs
    
    dct_coeffs_init = _get_dct_coeffs(best_h, K)
    
    # Try multiple betas for smooth surrogate
    for beta in [30.0, 60.0, 100.0, 200.0]:
        if not deadline_ok():
            break
        try:
            res_dct = minimize(
                lambda c: _dct_objective(c, beta),
                dct_coeffs_init,
                method='L-BFGS-B',
                bounds=[(-3, 3)] * K,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_dct.x is not None:
                h = _dct_basis_h(res_dct.x, K)
                v = _value_from_h(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
                    dct_coeffs_init = res_dct.x.copy()
        except Exception:
            pass

    # Phase 3: Differential evolution on cubic chirp params
    def cubic_chirp_obj(p):
        h = _cubic_chirp(p)
        return _value_from_h(h)

    bounds_cc = [(-np.pi, np.pi), (-20, 20), (-50, 50), (-100, 100),
                 (-3, 3), (-3, 3), (-2, 2)]
    try:
        if deadline_ok():
            res = differential_evolution(
                cubic_chirp_obj, bounds_cc,
                maxiter=30, popsize=15, seed=7, tol=1e-4,
                polish=False,
            )
            if res.x is not None:
                h = _cubic_chirp(res.x)
                v = _value_from_h(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
            pass

    # Phase 3b: Also try linear chirp DE
    def chirp_obj(p):
        h = _chirp(p)
        return _value_from_h(h)

    bounds_lc = [(-np.pi, np.pi), (-100, 100), (-2, 2), (-2, 2), (-2, 2)]
    try:
        if deadline_ok():
            res_lc = differential_evolution(
                chirp_obj, bounds_lc,
                maxiter=20, popsize=10, seed=7, tol=1e-4,
                polish=False,
            )
            if res_lc.x is not None:
                h = _chirp(res_lc.x)
                v = _value_from_h(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
            pass

    # Phase 4: Differential evolution on Zadoff-Chu params
    def zadoff_obj(p):
        h = _zadoff_chu_h(p)
        return _value_from_h(h)

    bounds_z = [(-np.pi, np.pi), (1, 400), (-1, 1)]
    try:
        if deadline_ok():
            res_z = differential_evolution(
                zadoff_obj, bounds_z,
                maxiter=25, popsize=12, seed=17, tol=1e-4,
                polish=False,
            )
            if res_z.x is not None:
                h = _zadoff_chu_h(res_z.x)
                v = _value_from_h(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
            pass

    # Phase 5: L-BFGS-B refinement using smooth-max surrogate on DFT basis
    def dft_obj_smooth(p, beta=50.0):
        h = _dft_basis_h(p)
        return _smooth_max_value(h, beta)

    i = np.arange(N, dtype=float)
    dc_init = np.mean(best_h)
    a1 = 2.0/N * np.sum(best_h * np.cos(2*np.pi*1*i/N))
    b1 = 2.0/N * np.sum(best_h * np.sin(2*np.pi*1*i/N))
    a2 = 2.0/N * np.sum(best_h * np.cos(2*np.pi*2*i/N))
    b2 = 2.0/N * np.sum(best_h * np.sin(2*np.pi*2*i/N))
    a3 = 2.0/N * np.sum(best_h * np.cos(2*np.pi*3*i/N))
    b3 = 2.0/N * np.sum(best_h * np.sin(2*np.pi*3*i/N))
    a4 = 2.0/N * np.sum(best_h * np.cos(2*np.pi*4*i/N))
    b4 = 2.0/N * np.sum(best_h * np.sin(2*np.pi*4*i/N))
    p_init = np.array([dc_init, a1, b1, a2, b2, a3, b3, a4, b4])

    bounds_dft = [(-3, 3)] + [(-2, 2)] * 8
    try:
        if deadline_ok():
            for beta in [30.0, 60.0, 120.0]:
                if not deadline_ok():
                    break
                res_dft = minimize(
                    lambda p: _smooth_max_value(_dft_basis_h(p), beta),
                    p_init,
                    method='L-BFGS-B',
                    bounds=bounds_dft,
                    options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res_dft.x is not None:
                    h = _dft_basis_h(res_dft.x)
                    v = _value_from_h(h)
                    if v < best_v:
                        best_v = v
                        best_h = h.copy()
    except Exception:
            pass

    # Phase 6: Local Nelder-Mead on 2-level parameterization
    def two_obj(p):
        h = _two_level(p)
        return _value_from_h(h)

    bounds2 = [(-3, 3), (-3, 3), (0.01, 0.99), (-1, 1)]
    try:
        if deadline_ok():
            res2 = differential_evolution(
                two_obj, bounds2,
                maxiter=20, popsize=10, seed=13, tol=1e-4, polish=False,
            )
            if res2.x is not None:
                h = _two_level(res2.x)
                v = _value_from_h(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
            pass

    # Phase 7: Final high-precision DCT polish with exact objective
    if deadline_ok():
        try:
            dct_coeffs_init = _get_dct_coeffs(best_h, K)
            res_dct_exact = minimize(
                _dct_objective_exact,
                dct_coeffs_init,
                method='L-BFGS-B',
                bounds=[(-3, 3)] * K,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dct_exact.x is not None:
                h = _dct_basis_h(res_dct_exact.x, K)
                v = _value_from_h(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass

    # Phase 8: Full-dimensional coordinate descent with smooth surrogate
    # This is the key addition from inspiration 1 - operates in full 400-dim space
    if deadline_ok():
        h_curr = best_h.copy()
        # Normalize to sum = N
        s = np.sum(h_curr)
        if abs(s) > 1e-6:
            h_curr = h_curr * N / s
        
        v_curr = _value_from_h(h_curr)
        
        # Multiple rounds of coordinate descent with decreasing tau
        taus = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
        
        for tau in taus:
            if time.time() > start + _DEADLINE - 15:
                break
            
            # Several passes of coordinate descent
            n_passes = 3
            for p in range(n_passes):
                if time.time() > start + _DEADLINE - 12:
                    break
                
                # Randomize order to avoid systematic bias
                order = rng.permutation(N)
                
                for idx in order:
                    if time.time() > start + _DEADLINE - 10:
                        break
                    
                    new_val = _coordinate_update(h_curr, idx, tau=tau, max_iter=25)
                    h_curr[idx] = new_val
                
                # Check improvement
                v_check = _value_from_h(h_curr)
                if v_check < best_v:
                    best_v = v_check
                    best_h = h_curr.copy()
                v_curr = v_check
        
        # Final fine-tuning with small perturbations
        if time.time() < start + _DEADLINE - 5:
            tau_final = 0.01
            for _ in range(100):
                if time.time() > start + _DEADLINE - 3:
                    break
                
                # Random small perturbation
                idx = rng.integers(0, N)
                h_new = h_curr.copy()
                h_new[idx] += rng.normal(0, 0.1)
                
                # Coordinate update at that index
                new_val = _coordinate_update(h_new, idx, tau=tau_final, max_iter=20)
                h_new[idx] = new_val
                
                v_new = _value_from_h(h_new)
                if v_new < v_curr:
                    v_curr = v_new
                    h_curr = h_new.copy()
                
                if v_curr < best_v:
                    best_v = v_curr
                    best_h = h_curr.copy()

    # Final: ensure validity and return
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
# EVOLVE-BLOCK-END
