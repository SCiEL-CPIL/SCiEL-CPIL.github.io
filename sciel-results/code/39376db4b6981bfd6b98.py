import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    Vectorized gradient computation using FFT.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    # Compute gradient using FFT-based convolution
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # conv1[j] = sum_k w[k] * h[j-k] (linear convolution)
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # conv2[j] = sum_k w[k] * h[j+k]
    v = h[::-1]
    V = np.fft.rfft(v, n=n_fft)
    conv2_rev = np.fft.irfft(W * V, n=n_fft)[:n]
    conv2 = conv2_rev[n-1::-1]
    
    # Combine
    grad = -(2.0 / n) * (conv1 + conv2)
    
    return val, grad


def _spectral_search(h0, time_budget):
    """
    Spectral search in the frequency domain.
    We parameterize h in terms of its DFT coefficients.
    h_i = (1/N) * sum_k H_k * exp(2*pi*i*k*i/N)
    
    The overlap values are related to the autocorrelation of h.
    We search over different spectral configurations.
    """
    t_start = time.time()
    n = len(h0)
    n_fft = 2 * n
    
    # Get the DFT of h0
    H0 = np.fft.rfft(h0, n=n_fft)
    
    # We'll search over scaling and phase shifts in frequency domain
    best_h = h0.copy()
    best_score = _score(h0)
    
    # Try different spectral modifications
    # 1. Scale high frequencies
    for scale_factor in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5]:
        if time.time() - t_start > time_budget:
            break
        
        H_mod = H0.copy()
        # Scale frequencies above 1
        for k in range(2, len(H_mod)):
            H_mod[k] *= scale_factor
        
        h_mod = np.fft.irfft(H_mod, n=n_fft)[:n]
        h_mod = _fix_sum(h_mod)
        s = _score(h_mod)
        
        if s < best_score:
            best_score = s
            best_h = h_mod.copy()
    
    # 2. Try phase rotations in frequency domain
    for phase_shift in np.linspace(0, 2*np.pi, 12, endpoint=False):
        if time.time() - t_start > time_budget:
            break
        
        H_mod = H0.copy()
        # Apply phase rotation to higher frequencies
        for k in range(2, len(H_mod)):
            H_mod[k] *= np.exp(1j * phase_shift * (k - 1))
        
        h_mod = np.fft.irfft(H_mod, n=n_fft)[:n]
        h_mod = _fix_sum(h_mod)
        s = _score(h_mod)
        
        if s < best_score:
            best_score = s
            best_h = h_mod.copy()
    
    return best_h, best_score


def _lp_spectral_optimization(h0, time_budget):
    """
    Use linear programming to optimize spectral coefficients.
    This is a heuristic approach - we approximate the problem in the frequency domain.
    """
    t_start = time.time()
    n = len(h0)
    n_fft = 2 * n
    
    # Get the DFT
    H = np.fft.rfft(h0, n=n_fft)
    
    # We'll try to minimize the max overlap by adjusting the spectral shape
    # This is a simplified approach - full LP in frequency domain is complex
    
    # Try different spectral profiles
    best_h = h0.copy()
    best_score = _score(h0)
    
    # Create basis spectral profiles
    k = np.arange(len(H))
    
    # 1. Low-pass filter
    for cutoff in [3, 5, 8, 12, 16]:
        if time.time() - t_start > time_budget:
            break
        
        H_lp = H.copy()
        H_lp[cutoff:] *= 0.1
        h_lp = np.fft.irfft(H_lp, n=n_fft)[:n]
        h_lp = _fix_sum(h_lp)
        s = _score(h_lp)
        
        if s < best_score:
            best_score = s
            best_h = h_lp.copy()
    
    # 2. Band-pass filter
    for low, high in [(2, 8), (3, 10), (4, 12), (2, 6), (5, 15)]:
        if time.time() - t_start > time_budget:
            break
        
        H_bp = H.copy()
        H_bp[:low] *= 0.1
        H_bp[high:] *= 0.1
        h_bp = np.fft.irfft(H_bp, n=n_fft)[:n]
        h_bp = _fix_sum(h_bp)
        s = _score(h_bp)
        
        if s < best_score:
            best_score = s
            best_h = h_bp.copy()
    
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Multiple structured initial guesses
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Bump function: high in middle, low at edges
    h_bump = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    h_bump = _fix_sum(h_bump)
    _update_best(h_bump)
    
    # Step-like: 1s in center, 0s at edges
    h_step = np.zeros(N)
    h_step[N//4:3*N//4] = 1.0
    h_step = _fix_sum(h_step)
    _update_best(h_step)
    
    # Uniform
    h_unif = np.full(N, 0.5)
    _update_best(h_unif)
    
    # Square wave (high freq)
    h_sq = np.where((i % (N//4)) < (N//8), 1.0, 0.0)
    h_sq = _fix_sum(h_sq)
    _update_best(h_sq)
    
    # Sine wave (high freq)
    h_sin = 0.5 + 0.5 * np.sin(4.0 * np.pi * i / N)
    h_sin = _fix_sum(h_sin)
    _update_best(h_sin)
    
    # Start from the best initial guess
    h0 = best_h.copy()
    
    def _run_optimization(h0, beta, maxiter, ftol, method='SLSQP'):
        nonlocal best_h, best_score
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            val, _ = _smooth_max_grad(x, beta)
            return val
            
        def jac_func(x):
            _, grad = _smooth_max_grad(x, beta)
            return grad
            
        try:
            res = minimize(obj_func, h0.copy(), jac=jac_func, method=method,
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    # Phase 1: Coarse optimization with low beta
    if time.time() - t0 < _DEADLINE - 50:
        _run_optimization(h0, beta=10.0, maxiter=80, ftol=1e-5)
        
    # Phase 2: Medium optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 35:
        _run_optimization(best_h, beta=30.0, maxiter=120, ftol=1e-7)
        
    # Phase 3: Higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        _run_optimization(best_h, beta=60.0, maxiter=150, ftol=1e-9)
        
    # Phase 4: Fine optimization with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        _run_optimization(best_h, beta=100.0, maxiter=200, ftol=1e-11)
        
    # Phase 5: Final polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _run_optimization(best_h, beta=150.0, maxiter=250, ftol=1e-13)
    
    # Spectral search
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_spec, s_spec = _spectral_search(best_h, time_budget=15)
        if s_spec < best_score:
            _update_best(h_spec)
            # Refine the spectral result
            if time.time() - t0 < _DEADLINE - 10:
                _run_optimization(h_spec, beta=80.0, maxiter=100, ftol=1e-10)
    
    # LP-based spectral optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        h_lp, s_lp = _lp_spectral_optimization(best_h, time_budget=10)
        if s_lp < best_score:
            _update_best(h_lp)
            # Refine the LP result
            if time.time() - t0 < _DEADLINE - 5:
                _run_optimization(h_lp, beta=100.0, maxiter=150, ftol=1e-11)
    
    # Pure tone spectral search
    if time.time() - t0 < _DEADLINE - 40:
        # Try different frequencies and phases
        freq_candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        for freq in freq_candidates:
            if time.time() - t0 > _DEADLINE - 20:
                break
            
            # Try multiple phases and amplitudes
            for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
                if time.time() - t0 > _DEADLINE - 15:
                    break
                
                for amp in [0.3, 0.4, 0.5, 0.6]:
                    if time.time() - t0 > _DEADLINE - 10:
                        break
                    
                    params = np.array([float(freq), phase, amp, 0.5])
                    i_arr = np.arange(N)
                    h_tone = params[3] + amp * np.cos(2.0 * np.pi * params[0] * i_arr / N + params[1])
                    h_tone = _fix_sum(h_tone)
                    s = _score(h_tone)
                    
                    if s < best_score:
                        _update_best(h_tone)
                        
                        # Local refinement with gradient
                        if time.time() - t0 > _DEADLINE - 8:
                            break
                            
                        try:
                            def obj(p):
                                h_p = p[3] + p[2] * np.cos(2.0 * np.pi * p[0] * np.arange(N) / N + p[1])
                                h_p = _fix_sum(h_p)
                                val, _ = _smooth_max_grad(h_p, 50.0)
                                return val
                            
                            res = minimize(obj, params.copy(), method='L-BFGS-B',
                                           bounds=[(0.5, 15), (0, 2*np.pi), (0.1, 0.8), (0.3, 0.7)],
                                           options={'maxiter': 100, 'ftol': 1e-12})
                            if res.fun < best_score:
                                h_refined = res.x[3] + res.x[2] * np.cos(2.0 * np.pi * res.x[0] * np.arange(N) / N + res.x[1])
                                _update_best(h_refined)
                        except Exception:
                            pass
    
    # Two-tone spectral search
    if time.time() - t0 < _DEADLINE - 25:
        # Try combinations of two frequencies
        for f1 in [1, 2, 3]:
            if time.time() - t0 > _DEADLINE - 12:
                break
            for f2 in [4, 5, 6, 7]:
                if time.time() - t0 > _DEADLINE - 10:
                    break
                
                # Try a few phase combinations
                for p1 in [0, np.pi/2, np.pi]:
                    if time.time() - t0 > _DEADLINE - 8:
                        break
                    for p2 in [0, np.pi/2, np.pi]:
                        if time.time() - t0 > _DEADLINE - 6:
                            break
                        
                        for a1 in [0.2, 0.3]:
                            if time.time() - t0 > _DEADLINE - 5:
                                break
                            for a2 in [0.1, 0.2]:
                                if time.time() - t0 > _DEADLINE - 4:
                                    break
                                
                                i_arr = np.arange(N)
                                h_two = 0.5
                                h_two += a1 * np.cos(2.0 * np.pi * f1 * i_arr / N + p1)
                                h_two += a2 * np.cos(2.0 * np.pi * f2 * i_arr / N + p2)
                                h_two = _fix_sum(h_two)
                                s = _score(h_two)
                                
                                if s < best_score:
                                    _update_best(h_two)
                                    
                                    # Local refinement
                                    if time.time() - t0 > _DEADLINE - 3:
                                        break
                                        
                                    try:
                                        def obj2(p):
                                            i_arr = np.arange(N)
                                            h_p = p[6]
                                            h_p += p[2] * np.cos(2.0 * np.pi * p[0] * i_arr / N + p[1])
                                            h_p += p[5] * np.cos(2.0 * np.pi * p[3] * i_arr / N + p[4])
                                            h_p = _fix_sum(h_p)
                                            val, _ = _smooth_max_grad(h_p, 50.0)
                                            return val
                                        
                                        res = minimize(obj2, np.array([float(f1), p1, a1, float(f2), p2, a2, 0.5]),
                                                       method='L-BFGS-B',
                                                       bounds=[(0.5, 5), (0, 2*np.pi), (0.05, 0.5),
                                                              (2, 10), (0, 2*np.pi), (0.05, 0.5),
                                                              (0.3, 0.7)],
                                                       options={'maxiter': 80, 'ftol': 1e-12})
                                        if res.fun < best_score:
                                            i_arr = np.arange(N)
                                            h_refined = res.x[6]
                                            h_refined += res.x[2] * np.cos(2.0 * np.pi * res.x[0] * i_arr / N + res.x[1])
                                            h_refined += res.x[5] * np.cos(2.0 * np.pi * res.x[3] * i_arr / N + res.x[4])
                                            _update_best(h_refined)
                                    except Exception:
                                        pass
    
    # Random restarts if time permits
    if time.time() - t0 < _DEADLINE - 25:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            if time.time() - t0 > _DEADLINE - 12:
                break
            _run_optimization(h_rand, beta=30.0, maxiter=60, ftol=1e-7)
            if best_h is not None and time.time() - t0 < _DEADLINE - 8:
                _run_optimization(best_h, beta=80.0, maxiter=100, ftol=1e-10)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
