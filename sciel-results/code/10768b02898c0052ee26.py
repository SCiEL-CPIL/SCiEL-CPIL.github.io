import numpy as np
import time
from scipy.optimize import minimize

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


def _pure_tone_to_h(params):
    """
    Convert pure tone parameters to step heights.
    params = [freq, phase, amp, offset]
    h_i = offset + amp * cos(2*pi*freq*i/N + phase)
    """
    freq = params[0]
    phase = params[1]
    amp = params[2]
    offset = params[3]
    
    i = np.arange(N)
    h = offset + amp * np.cos(2.0 * np.pi * freq * i / N + phase)
    return h


def _pure_tone_score(params):
    """
    Score of the pure tone parameterization.
    """
    h = _pure_tone_to_h(params)
    h = _fix_sum(h)
    return _score(h)


def _pure_tone_grad(params, beta):
    """
    Compute gradient of smooth max w.r.t. pure tone parameters.
    """
    h = _pure_tone_to_h(params)
    h_fixed = _fix_sum(h)
    
    _, grad_h = _smooth_max_grad(h_fixed, beta)
    
    freq = params[0]
    phase = params[1]
    amp = params[2]
    offset = params[3]
    
    i = np.arange(N)
    base = 2.0 * np.pi * freq * i / N + phase
    
    grad = np.zeros(4)
    grad[0] = np.dot(grad_h, -amp * (2.0 * np.pi * i / N) * np.sin(base))
    grad[1] = np.dot(grad_h, -amp * np.sin(base))
    grad[2] = np.dot(grad_h, np.cos(base))
    grad[3] = np.dot(grad_h, np.ones(N))
    
    val, _ = _smooth_max_grad(h_fixed, beta)
    return val, grad


def _two_tone_to_h(params):
    """
    Convert two tone parameters to step heights.
    params = [f1, p1, a1, f2, p2, a2, offset]
    """
    i = np.arange(N)
    h = params[6]  # offset
    h += params[2] * np.cos(2.0 * np.pi * params[0] * i / N + params[1])
    h += params[5] * np.cos(2.0 * np.pi * params[3] * i / N + params[4])
    return h


def _two_tone_score(params):
    """
    Score of the two tone parameterization.
    """
    h = _two_tone_to_h(params)
    h = _fix_sum(h)
    return _score(h)


def _two_tone_grad(params, beta):
    """
    Compute gradient of smooth max w.r.t. two tone parameters.
    """
    h = _two_tone_to_h(params)
    h_fixed = _fix_sum(h)
    
    _, grad_h = _smooth_max_grad(h_fixed, beta)
    
    i = np.arange(N)
    f1, p1, a1 = params[0], params[1], params[2]
    f2, p2, a2 = params[3], params[4], params[5]
    
    base1 = 2.0 * np.pi * f1 * i / N + p1
    base2 = 2.0 * np.pi * f2 * i / N + p2
    
    grad = np.zeros(7)
    grad[0] = np.dot(grad_h, -a1 * (2.0 * np.pi * i / N) * np.sin(base1))
    grad[1] = np.dot(grad_h, -a1 * np.sin(base1))
    grad[2] = np.dot(grad_h, np.cos(base1))
    grad[3] = np.dot(grad_h, -a2 * (2.0 * np.pi * i / N) * np.sin(base2))
    grad[4] = np.dot(grad_h, -a2 * np.sin(base2))
    grad[5] = np.dot(grad_h, np.cos(base2))
    grad[6] = np.dot(grad_h, np.ones(N))
    
    val, _ = _smooth_max_grad(h_fixed, beta)
    return val, grad


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
                    
                    # Evaluate
                    h_tone = _pure_tone_to_h(params)
                    h_tone = _fix_sum(h_tone)
                    s = _score(h_tone)
                    
                    if s < best_score:
                        _update_best(h_tone)
                        
                        # Local refinement with gradient
                        if time.time() - t0 > _DEADLINE - 8:
                            break
                            
                        try:
                            def obj(p):
                                return _pure_tone_grad(p, 50.0)[0]
                            
                            def jac(p):
                                return _pure_tone_grad(p, 50.0)[1]
                            
                            res = minimize(obj, params.copy(), jac=jac, method='L-BFGS-B',
                                           bounds=[(0.5, 15), (0, 2*np.pi), (0.1, 0.8), (0.3, 0.7)],
                                           options={'maxiter': 100, 'ftol': 1e-12})
                            if res.fun < best_score:
                                h_refined = _pure_tone_to_h(res.x)
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
                                
                                params = np.array([float(f1), p1, a1, float(f2), p2, a2, 0.5])
                                h_two = _two_tone_to_h(params)
                                h_two = _fix_sum(h_two)
                                s = _score(h_two)
                                
                                if s < best_score:
                                    _update_best(h_two)
                                    
                                    # Local refinement
                                    if time.time() - t0 > _DEADLINE - 3:
                                        break
                                        
                                    try:
                                        def obj2(p):
                                            return _two_tone_grad(p, 50.0)[0]
                                        
                                        def jac2(p):
                                            return _two_tone_grad(p, 50.0)[1]
                                        
                                        res = minimize(obj2, params.copy(), jac=jac2, method='L-BFGS-B',
                                                       bounds=[(0.5, 5), (0, 2*np.pi), (0.05, 0.5),
                                                              (2, 10), (0, 2*np.pi), (0.05, 0.5),
                                                              (0.3, 0.7)],
                                                       options={'maxiter': 80, 'ftol': 1e-12})
                                        if res.fun < best_score:
                                            h_refined = _two_tone_to_h(res.x)
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
