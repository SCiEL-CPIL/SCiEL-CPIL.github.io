"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses epigraph formulation with SLSQP, refined with higher precision polishes.
Combines cosine initialization with random restarts and adaptive beta tuning."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill


def _autocorr(h):
    """
    Compute linear autocorrelation r_k = sum_{i=k}^{N-1} h_i * h_{i-k} for k=0..N-1.
    Uses FFT for O(N log N) computation.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    return corr[:n]


def _overlap_values(h):
    """
    Compute overlap values C_k = (2/N) * (N/2 - r_k) for k=0..N-1.
    """
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    return float(np.max(cs))


def _smooth_max_value(h, beta):
    """
    Smooth maximum of overlap values using log-sum-exp.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    m = np.max(cs)
    return m + np.log(np.sum(np.exp(beta * (cs - m)))) / beta


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    Vectorized: computes all k-shifts simultaneously via cumsum tricks.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for soft max
    
    val = m + np.log(sum_exp) / beta
    
    # Vectorized gradient computation:
    # dC_k/dh_j = -(2/n) * [h_{j-k} (if j>=k) + h_{j+k} (if j+k<n)]
    # For each j, grad[j] = -(2/n) * sum_k w_k * (h_{j-k} + h_{j+k})
    # = -(2/n) * [sum_{k: k<=j} w_k * h_{j-k} + sum_{k: k<n-j} w_k * h_{j+k}]
    # First term: for fixed j, sum over k=0..j of w_k * h_{j-k} = (w * h)_j (convolution)
    # Second term: for fixed j, sum over k=0..n-1-j of w_k * h_{j+k} = (w * h_flipped)_{n-1-j}
    
    # Compute linear convolution w * h (length 2n-1, take first n)
    # Use FFT for efficiency
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(W * H, n=n_fft)[:n]  # conv[j] = sum_{k=0}^j w_k * h_{j-k}
    
    # Second term: sum_{k=0}^{n-1-j} w_k * h_{j+k}
    # Let h_flip = h[::-1], w_flip = w[::-1]
    # sum_{k=0}^{n-1-j} w_k * h_{j+k} = sum_{k=0}^{n-1-j} w_k * h_flip[n-1-j-k]
    # = (w * h_flip)[n-1-j]
    h_flip = h[::-1]
    W2 = np.fft.rfft(w, n=n_fft)
    HF = np.fft.rfft(h_flip, n=n_fft)
    conv2 = np.fft.irfft(W2 * HF, n=n_fft)
    # conv2[m] = sum_{k=0}^{m} w_k * h_flip[m-k] for m < n
    # We need conv2[n-1-j] for j=0..n-1, so indices n-1, n-2, ..., 0
    second_term = conv2[:n][::-1]  # second_term[j] = conv2[n-1-j]
    
    grad = -(2.0 / n) * (conv + second_term)
    
    return val, grad


def _epigraph_objective(x, beta):
    """
    Epigraph formulation: minimize t subject to smooth_max(h) <= t,
    with x = [h_0, ..., h_{N-1}, t].
    """
    h = x[:N]
    t = x[N]
    val, _ = _smooth_max_grad(h, beta)
    return t + 0.01 * max(0, val - t)  # Penalize constraint violation


def _epigraph_grad(x, beta):
    """
    Gradient of epigraph objective.
    """
    h = x[:N]
    t = x[N]
    val, grad_h = _smooth_max_grad(h, beta)
    
    grad = np.zeros(len(x))
    if val > t:
        grad[:N] = grad_h * 0.01
        grad[N] = 1.0 + 0.01
    else:
        grad[:N] = grad_h * 0.0
        grad[N] = 1.0
    
    return grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # Second pass if clipping changed sum
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _run_slsqp(h0, beta, maxiter, ftol, gtol):
    """
    Run SLSQP optimization on the smooth max objective.
    """
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
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res.x
    except Exception:
        return h0


def _adaptive_beta_slsqp(h0, beta_start=10.0, beta_end=300.0, 
                          n_phases=5, maxiter_per_phase=200, 
                          ftol=1e-10, gtol=1e-8, deadline=None):
    """
    Multi-phase SLSQP with increasing beta for progressively sharper max approximation.
    """
    h = h0.copy()
    for phase in range(n_phases):
        if deadline is not None and time.time() > deadline:
            break
        frac = (phase + 1) / n_phases
        beta = beta_start * (beta_end / beta_start) ** frac
        h_new = _run_slsqp(h, beta, maxiter_per_phase, ftol, gtol)
        h_new = _fix_sum(h_new)
        if _score(h_new) <= _score(h) + 1e-12:
            h = h_new
    return h


def _polish(h, beta, maxiter, ftol, gtol):
    """
    High-precision polish pass with tighter tolerances.
    """
    h0 = h.copy()
    h_new = _run_slsqp(h0, beta, maxiter, ftol, gtol)
    h_new = _fix_sum(h_new)
    
    # Verify improvement
    if _score(h_new) < _score(h0):
        return h_new
    return h0


def _spectral_h_from_coeffs(coeffs):
    """
    Reconstruct h from real DFT coefficients (modes 1..K).
    coeffs = [a1, b1, a2, b2, ..., aK, bK]
    h_i = 0.5 + sum_{k=1}^K (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    """
    n_modes = len(coeffs) // 2
    h = np.full(N, 0.5)
    i = np.arange(N)
    for k in range(1, n_modes + 1):
        a = coeffs[2*(k-1)]
        b = coeffs[2*(k-1)+1]
        h += a * np.cos(2.0 * np.pi * k * i / N)
        h += b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral_coeffs(h, K):
    """
    Extract the first K real DFT coefficients from h.
    Returns [a1, b1, ..., aK, bK] where h ~ 0.5 + sum (a_k cos + b_k sin)
    """
    H = np.fft.rfft(h)
    out = []
    for k in range(1, K + 1):
        a = 2.0 / N * H[k].real
        b = -2.0 / N * H[k].imag
        out.append(a)
        out.append(b)
    return out


def _optimize_spectral(h_init, K, beta, maxiter):
    """
    Optimize the spectral coefficients of h to minimize the smooth max objective.
    """
    coeffs0 = _h_to_spectral_coeffs(h_init, K)
    
    def obj(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        
        # Penalty for out-of-bounds to keep solution valid
        penalty = 0.0
        if np.any(h < 0) or np.any(h > 1):
            penalty = np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h-1)**2)
        
        val, _ = _smooth_max_grad(h, beta)
        return val + 1.0 * penalty

    try:
        res = minimize(obj, coeffs0, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.success:
            h_opt = _spectral_h_from_coeffs(res.x)
            return h_opt
    except Exception:
        pass
    return h_init


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
    
    # 1. Cosine start
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 2. Spectral Optimization (quick, uses L-BFGS-B on low-dim space)
    for K in [2, 4, 8, 16]:
        if time.time() - t0 > _DEADLINE - 40:
            break
        h_spec = _optimize_spectral(h_cos, K, beta=50.0, maxiter=200)
        _update_best(h_spec)
    
    # 3. Adaptive beta SLSQP (main optimization, now fast due to vectorized grad)
    if time.time() - t0 < _DEADLINE - 30:
        h_opt = _adaptive_beta_slsqp(best_h, beta_start=10.0, beta_end=300.0,
                                      n_phases=5, maxiter_per_phase=300,
                                      ftol=1e-10, gtol=1e-8,
                                      deadline=_DEADLINE - 15)
        _update_best(h_opt)
    
    # 4. Random restarts with adaptive beta
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(4):
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            h_spec_rand = _optimize_spectral(h_rand, K=8, beta=50.0, maxiter=100)
            _update_best(h_spec_rand)
            if time.time() - t0 < _DEADLINE - 10:
                h_adapt = _adaptive_beta_slsqp(h_spec_rand, beta_start=20.0, beta_end=200.0,
                                                n_phases=3, maxiter_per_phase=150,
                                                ftol=1e-10, gtol=1e-8,
                                                deadline=_DEADLINE - 8)
                _update_best(h_adapt)
    
    # 5. High-precision polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 8:
        h_polished = _polish(best_h, beta=200.0, maxiter=500, ftol=1e-12, gtol=1e-10)
        _update_best(h_polished)
        
    # 6. Final ultra-precise polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 4:
        h_final = _polish(best_h, beta=500.0, maxiter=800, ftol=1e-14, gtol=1e-12)
        _update_best(h_final)
    
    # 7. One more pass if time allows
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_final2 = _polish(best_h, beta=1000.0, maxiter=500, ftol=1e-15, gtol=1e-13)
        _update_best(h_final2)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
