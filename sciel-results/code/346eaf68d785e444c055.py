import numpy as np
import time
from scipy.optimize import minimize

N = 50
_TIME_LIMIT = 145.0
_start_time = time.time()

def _score(h):
    """
    Score function: (sum h_i^2) / ( (sum h_i) * max_k conv_k )
    where conv is the linear autoconvolution.
    """
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _score_neg(h):
    return -_score(h)

def _multiplicative_update(h, max_iter=100, tol=1e-10):
    """
    Iterative multiplicative update (Sinkhorn-Knopp style) for the min-max game.
    The attacker selects lag k to maximize conv_k, defender adjusts h to minimize ratio.
    """
    h = np.array(h, dtype=float)
    h = np.clip(h, 0, None)
    if np.sum(h) < 1e-12:
        h = np.ones(N) / N
    
    best_h = h.copy()
    best_score = _score(h)
    
    for it in range(max_iter):
        if time.time() - _start_time > _TIME_LIMIT - 15:
            break
        
        s1 = np.sum(h)
        if s1 < 1e-12:
            break
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-12:
            break
        
        # Attacker: find the lag with maximum autoconvolution
        k_max = np.argmax(conv)
        
        # Defender: adjust h to reduce peak at k_max while maintaining concentration
        # The gradient of max_conv w.r.t. h_i is: sum over j where k_max = i + j of h_j
        # This equals h[k_max - i] + h[i - k_max] (with bounds)
        
        # Compute sensitivity of conv_k_max to each h_i
        d_conv_dh = np.zeros(N)
        for i in range(N):
            # conv_k = sum_j h_j * h_{k-j}
            # d(conv_k)/d(h_i) = h_{k-i} + h_{i-k}
            j1 = k_max - i
            j2 = i - k_max
            term = 0.0
            if 0 <= j1 < N:
                term += h[j1]
            if 0 <= j2 < N:
                term += h[j2]
            d_conv_dh[i] = term
        
        # Multiplicative update: reduce h_i where it contributes most to the peak
        # h_new_i = h_i * exp(-eta * d_conv_dh_i / max_conv)
        eta = 0.1
        h_new = h * np.exp(-eta * d_conv_dh / max_conv)
        h_new = np.clip(h_new, 0, None)
        
        # Renormalize to maintain scale
        if np.sum(h_new) > 1e-12:
            h_new = h_new / np.sum(h_new) * s1
        
        s_new = _score(h_new)
        if s_new > best_score:
            best_score = s_new
            best_h = h_new.copy()
        
        if np.max(np.abs(h_new - h)) < tol:
            h = h_new
            break
        h = h_new
    
    return best_h, best_score

def _gradient_ascent(h, max_iter=200, step_size=0.05):
    """
    Gradient ascent on the score function with adaptive step size.
    """
    h = np.array(h, dtype=float)
    h = np.clip(h, 0, None)
    
    best_h = h.copy()
    best_score = _score(h)
    
    for it in range(max_iter):
        if time.time() - _start_time > _TIME_LIMIT - 20:
            break
        
        s2 = np.dot(h, h)
        s1 = np.sum(h)
        if s1 < 1e-12:
            break
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-12:
            break
        
        # Find the lag with maximum autoconvolution
        k_max = np.argmax(conv)
        
        # Gradient of score w.r.t. h_i
        # score = s2 / (s1 * max_conv)
        # d(score)/d(h_i) = [2*h_i * s1 * max_conv - s2 * (max_conv + s1 * d_max_conv_dh_i)] / (s1 * max_conv)^2
        
        d_max_conv_dh = np.zeros(N)
        for i in range(N):
            j1 = k_max - i
            j2 = i - k_max
            term = 0.0
            if 0 <= j1 < N:
                term += h[j1]
            if 0 <= j2 < N:
                term += h[j2]
            d_max_conv_dh[i] = term
        
        grad = np.zeros(N)
        for i in range(N):
            numerator = 2 * h[i] * s1 * max_conv - s2 * (max_conv + s1 * d_max_conv_dh[i])
            denominator = (s1 * max_conv) ** 2
            grad[i] = numerator / denominator
        
        # Gradient ascent with adaptive step size
        h_new = h + step_size * grad
        h_new = np.clip(h_new, 0, None)
        
        s_new = _score(h_new)
        if s_new > best_score:
            best_score = s_new
            best_h = h_new.copy()
            h = h_new
            # Slightly increase step size on success
            step_size = min(step_size * 1.05, 0.5)
        else:
            # Reduce step size on failure
            step_size *= 0.5
            if step_size < 1e-8:
                break
    
    return best_h, best_score

def _spectral_init():
    """
    Initialize using spectral considerations: distribute mass to minimize
    the peak autoconvolution while maintaining L2 concentration.
    """
    # Start with a smooth distribution that has low autoconvolution peak
    # Use a cosine window centered
    t = np.linspace(0, 1, N)
    h = np.cos(np.pi * t) ** 2
    h = np.clip(h, 0, None)
    h = h / np.sum(h)
    return h

def _random_init(rng):
    """Generate a random nonnegative initialization."""
    h = rng.random(N)
    h = np.clip(h, 0, None)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h

def construct():
    best_h = None
    best_score = 0.0
    
    # Time budget tracking
    deadline = _start_time + _TIME_LIMIT - 10
    
    # Phase 1: Try structured initializations with multiplicative updates
    inits = []
    
    # Single spike (theoretical max = 1)
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    inits.append(h_spike)
    
    # Two-point concentration
    for a in [0.5, 0.6, 0.7, 0.8]:
        h_two = np.zeros(N)
        h_two[0] = a
        h_two[1] = 1 - a
        inits.append(h_two)
    
    # Smooth cosine window
    inits.append(_spectral_init())
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(_random_init(rng))
    
    for h0 in inits:
        if time.time() > deadline:
            break
        
        # Try multiplicative update
        h_opt, s_opt = _multiplicative_update(h0, max_iter=80)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # Then gradient ascent polish
        if time.time() < deadline - 10:
            h_opt2, s_opt2 = _gradient_ascent(h_opt, max_iter=100)
            if s_opt2 > best_score:
                best_score = s_opt2
                best_h = h_opt2.copy()
    
    # Phase 2: Local optimization with scipy
    if best_h is not None and time.time() < deadline - 20:
        try:
            result = minimize(
                _score_neg,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            h_opt = np.clip(result.x, 0, None)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Phase 3: Final multiplicative update polish
    if best_h is not None and time.time() < deadline - 5:
        h_final, s_final = _multiplicative_update(best_h, max_iter=50)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final
    
    if best_h is None:
        # Fallback: single spike
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return [float(x) for x in best_h]
