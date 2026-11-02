import numpy as np
from scipy.optimize import minimize
import time

N = 50
_TIME_LIMIT = 135.0
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
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _softmax_score(h, beta=50.0):
    """
    Smooth surrogate using log-sum-exp for the max in denominator.
    """
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m < 1e-12:
        return 0.0
    log_sum_exp = m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
    return s2 / (s1 * log_sum_exp)

def _optimize_from_init(h0, beta=50.0):
    """
    Optimize using Nelder-Mead on smooth surrogate, then polish with exact score.
    """
    h = np.array(h0, dtype=float)
    h = np.clip(h, 0, None)
    if np.sum(h) < 1e-12:
        return h, 0.0
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Use scipy minimize with Nelder-Mead on the negative smooth surrogate
    def neg_surrogate(x):
        x = np.clip(x, 0, None)
        return -_softmax_score(x, beta=beta)
    
    try:
        result = minimize(neg_surrogate, h, method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8,
                                 'adaptive': True})
        h_opt = np.clip(result.x, 0, None)
        s_opt = _score(h_opt)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    except Exception:
        pass
    
    # Local random refinement
    rng = np.random.default_rng(int(abs(best_score * 1e6)) % 10000)
    for _ in range(50):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        h_pert = best_h + rng.normal(0, 0.02, N)
        h_pert = np.clip(h_pert, 0, None)
        s_pert = _score(h_pert)
        if s_pert > best_score:
            best_score = s_pert
            best_h = h_pert.copy()
    
    return best_h, best_score

def construct():
    # Known best vector
    h0 = [0.0] * N
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    best_h = np.array(h0)
    best_score = _score(best_h)
    
    # Try various initializations including sparse patterns
    inits = [
        h0,
        [1.0/2, 1.0/2] + [0.0]*(N-2),
        [1.0/3]*3 + [0.0]*(N-3),
        [0.4, 0.3, 0.3] + [0.0]*(N-3),
        [0.5, 0.25, 0.25] + [0.0]*(N-3),
        # More spread out patterns
        [1.0/4]*4 + [0.0]*(N-4),
        [0.2]*5 + [0.0]*(N-5),
        [0.15]*7 + [0.0]*(N-7),
        [0.1]*10 + [0.0]*(N-10),
        # Uneven spread
        [0.3, 0.2, 0.2, 0.1, 0.1, 0.1] + [0.0]*(N-6),
        [0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05] + [0.0]*(N-7),
    ]
    
    for init in inits:
        if time.time() - _start_time > _TIME_LIMIT:
            break
        h_opt, s_opt = _optimize_from_init(init, beta=50.0)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Final random search around best
    rng = np.random.default_rng(42)
    for _ in range(100):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        h_pert = best_h + rng.normal(0, 0.03, N)
        h_pert = np.clip(h_pert, 0, None)
        s_pert = _score(h_pert)
        if s_pert > best_score:
            best_score = s_pert
            best_h = h_pert
    
    return [float(x) for x in best_h]
