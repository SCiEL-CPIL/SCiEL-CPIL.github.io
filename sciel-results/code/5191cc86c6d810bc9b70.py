"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-accelerated linear overlap computation with Nesterov accelerated
projected gradient on a log-sum-exp smooth surrogate. Multi-phase beta schedule
for coarse-to-fine optimization. Includes chirp parameterization for diverse
initialization and SLSQP for final polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_TARGET = N / 2.0
_NFFT = 1  # will be set in init
while _NFFT < 2 * N:
    _NFFT *= 2

# Safety margin: Evaluator kills at 150s. We target finishing by 140s.
_DEADLINE = time.time() + 140


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    h_pad = np.zeros(_NFFT)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=_NFFT)
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


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    
    # Numerical stability for log-sum-exp
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        w = np.full(n, 1.0/n)
        val = m
    else:
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
    grad = np.zeros(n)
    
    for k in range(n):
        if w[k] < 1e-12:
            continue
        factor = -(2.0 / n) * w[k]
        
        if k > 0:
            grad[k:] += factor * h[:n-k]
        grad[:n-k] += factor * h[k:]
            
    return val, grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
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


def _chirp_parameterization(sweep_rate):
    """
    Generate step heights from a quadratic chirp phase profile.
    h_i = |A * exp(i * (omega*i^2 + phi))|^2 scaled to sum to N/2.
    The squared magnitude of a chirp is constant, so we use the real part
    or a more complex construction. Actually, let's use:
    h_i = 0.5 + 0.5 * cos(sweep_rate * i^2 + phase)
    This gives a signal with near-constant autocorrelation sidelobes.
    """
    i = np.arange(N)
    # Quadratic phase chirp
    phase = sweep_rate * i * i
    # Use cosine to get values in [0, 1]
    h = 0.5 + 0.5 * np.cos(phase)
    return _fix_sum(h)


def _run_slsqp(h0, beta, maxiter, ftol):
    """
    Run SLSQP optimization with smooth max objective.
    Returns the optimized array or the input if it fails/times out.
    """
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - _TARGET,
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
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.x is not None:
            return res.x
        return h0
    except Exception:
        return h0


def _nesterov_projected_gradient(h0, beta, max_iter, lr_init=0.05):
    """
    Nesterov accelerated projected gradient method on smooth max.
    Projects onto {h: 0 <= h_i <= 1, sum(h) = N/2}.
    """
    h = _fix_sum(h0.copy())
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        val, grad = _smooth_max_grad(h, beta)
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project onto box [0,1]
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # Project onto sum constraint
        h_new = _fix_sum(h_new)
        
        # Nesterov momentum update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        
        # Project again
        h_next = np.clip(h_next, 0.0, 1.0)
        h_next = _fix_sum(h_next)
        
        h_prev = h
        h = h_next
        t_k = t_new
        
        # Adaptive learning rate: decrease slowly
        if it % 100 == 99:
            lr = max(0.001, lr * 0.9)
            
    return h


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
    
    i = np.arange(N)
    
    # 1. Initial Candidates
    candidates = []
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    candidates.append(h_cos)
    
    # Constant
    h_const = np.full(N, 0.5)
    candidates.append(h_const)
    
    # Sine
    h_sin = 0.5 + 0.4 * np.sin(2.0 * np.pi * i / N)
    candidates.append(h_sin)
    
    # Alternating
    h_alt = np.where(i % 2 == 0, 0.8, 0.2)
    candidates.append(h_alt)
    
    # Chirp parameterization with various sweep rates
    for sweep in [0.01, 0.05, 0.1, 0.2, 0.5]:
        h_chirp = _chirp_parameterization(sweep)
        candidates.append(h_chirp)
    
    # Random
    for _ in range(3):
        h_rand = np.random.uniform(0.1, 0.9, N)
        candidates.append(h_rand)
    
    # Evaluate all candidates
    for h0 in candidates:
        _update_best(h0)
    
    # 2. Optimization Phases
    
    # Phase 1: Nesterov from best candidate with moderate beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 60:
        h_opt = _nesterov_projected_gradient(best_h, beta=50.0, max_iter=1000, lr_init=0.05)
        _update_best(h_opt)
        
    # Phase 2: SLSQP refinement from best with higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_opt = _run_slsqp(best_h, beta=100.0, maxiter=300, ftol=1e-10)
        _update_best(h_opt)
        
    # Phase 3: Nesterov with high beta for sharper max approximation
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_opt = _nesterov_projected_gradient(best_h, beta=200.0, max_iter=500, lr_init=0.02)
        _update_best(h_opt)
        
    # Phase 4: SLSQP final polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_opt = _run_slsqp(best_h, beta=300.0, maxiter=200, ftol=1e-12)
        _update_best(h_opt)
        
    # Phase 5: Emergency Nesterov polish if time remains
    if best_h is not None and time.time() - t0 < _DEADLINE - 1:
        h_opt = _nesterov_projected_gradient(best_h, beta=500.0, max_iter=200, lr_init=0.01)
        _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
