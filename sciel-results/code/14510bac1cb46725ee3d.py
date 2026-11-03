import time
import numpy as np
from scipy.optimize import minimize

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145

def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _project(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0

    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid

    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)

    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj

def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C_k.
    """
    cs = _linear_overlap_values(h)
    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp

    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)

    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2

    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h

    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)

    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]

    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)

    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]

    grad = -(2.0 / N) * (A + B)

    return val, grad

def _slsqp_polish(h0, beta=10.0, max_iter=50):
    """
    Polish using SLSQP with equality constraint for sum.
    """
    def objective(x):
        val, _ = _smooth_max_obj_grad(x, beta)
        return val
    
    def gradient(x):
        _, grad = _smooth_max_obj_grad(x, beta)
        return grad

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0, 'jac': lambda x: np.ones(N)}
    ]
    bounds = [(0.0, 1.0) for _ in range(N)]

    try:
        res = minimize(objective, h0, method='SLSQP', jac=gradient, 
                       bounds=bounds, constraints=constraints, 
                       options={'maxiter': max_iter, 'ftol': 1e-9})
        if res.success:
            return res.x, _score(res.x)
    except Exception:
        pass
    return h0, _score(h0)

def construct():
    best_h = None
    best_score = float('inf')

    np.random.seed(42)
    
    # Initializations
    inits = []
    # Random
    for _ in range(10):
        h_init = np.random.rand(N)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
        
    # Sine waves
    for freq in [1, 2, 3, 5, 8]:
        h_init = 0.5 + 0.3 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        h_init = np.clip(h_init, 0.0, 1.0)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
        
    # Constant
    h_init = np.full(N, 0.5)
    inits.append(h_init)

    # Phase 1: SLSQP Polish on diverse inits
    for h_init in inits:
        if time.time() > _DEADLINE - 20:
            break
        h_opt, s_opt = _slsqp_polish(h_init, beta=10.0, max_iter=100)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()
            
    # Phase 2: Further polish on best
    if best_h is not None:
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > _DEADLINE - 5:
                break
            h_opt, s_opt = _slsqp_polish(best_h, beta=beta, max_iter=100)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h, N / 2.0)
    return best_h
