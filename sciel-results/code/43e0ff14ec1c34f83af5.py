import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _project_box_sum(h, target):
    """
    Project h onto {x in [0,1]^N : sum(x) = target}.
    Uses bisection on the Lagrange multiplier.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    
    # If already in range and sum is correct
    if np.all(h >= -1e-15) and np.all(h <= 1 + 1e-15):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            return np.clip(h, 0.0, 1.0)
    
    # Bisection on lambda: x_i = clip(h_i - lambda, 0, 1)
    # sum_i clip(h_i - lambda, 0, 1) = target
    
    lo, hi = -1.0, 1.0  # lambda bounds
    
    # Ensure we can find a solution
    # If lambda = -1, all x_i = 1, sum = n
    # If lambda = 1, all x_i = 0, sum = 0
    # So we need target in [0, n]
    
    for _ in range(80):
        mid = (lo + hi) / 2.0
        x = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(x)
        if s > target:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2.0
    x = np.clip(h - lam, 0.0, 1.0)
    
    # Fix any residual sum error by adjusting interior points
    diff = target - np.sum(x)
    if abs(diff) > 1e-14:
        mask = (x > 1e-10) & (x < 1.0 - 1e-10)
        if np.any(mask):
            x[mask] += diff / np.sum(mask)
            x = np.clip(x, 0.0, 1.0)
    
    # Final fix
    diff = target - np.sum(x)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(x)
        else:
            idx = np.argmax(x)
        x[idx] += diff
        x = np.clip(x, 0.0, 1.0)
    
    return x


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
    return _project_box_sum(h, N / 2.0)


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


def _nesterov_pgd(h0, beta, maxiter, step_scale, time_budget):
    """
    Nesterov-accelerated projected gradient descent.
    Minimize smooth max of overlaps with box+sum constraints.
    """
    h = np.array(h0, dtype=np.float64)
    y = h.copy()
    target = N / 2.0
    
    # Estimate Lipschitz constant via power iteration
    # L ≈ max eigenvalue of Hessian of smooth max
    # Approximate: use gradient norm scaling
    val0, grad0 = _smooth_max_grad(y, beta)
    grad_norm = np.linalg.norm(grad0)
    if grad_norm < 1e-10:
        L = beta * 2.0 / N
    else:
        # Conservative estimate
        L = beta * 4.0 / N
    
    t_k = 1.0
    best_h = h.copy()
    best_val = _score(h)
    
    for it in range(maxiter):
        if time.time() > time_budget:
            break
        
        # Nesterov momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        alpha = (t_k - 1.0) / t_new
        y = h + alpha * (h - y)
        
        # Gradient step
        val, grad = _smooth_max_grad(y, beta)
        step = 1.0 / L
        h_new = y - step * grad
        
        # Project onto feasible set
        h_new = _project_box_sum(h_new, target)
        
        # Check convergence
        h = h_new
        t_k = t_new
        
        # Track best
        if it % 10 == 0 or it == maxiter - 1:
            s = _score(h)
            if s < best_val:
                best_val = s
                best_h = h.copy()
    
    return best_h, best_val


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


def _coordinate_descent_polish(h, beta, max_iter, time_budget):
    """
    Coordinate descent polish: iterate over coordinates, find optimal value
    given others fixed.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    n = len(h)
    target = n / 2.0
    
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        
        improved = False
        for j in range(n):
            if time.time() > time_budget:
                break
            
            h_j = h[j]
            
            # Compute sum of others
            sum_others = np.sum(h) - h_j
            
            # For fixed others, h_j is determined by sum constraint:
            # h_j = target - sum_others
            # But must be in [0,1]
            h_j_new = target - sum_others
            h_j_new = np.clip(h_j_new, 0.0, 1.0)
            
            if abs(h_j_new - h_j) > 1e-12:
                h[j] = h_j_new
                improved = True
        
        if not improved:
            break
    
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
    
    # Phase 1: Nesterov PGD with low beta (fast, many iterations)
    if time.time() - t0 < _DEADLINE - 60:
        h1, s1 = _nesterov_pgd(h0, beta=10.0, maxiter=500, step_scale=1.0,
                               time_budget=_DEADLINE - 55)
        _update_best(h1)
    
    # Phase 2: Nesterov PGD with medium beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 45:
        h2, s2 = _nesterov_pgd(best_h, beta=30.0, maxiter=400, step_scale=1.0,
                               time_budget=_DEADLINE - 40)
        _update_best(h2)
    
    # Phase 3: Nesterov PGD with higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h3, s3 = _nesterov_pgd(best_h, beta=60.0, maxiter=300, step_scale=1.0,
                               time_budget=_DEADLINE - 25)
        _update_best(h3)
    
    # Phase 4: Nesterov PGD with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 18:
        h4, s4 = _nesterov_pgd(best_h, beta=100.0, maxiter=250, step_scale=1.0,
                               time_budget=_DEADLINE - 15)
        _update_best(h4)
    
    # Phase 5: Final high-beta polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h5, s5 = _nesterov_pgd(best_h, beta=150.0, maxiter=200, step_scale=1.0,
                               time_budget=_DEADLINE - 8)
        _update_best(h5)
    
    # Coordinate descent polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_cd = _coordinate_descent_polish(best_h, beta=100.0, max_iter=50,
                                          time_budget=_DEADLINE - 3)
        _update_best(h_cd)
    
    # Pure tone spectral search (if time permits)
    if time.time() - t0 < _DEADLINE - 35:
        freq_candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        for freq in freq_candidates:
            if time.time() - t0 > _DEADLINE - 20:
                break
            
            for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
                if time.time() - t0 > _DEADLINE - 15:
                    break
                
                for amp in [0.3, 0.4, 0.5, 0.6]:
                    if time.time() - t0 > _DEADLINE - 10:
                        break
                    
                    params = np.array([float(freq), phase, amp, 0.5])
                    h_tone = _pure_tone_to_h(params)
                    h_tone = _fix_sum(h_tone)
                    s = _score(h_tone)
                    
                    if s < best_score:
                        _update_best(h_tone)
                        
                        # Quick Nesterov refinement from this starting point
                        if time.time() - t0 < _DEADLINE - 8:
                            h_ref, s_ref = _nesterov_pgd(h_tone, beta=50.0,
                                                         maxiter=100, step_scale=1.0,
                                                         time_budget=_DEADLINE - 6)
                            _update_best(h_ref)
    
    # Random restarts if time permits
    if time.time() - t0 < _DEADLINE - 25:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            if time.time() - t0 < _DEADLINE - 12:
                h_r, s_r = _nesterov_pgd(h_rand, beta=30.0, maxiter=150,
                                         step_scale=1.0, time_budget=_DEADLINE - 10)
                _update_best(h_r)
            if best_h is not None and time.time() - t0 < _DEADLINE - 8:
                h_p, s_p = _nesterov_pgd(best_h, beta=80.0, maxiter=100,
                                         step_scale=1.0, time_budget=_DEADLINE - 6)
                _update_best(h_p)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
