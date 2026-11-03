import time
import numpy as np

N = 100
_DEADLINE = time.time() + 145  # 5s safety margin before 150s kill

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

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective (LogSumExp).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    shifted = beta * (cs - m)
    shifted = np.clip(shifted, -500, 500)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k  (Note: 2/n * n/2 = 1, so C_k = 1 - 2/n * R_k)
    # Actually C_k = (2/n)(n/2 - R_k) = 1 - (2/n)R_k
    # dC_k/dh_j = -(2/n) * dR_k/dh_j
    
    for k in range(n):
        if w[k] < 1e-10:
            continue
        factor = -(2.0 / n) * w[k]
        
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad

def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
    """
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0
    
    def g(tau):
        x = np.clip(y + tau, 0.0, 1.0)
        return np.sum(x) - target_sum
        
    if g(lo) > 0:
        return np.clip(y + lo, 0.0, 1.0)
    if g(hi) < 0:
        return np.clip(y + hi, 0.0, 1.0)
        
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
            
    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)

def _pgd_optimize(h0, beta, steps, lr, deadline, momentum=0.9):
    """
    Nesterov-accelerated Projected Gradient Descent optimization.
    """
    h = h0.copy()
    target_sum = N / 2.0
    current_lr = lr
    prev_val, _ = _smooth_max_grad(h, beta)
    y = h.copy()
    z = h.copy()
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        val, grad = _smooth_max_grad(y, beta)
        
        step_lr = current_lr
        new_h = z - step_lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        new_val, _ = _smooth_max_grad(new_h_proj, beta)
        
        while new_val > prev_val and step_lr > 1e-8:
            step_lr /= 2.0
            new_h = z - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _smooth_max_grad(new_h_proj, beta)
            
        if new_val < prev_val:
            # Nesterov update
            new_z = new_h_proj
            new_y = new_z + momentum * (new_z - z)
            h = new_y
            z = new_z
            prev_val = new_val
            if i % 10 == 0:
                current_lr = min(current_lr * 1.05, 0.1)
        else:
            current_lr = max(current_lr * 0.5, 1e-6)
            # Reset momentum on failure
            z = h
            y = h
            
    return h

def _coordinate_polish(h0, deadline, max_iters=20000, tol=1e-12):
    """
    Coordinate descent polish: for each coordinate, find the optimal value
    given all others fixed (subject to sum constraint and [0,1] bounds).
    Uses the true max overlap as the objective (not the smooth surrogate).
    """
    h = h0.copy()
    n = len(h)
    target_sum = N / 2.0
    current_val = _score(h)
    no_improve = 0
    it = 0
    
    while it < max_iters and no_improve < 200:
        if time.time() > deadline:
            break
        
        improved = False
        for j in range(n):
            if time.time() > deadline:
                break
            
            # Remove h[j] from consideration
            h[j] = 0.0
            others_sum = np.sum(h)
            
            # h[j] must be in [max(0, target_sum - others_sum), min(1, target_sum - others_sum)]
            lo = max(0.0, target_sum - others_sum)
            hi = min(1.0, target_sum - others_sum)
            
            if lo >= hi:
                h[j] = lo
                continue
            
            # Minimize max_k C_k(h) over h[j] in [lo, hi]
            # C_k is linear in h[j] for each k, so max is convex piecewise linear.
            # Use ternary search on the convex function.
            a, b = lo, hi
            for _ in range(30):
                m1 = a + (b - a) / 3.0
                m2 = b - (b - a) / 3.0
                h[j] = m1
                v1 = _score(h)
                h[j] = m2
                v2 = _score(h)
                if v1 < v2:
                    b = m2
                else:
                    a = m1
            
            h[j] = (a + b) / 2.0
            new_val = _score(h)
            
            if new_val < current_val - 1e-15:
                current_val = new_val
                improved = True
            else:
                h[j] = 0.0
                h[j] = h[j]  # keep the value that minimizes
        
        it += 1
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return h

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # 1. Initial Candidates
    i = np.arange(N)
    
    # Uniform
    h_unif = np.full(N, 0.5)
    _update_best(h_unif)
    
    # Chirps (Quadratic Phase)
    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0]:
        for phase in [0, np.pi/2]:
            h_chirp = 0.5 + 0.5 * np.cos(2.0 * np.pi * alpha * (i**2) / N + phase)
            h_chirp = _project_sum_box(h_chirp, N / 2.0)
            _update_best(h_chirp)

    # Linear Frequencies
    for freq in [1, 2, 3, 5, 10]:
        h_lin = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N)
        h_lin = _project_sum_box(h_lin, N / 2.0)
        _update_best(h_lin)
        
    # Random
    for _ in range(10):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        _update_best(h_rand)

    # 2. Optimization Phases
    # Phase 1: Coarse
    if best_h is not None and time.time() - t0 < _DEADLINE - 60:
        h_opt = _pgd_optimize(best_h, beta=10.0, steps=500, lr=0.05, deadline=_DEADLINE - 60)
        _update_best(h_opt)
        
    # Phase 2: Medium
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_opt = _pgd_optimize(best_h, beta=50.0, steps=500, lr=0.01, deadline=_DEADLINE - 30)
        _update_best(h_opt)
        
    # Phase 3: High
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_opt = _pgd_optimize(best_h, beta=100.0, steps=500, lr=0.005, deadline=_DEADLINE - 10)
        _update_best(h_opt)
        
    # Phase 4: Ultra-high polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_opt = _pgd_optimize(best_h, beta=300.0, steps=300, lr=0.002, deadline=_DEADLINE - 5)
        _update_best(h_opt)
        
    # Phase 5: Coordinate descent polish on the true objective
    if best_h is not None and time.time() - t0 < _DEADLINE - 1:
        h_cd = _coordinate_polish(best_h, deadline=_DEADLINE - 1, max_iters=5000, tol=1e-12)
        _update_best(h_cd)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project_sum_box(best_h, N / 2.0)
    return best_h
