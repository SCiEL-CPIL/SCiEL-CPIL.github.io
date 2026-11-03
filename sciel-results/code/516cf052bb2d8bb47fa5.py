"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses FFT-based smooth-max optimization with Nesterov acceleration,
epigraph formulation, and coordinate descent polish.
"""
import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 140.0


def _linear_overlap_values(h):
    """
    Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}.
    Uses FFT for efficient autocorrelation calculation.
    """
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
    """Compute the actual maximum overlap score."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    current_sum = np.sum(h)
    
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0
    
    if np.sum(np.clip(h + low, 0.0, 1.0)) > target_sum:
        low = -10.0
    if np.sum(np.clip(h + high, 0.0, 1.0)) < target_sum:
        high = 10.0

    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = np.sum(np.clip(h + mid, 0.0, 1.0))
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
            h_proj[idx] = min(1.0, h_proj[idx] + diff)
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] = max(0.0, h_proj[idx] + diff)
        h_proj = np.clip(h_proj, 0.0, 1.0)
        
    return h_proj


def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C_k.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    """
    cs = _linear_overlap_values(h)
    n = len(h)
    
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


def _epigraph_obj_grad(h, t, beta):
    """
    Epigraph formulation: minimize t + (1/beta)*log(sum(exp(beta*(C_k - t))))
    Gradient wrt h: sum(w_k * grad C_k)
    Gradient wrt t: 1 - sum(w_k) = 0 (since w sums to 1)
    """
    cs = _linear_overlap_values(h)
    n = len(h)
    
    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - t - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp
    
    val = t + (1.0 / beta) * (np.log(sum_exp) + beta * (cs_max - t))
    
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
    
    grad_h = -(2.0 / N) * (A + B)
    grad_t = 1.0 - np.sum(w)
    
    return val, grad_h, grad_t


def _nesterov_smooth_max(h0, beta, n_iter=200, lr=0.05):
    """
    Nesterov accelerated gradient descent on the smooth max objective.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    h_prev = h.copy()
    t = 1.0
    
    best_h = h.copy()
    best_score = _score(h)
    
    lr_current = lr
    max_backtracks = 8
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        
        y = h + momentum * (h - h_prev)
        
        val, grad = _smooth_max_obj_grad(y, beta)
        
        h_new = y - lr_current * grad
        h_new = _project(h_new, N / 2.0)
        val_new = _score(h_new)
        
        backtrack_count = 0
        while val_new > val + 1e-10 * np.abs(val) and backtrack_count < max_backtracks:
            lr_current *= 0.5
            h_new = y - lr_current * grad
            h_new = _project(h_new, N / 2.0)
            val_new = _score(h_new)
            backtrack_count += 1
        
        if backtrack_count == max_backtracks:
            lr_current = max(lr * 0.1, 1e-6)
        
        h_prev = h
        h = h_new
        t = t_new
        
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h, best_score


def _nesterov_epigraph(h0, t0, beta, n_iter=200, lr=0.05):
    """
    Nesterov accelerated gradient descent on the epigraph objective.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    t = t0
    
    h_prev = h.copy()
    t_prev = t
    tau = 1.0
    
    best_h = h.copy()
    best_t = t
    best_score = _score(h)
    
    lr_current = lr
    max_backtracks = 8
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        tau_new = (1 + np.sqrt(1 + 4 * tau * tau)) / 2
        momentum = (tau - 1) / tau_new
        
        y_h = h + momentum * (h - h_prev)
        y_t = t + momentum * (t - t_prev)
        
        val, grad_h, grad_t = _epigraph_obj_grad(y_h, y_t, beta)
        
        h_new = y_h - lr_current * grad_h
        h_new = _project(h_new, N / 2.0)
        t_new = y_t - lr_current * grad_t
        t_new = max(0.0, t_new)
        
        val_new = _score(h_new)
        
        backtrack_count = 0
        while val_new > val + 1e-10 * np.abs(val) and backtrack_count < max_backtracks:
            lr_current *= 0.5
            h_new = y_h - lr_current * grad_h
            h_new = _project(h_new, N / 2.0)
            t_new = y_t - lr_current * grad_t
            t_new = max(0.0, t_new)
            val_new = _score(h_new)
            backtrack_count += 1
        
        if backtrack_count == max_backtracks:
            lr_current = max(lr * 0.1, 1e-6)
        
        h_prev = h
        t_prev = t
        h = h_new
        t = t_new
        tau = tau_new
        
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            best_t = t
            
    return best_h, best_score


def _coordinate_descent(h, n_iter=200, tol=1e-10):
    """
    Coordinate descent polish: optimize one coordinate at a time.
    """
    h = np.array(h, dtype=np.float64).copy()
    h = _project(h, N / 2.0)
    
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 3:
            break
        
        improved = False
        for j in range(N):
            # Fix all other coordinates, optimize h[j]
            # The objective is max_k C_k where C_k depends on h[j]
            # For a given j, C_k is linear in h[j] for each k
            # So max_k C_k is piecewise linear in h[j]
            # We can find the optimal h[j] by checking breakpoints
            
            h_save = h[j]
            h_other_sum = np.sum(h) - h[j]
            
            # For each k, C_k = (2/N)*(N/2 - r_k)
            # r_k depends on h[j] if j >= k (h_j appears in r_k) or j+k < N (h_{j+k} appears)
            # r_k = sum_{i=k}^{N-1} h_i h_{i-k}
            # Term with h_j: h_j * h_{j-k} if j>=k, plus h_{j+k} * h_j if j+k<N
            
            # C_k = (2/N)*(N/2 - r_k)
            # = (2/N)*(N/2 - (r_k_base + h_j * (h_{j-k}[j>=k] + h_{j+k}[j+k<N])))
            # = C_k_base - (2/N)*h_j*(h_{j-k}[j>=k] + h_{j+k}[j+k<N])
            
            # So C_k is affine in h_j: C_k = a_k - b_k * h_j
            # where b_k = (2/N)*(h_{j-k}[j>=k] + h_{j+k}[j+k<N]) >= 0
            # and a_k = C_k with h_j=0
            
            # We want to minimize max_k (a_k - b_k * h_j) over h_j in [0, 1]
            # with the constraint that sum = N/2, so h_j is determined by others... 
            # Actually, we're optimizing h_j freely in [0,1] and the sum constraint 
            # is handled by projection. But in coordinate descent, we fix sum.
            
            # Let's just do a simple line search: try a few values
            # Or better: the optimal h_j is where the max is minimized
            # This happens at a breakpoint or at boundary
            
            # Compute coefficients
            b = np.zeros(N)
            for k in range(N):
                coeff = 0.0
                if j >= k:
                    coeff += h[j - k]
                if j + k < N:
                    coeff += h[j + k]
                b[k] = (2.0 / N) * coeff
            
            # C_k = a_k - b_k * h_j
            # a_k = C_k when h_j = 0, which means r_k with h_j=0
            # r_k(h_j=0) = r_k - h_j*(h_{j-k}[j>=k] + h_{j+k}[j+k<N])
            # So a_k = C_k + (2/N)*h_j*(h_{j-k}[j>=k] + h_{j+k}[j+k<N]) = C_k + b_k*h_j
            
            cs_current = _linear_overlap_values(h)
            a = cs_current + b * h[j]
            
            # Minimize max_k (a_k - b_k * x) over x in [0,1]
            # This is a linear programming problem in 1D
            # The optimum is at a point where two lines cross, or at boundary
            
            # Find all breakpoints: a_i - b_i*x = a_j - b_j*x
            # x = (a_i - a_j) / (b_j - b_i) for b_i != b_j
            
            candidates = [0.0, 1.0]
            for i in range(N):
                for j in range(i+1, N):
                    if abs(b[i] - b[j]) > 1e-15:
                        x = (a[i] - a[j]) / (b[j] - b[i])
                        if 0.0 <= x <= 1.0:
                            candidates.append(x)
            
            # Evaluate at all candidates
            best_x = h_save
            best_val = np.max(a - b * h_save)
            for x in candidates:
                val = np.max(a - b * x)
                if val < best_val - 1e-12:
                    best_val = val
                    best_x = x
            
            if best_x != h_save:
                h[j] = best_x
                h = _project(h, N / 2.0)
                new_score = _score(h)
                if new_score < best_score - 1e-12:
                    best_score = new_score
                    best_h = h.copy()
                    improved = True
                else:
                    h[j] = h_save
            else:
                h[j] = h_save
        
        if not improved:
            break
    
    return best_h, best_score


def _polish(h, n_iter=100, lr=0.005, beta=2000.0):
    """
    Final high-precision polish pass.
    """
    h = np.array(h, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 2:
            break
            
        val, grad = _smooth_max_obj_grad(h, beta)
        
        h_new = h - lr * grad
        h_new = _project(h_new, N / 2.0)
        
        current_score = _score(h_new)
        if current_score < best_score:
            best_score = current_score
            best_h = h_new.copy()
            h = h_new
        else:
            h_new = h - lr * 0.5 * grad
            h_new = _project(h_new, N / 2.0)
            current_score = _score(h_new)
            if current_score < best_score:
                best_score = current_score
                best_h = h_new.copy()
                h = h_new
            else:
                break
                
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    initializations = [
        np.full(N, 0.5),
        0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N),
        0.5 + 0.2 * np.random.randn(N),
        0.5 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N),
        np.random.rand(N)
    ]
    
    inits = []
    for init in initializations:
        h_init = np.clip(init, 0.0, 1.0)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
        
    # Phase 1: Smooth max with Nesterov
    betas = [5.0, 20.0, 100.0, 500.0]
    
    for beta in betas:
        if time.time() > _DEADLINE - 50:
            break
            
        if beta <= 20:
            n_iter = 400
            lr = 0.05
        elif beta <= 100:
            n_iter = 300
            lr = 0.03
        else:
            n_iter = 200
            lr = 0.01
            
        for h_init in inits:
            if time.time() > _DEADLINE - 20:
                break
                
            h_opt, s_opt = _nesterov_smooth_max(h_init, beta, n_iter=n_iter, lr=lr)
            
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 2: Epigraph formulation
    if best_h is not None and time.time() < _DEADLINE - 30:
        for beta in [100.0, 500.0]:
            if time.time() > _DEADLINE - 15:
                break
            n_iter = 200
            lr = 0.02
            t0 = _score(best_h)
            h_opt, s_opt = _nesterov_epigraph(best_h, t0, beta, n_iter=n_iter, lr=lr)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 3: Coordinate descent polish
    if best_h is not None and time.time() < _DEADLINE - 15:
        h_cd, s_cd = _coordinate_descent(best_h, n_iter=300)
        if s_cd < best_score:
            best_score = s_cd
            best_h = h_cd.copy()

    # Phase 4: Final high-precision polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_polished, s_polished = _polish(best_h, n_iter=100, lr=0.005, beta=2000.0)
        if s_polished < best_score:
            best_score = s_polished
            best_h = h_polished.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h, N / 2.0)
    
    return best_h
