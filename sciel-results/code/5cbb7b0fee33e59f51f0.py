"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-based smooth-max optimization with Nesterov acceleration and coordinate descent.
"""
import time
import numpy as np

N = 100
_T_START = time.time()
# Safety margin: evaluator kills at 150s. We aim to finish well before.
_DEADLINE = _T_START + 140.0


def _linear_overlap_values(h):
    """
    Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}.
    Uses FFT for efficient autocorrelation calculation.
    """
    n = len(h)
    # Next power of 2 for FFT efficiency
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    # Autocorrelation via power spectrum
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    # The first n values of corr correspond to lags 0 to n-1
    r = corr[:n]
    # C_k is the overlap measure. 
    # Integral of h(x)(1-h(x-s)) = 1 - (h*h)(s).
    # In discrete terms with width 2/N:
    # Sum(h_i * (1-h_{i-k})) * (2/N) = (2/N) * (Sum(h_i) - Sum(h_i h_{i-k}))
    # Sum(h_i) = N/2.
    # So C_k = (2/N) * (N/2 - r_k).
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
    Uses bisection on the Lagrange multiplier (offset) for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64)
    current_sum = np.sum(h)
    
    # Quick check
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    # Bisection for offset alpha such that sum(clip(h + alpha, 0, 1)) = target_sum
    # The function f(alpha) = sum(clip(h + alpha, 0, 1)) is monotonic increasing in alpha.
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0
    
    # Ensure bounds are valid for bisection
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
    
    # Final correction for numerical precision
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
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    grad F = sum(w_k * grad C_k)
    """
    cs = _linear_overlap_values(h)
    n = len(h)
    
    # Compute weights w_k for the soft-max
    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp
    
    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)
    
    # Compute gradient
    # C_k = (2/N) * (N/2 - r_k)
    # r_k = sum_{i=k}^{N-1} h_i h_{i-k}
    # d(r_k)/d(h_j) = h_{j-k} [j>=k] + h_{j+k} [j+k<N]
    # d(C_k)/d(h_j) = -(2/N) * (h_{j-k} [j>=k] + h_{j+k} [j+k<N])
    
    # We need grad_F_j = sum_k w_k * d(C_k)/d(h_j)
    # grad_F_j = -(2/N) * ( sum_{k=0}^j w_k h_{j-k} + sum_{k=0}^{N-1-j} w_k h_{j+k} )
    
    # Let A_j = sum_{k=0}^j w_k h_{j-k}  (Convolution of w and h)
    # Let B_j = sum_{k=0}^{N-1-j} w_k h_{j+k} (Correlation of w and h)
    
    # Efficient calculation using FFT
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
        
    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    
    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)
    
    # Convolution w * h
    conv = np.fft.irfft(W * H, n=n_fft)
    # A_j is conv[j] for j=0..N-1
    A = conv[:N]
    
    # B_j is correlation. 
    # B_j = sum_{k=0}^{N-1-j} w_k h_{j+k}
    # This is the j-th element of the cross-correlation of w and h.
    # conv(w_rev, h)[N-1+j] = sum_k w[k] h[j+k]
    
    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)
    
    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]
    
    grad = -(2.0 / N) * (A + B)
    
    return val, grad


def _nesterov_accelerated_gradient(h0, beta, n_iter=200, lr=0.05):
    """
    Nesterov accelerated gradient descent on the smooth max objective with adaptive LR.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    h_prev = h.copy()
    t = 1.0
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Adaptive learning rate with backtracking
    lr_current = lr
    max_backtracks = 8
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        # Nesterov step
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        
        y = h + momentum * (h - h_prev)
        
        val, grad = _smooth_max_obj_grad(y, beta)
        
        # Backtracking line search
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
        
        # If we backtracked too much, reset LR
        if backtrack_count == max_backtracks:
            lr_current = max(lr * 0.1, 1e-6)
        
        # Update momentum
        h_prev = h
        h = h_new
        t = t_new
        
        # Check improvement on actual objective
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h, best_score

def _polish(h, n_iter=50, lr=0.01, beta=1000.0):
    """
    Final high-precision polish pass with very high beta and small learning rate.
    """
    h = np.array(h, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 2:
            break
            
        val, grad = _smooth_max_obj_grad(h, beta)
        
        # Small gradient step
        h_new = h - lr * grad
        h_new = _project(h_new, N / 2.0)
        
        # Check improvement
        current_score = _score(h_new)
        if current_score < best_score:
            best_score = current_score
            best_h = h_new.copy()
            h = h_new
        else:
            # Try smaller step
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
    
    # Initializations
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
        
    # Run NAG with different betas and initializations
    # Start with lower beta for smoothness, increase for sharpness
    # Use more aggressive beta progression
    betas = [5.0, 20.0, 100.0, 500.0]
    
    for beta in betas:
        if time.time() > _DEADLINE - 40:
            break
            
        # Use fewer iterations for higher betas since they're more sensitive
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
            if time.time() > _DEADLINE - 15:
                break
                
            # Run NAG with adaptive LR
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=n_iter, lr=lr)
            
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Final high-precision polish
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_polished, s_polished = _polish(best_h, n_iter=100, lr=0.005, beta=2000.0)
        if s_polished < best_score:
            best_score = s_polished
            best_h = h_polished.copy()

    if best_h is None:
        # Fallback
        best_h = np.full(N, 0.5)
        
    # Ensure exact sum and bounds
    best_h = _project(best_h, N / 2.0)
    
    return best_h
