"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Spectral-parameterized optimization with Nesterov acceleration.
We optimize the power spectrum of h, which determines the autocorrelation (and thus overlap).
Projection onto the feasible set [0,1]^N with sum constraint is done via bisection.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _autocorr(h):
    """Compute autocorrelation of h using FFT."""
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    return corr[:n]

def _overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - autocorr_k)
    """
    n = len(h)
    r = _autocorr(h)
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    return float(np.max(cs))

def _project(h, target_sum):
    """
    Project h onto the set {x in [0,1]^N : sum(x) = target_sum}.
    Uses bisection on the Lagrange multiplier lambda for the sum constraint,
    combined with clipping for the box constraints.
    """
    # Initial guess for lambda
    # If sum is too high, lambda > 0 (push down)
    # If sum is too low, lambda < 0 (push up)
    # h_proj = clip(h - lambda, 0, 1)
    # We want sum(clip(h - lambda, 0, 1)) = target_sum
    
    lo = -max(h) - 1.0
    hi = 1.0 - min(h) + 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    h_proj = np.clip(h - (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final fine-tuning to ensure exact sum
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-12:
        # Adjust interior points
        mask = (h_proj > 1e-9) & (h_proj < 1.0 - 1e-9)
        if np.any(mask):
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        
        # If still off, adjust one point
        diff = target_sum - np.sum(h_proj)
        if abs(diff) > 1e-15:
            if diff > 0:
                idx = np.argmin(h_proj)
            else:
                idx = np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)
            
    return h_proj

def _smooth_max_and_grad(h, beta):
    """
    Compute smooth max of overlap values and its gradient.
    """
    cs = _overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    
    val = m + np.log(sum_exp) / beta
    
    # Gradient of C_k w.r.t h
    # C_k = (2/N) * (N/2 - r_k)
    # r_k = sum_{i=k}^{N-1} h_i h_{i-k}
    # d r_k / d h_j = h_{j-k} (if j>=k) + h_{j+k} (if j+k < N)
    # Note: autocorr is symmetric, so r_k = r_{-k} effectively in circular, 
    # but here it's linear. 
    # Let's derive carefully.
    # r_k = sum_{i=k}^{N-1} h_i h_{i-k}
    # d r_k / d h_j:
    # Term 1: i=j. Requires j>=k. Term is h_{j-k}.
    # Term 2: i-k=j => i=j+k. Requires j+k <= N-1. Term is h_j.
    # Wait, r_k is not symmetric in index j for linear autocorr?
    # Actually, r_k = sum_{i=k}^{N-1} h_i h_{i-k}.
    # If we swap indices, it's the same sum.
    # Gradient w.r.t h_j:
    # From term i=j: h_{j-k} (valid if j>=k)
    # From term i-k=j: h_j (valid if j+k < N)
    # So d r_k / d h_j = (h_{j-k} if j>=k else 0) + (h_j if j+k<N else 0)
    
    grad = np.zeros_like(h)
    for k in range(len(cs)):
        if w[k] < 1e-15:
            continue
        factor = -(2.0 / N) * w[k]
        # Contribution to grad from C_k
        # d C_k / d h_j = -factor * (d r_k / d h_j)
        
        # Part 1: j >= k
        if k > 0:
            grad[k:] += factor * h[:N-k] # h[j-k] for j in [k, N-1]
        # Part 2: j + k < N
        if k > 0:
            grad[:N-k] += factor * h[k:] # h[j] for j in [0, N-k-1]
        # If k==0:
        # j>=0 always. h_j.
        # j+0<N always. h_j.
        # So d r_0 / d h_j = 2 h_j.
        if k == 0:
            grad += factor * 2.0 * h
            
    return val, grad

def _nesterov_projected(h, beta, steps, lr_init, lr_decay):
    """
    Nesterov accelerated gradient descent on h directly, with projection.
    We use the smooth max objective.
    """
    h_curr = h.copy()
    h_next = h.copy()
    v = np.zeros_like(h)
    
    best_h = h_curr.copy()
    best_score = _score(best_h)
    
    target_sum = N / 2.0
    
    for t in range(steps):
        # Nesterov look-ahead
        h_look = h_curr + (t / (t + 2.0)) * (h_curr - h_next)
        
        # Project look-ahead to feasible set to ensure we evaluate gradient on feasible point?
        # Or just evaluate gradient and then project update?
        # Standard projected gradient: x_{k+1} = P(x_k - alpha * grad(x_k))
        # Nesterov: x_{k+1} = P(x_k - alpha * grad(y_k)) where y_k is look-ahead.
        # Usually y_k is not projected, but gradient is evaluated at y_k.
        # However, our gradient formula assumes h is in [0,1]? No, it's linear.
        # But the objective is only defined/meaningful in [0,1]? 
        # The overlap formula works for any real h, but the problem constraint is [0,1].
        # Evaluating gradient outside [0,1] might lead to large steps.
        # Let's project y_k for stability.
        y_k = _project(h_look, target_sum)
        
        val, grad = _smooth_max_and_grad(y_k, beta)
        
        # Update momentum
        # v_{k+1} = (t / (t+2)) * v_k
        momentum = t / (t + 2.0)
        v = momentum * v - lr_init * (1.0 / (1.0 + t * lr_decay)) * grad
        
        # Update position
        h_next = y_k + v
        
        # Project to feasible set
        h_next = _project(h_next, target_sum)
        
        # Check score
        s = _score(h_next)
        if s < best_score:
            best_score = s
            best_h = h_next.copy()
            
        h_curr = h_next
        
        if time.time() > _DEADLINE - 5:
            break
            
    return best_h, best_score

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h, N/2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        return s

    i = np.arange(N)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Constant
    candidates.append(np.full(N, 0.5))
    
    # 2. Cosine waves
    for freq in range(1, 21):
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            h = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N + phase)
            candidates.append(h)
            
    # 3. Random
    for _ in range(20):
        h = np.random.uniform(0, 1, N)
        candidates.append(h)
        
    # 4. Block patterns
    for block_size in [1, 2, 4, 5, 10, 20, 25, 50]:
        h = np.zeros(N)
        for start in range(0, N, 2*block_size):
            h[start:start+block_size] = 1.0
        candidates.append(h)
        
    # Score and sort candidates
    scored = []
    for h in candidates:
        h_fixed = _project(h, N/2.0)
        s = _score(h_fixed)
        scored.append((s, h_fixed))
        
    scored.sort(key=lambda x: x[0])
    
    for s, h in scored:
        _update_best(h)
        
    # Optimize top candidates using Nesterov
    top_k = min(10, len(scored))
    for rank in range(top_k):
        if time.time() - t0 > _DEADLINE - 10:
            break
        _, h0 = scored[rank]
        
        # Try different betas and learning rates
        for beta in [50.0, 100.0, 200.0]:
            if time.time() - t0 > _DEADLINE - 5:
                break
            # Adaptive steps: use as many as possible
            # Estimate time per step: ~1ms?
            # Let's try 5000 steps max, or until time limit
            h_opt, s_opt = _nesterov_projected(h0, beta, 5000, lr_init=0.01, lr_decay=0.001)
            if s_opt < best_score:
                _update_best(h_opt)
                
    # Final polish on best
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        for beta in [200.0, 500.0]:
            if time.time() - t0 > _DEADLINE - 1:
                break
            h_opt, s_opt = _nesterov_projected(best_h, beta, 5000, lr_init=0.005, lr_decay=0.002)
            if s_opt < best_score:
                _update_best(h_opt)
                
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h, N/2.0)
    return best_h
