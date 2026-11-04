import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _project_simplex(v):
    """Project vector v onto the probability simplex."""
    N = len(v)
    u = np.zeros(N)
    b = np.zeros(N)
    rpos = 0
    rsum = 0.0
    rho = 0
    vsorted = np.sort(v)[::-1]
    for i in range(N):
        vtest = vsorted[i] - rsum / (i + 1.0)
        if vtest > 0:
            rpos = i + 1
            rsum += vsorted[i]
        else:
            break
    if rpos > 0:
        rho = (rsum - 1.0) / rpos
        for i in range(rpos):
            u[i] = vsorted[i] - rho
        # Sort back to original order
        # This simple sort-back is O(N log N) which is fine for N=50
        # Better: use argsort
        idx = np.argsort(v)[::-1]
        u_orig = np.zeros(N)
        for i in range(rpos):
            u_orig[idx[i]] = vsorted[i] - rho
        return u_orig
    else:
        return v / np.sum(v) # Fallback, shouldn't happen if v has positive sum

def _ratio_grad(h, beta=500.0):
    """
    Compute ratio and gradient for normalized h (sum=1, h>=0).
    Returns (ratio, gradient).
    """
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0, np.zeros(N)
    
    u = h / l1
    l2_sq = np.sum(u ** 2)
    conv = _autoconv(u)
    
    # Soft-max
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        return 0.0, np.zeros(N)
        
    p = exp_shifted / sum_exp
    soft_max = max_val + np.log(sum_exp) / beta
    
    if soft_max <= 0:
        return 0.0, np.zeros(N)
        
    ratio = l2_sq / soft_max
    
    # Gradient of ratio wrt u
    # J = l2_sq / soft_max
    # dJ/du = (2u * soft_max - l2_sq * d(soft_max)/du) / soft_max^2
    
    # d(soft_max)/du
    # soft_max = max_val + (1/beta) log(sum exp(beta conv))
    # d(soft_max)/du_k = (1/beta) * sum_j p_j * beta * d(conv_j)/du_k
    #                   = sum_j p_j * d(conv_j)/du_k
    
    # conv_j = sum_i u_i u_{j-i}
    # d(conv_j)/du_k = 2 u_{j-k} (with bounds)
    # Let's compute this efficiently.
    # grad_softmax[k] = sum_j p_j * 2 u_{j-k}
    # This is 2 * (p * u) but with specific alignment.
    # p has length 2N-1. u has length N.
    # conv = np.convolve(u, u, 'full')
    # The derivative of conv wrt u is related to convolving p with u?
    # Let's check:
    # grad_softmax = 2 * np.convolve(p, u, mode='same')? No.
    
    # Let's do it explicitly for N=50, it's fast enough.
    # Or use FFT.
    # d(conv_j)/du_k = u_{j-k} + u_{j-k} = 2 u_{j-k} if j-k is in [0, N-1].
    # So grad_softmax[k] = 2 * sum_{j} p_j u_{j-k}
    
    # We can compute this by padding u and convolving with p?
    # sum_j p_j u_{j-k} is cross-correlation of p and u.
    # corr[k] = sum_j p_j u_{j-k}
    # This is equivalent to convolving p with reversed u?
    # Let's just use a loop or vectorized numpy for N=50.
    
    grad_softmax = np.zeros(N)
    # p indices 0..2N-2
    # u indices 0..N-1
    # For each k in 0..N-1:
    # grad_softmax[k] = 2 * sum_{j} p[j] * u[j-k] where 0 <= j-k < N
    
    # Vectorized approach:
    # Create a matrix M where M[k, j] = u[j-k] if valid else 0
    # Then grad_softmax = 2 * (p @ M)
    
    # Simpler:
    # u_padded = np.pad(u, (0, 2*N-1))
    # We want sum_j p_j u_{j-k}
    # This is the k-th element of the cross-correlation?
    # Let's use FFT for speed if needed, but N=50 is small.
    
    # Let's try:
    # conv_pu = np.convolve(p, u, mode='full')
    # Does this give what we want?
    # conv_pu[m] = sum_i p_i u_{m-i}
    # We want sum_j p_j u_{j-k}.
    # Let m = j-k? No.
    # Let's just loop, it's safe.
    for k in range(N):
        # j ranges such that 0 <= j-k < N => k <= j < k+N
        # j also in [0, 2N-2]
        j_start = max(0, k)
        j_end = min(2*N-1, k+N)
        if j_start < j_end:
            # u indices: j-k
            u_slice = u[j_start - k : j_end - k]
            p_slice = p[j_start : j_end]
            grad_softmax[k] = 2.0 * np.dot(p_slice, u_slice)
            
    # Now combine
    # grad_ratio = (2u * soft_max - l2_sq * grad_softmax) / soft_max**2
    grad_ratio = (2.0 * u * soft_max - l2_sq * grad_softmax) / (soft_max ** 2)
    
    # This is the gradient wrt u.
    # We are optimizing h, but we can just optimize u and return u (or h=u).
    # Since the score is scale invariant, returning u (sum=1) is fine.
    
    return ratio, grad_ratio

def _local_optimize_analytic(h0, start_time, max_iter=2000, beta=500.0):
    """
    Optimize h using analytic gradient and projected gradient descent.
    h0 is initial guess (will be normalized).
    """
    h = np.maximum(h0, 0.0).copy()
    l1 = np.sum(h)
    if l1 == 0:
        h = np.ones(N) / N
    else:
        h = h / l1
        
    best_h = h.copy()
    best_score, _ = _ratio_grad(h, beta)
    
    lr = 0.01
    for i in range(max_iter):
        if _time_left(start_time) < 2.0:
            break
            
        score, grad = _ratio_grad(h, beta)
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
        # Gradient ascent
        # h_new = h + lr * grad
        # Project to simplex
        h_new = h + lr * grad
        h_new = _project_simplex(h_new)
        
        # Simple adaptive learning rate: if score decreases, reduce lr
        score_new, _ = _ratio_grad(h_new, beta)
        if score_new < score:
            lr *= 0.5
        else:
            h = h_new
            # Increase lr slightly if we are far from convergence?
            # Keep it simple for now.
            
    return best_h, best_score

def construct():
    start_time = time.time()
    
    best_h = np.ones(N) / N
    best_score = _ratio_grad(best_h, beta=100.0)[0]
    
    # Generate diverse initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Single Gaussians
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    for center in [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]:
        for width in [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]:
            h = np.exp(-((x - center) ** 2) / (2 * width ** 2))
            initializations.append(h)
            
    # Lorentzians
    for center in [-0.15, -0.1, 0.0, 0.1, 0.15]:
        for gamma in [0.03, 0.05, 0.08, 0.1, 0.15]:
            h = (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)
            initializations.append(h)
            
    # Power laws
    for k in [2, 4, 6, 8, 10]:
        h_right = np.zeros(N)
        mask = x > 0
        h_right[mask] = x[mask] ** k
        initializations.append(h_right)
        
        h_left = np.zeros(N)
        mask = x < 0
        h_left[mask] = (-x[mask]) ** k
        initializations.append(h_left)
        
        h_both = np.abs(x) ** k
        initializations.append(h_both)
        
    # Two Gaussians
    for c1, c2 in [(-0.15, 0.15), (-0.1, 0.1), (-0.2, 0.2), (-0.12, 0.12), (-0.08, 0.08)]:
        for w in [0.04, 0.06, 0.08, 0.1, 0.12]:
            h1 = np.exp(-((x - c1) ** 2) / (2 * w ** 2))
            h2 = np.exp(-((x - c2) ** 2) / (2 * w ** 2))
            initializations.append(h1 + h2)
            
    # Random
    rng = np.random.RandomState(42)
    for _ in range(20):
        initializations.append(rng.random(N))
        
    # Sparse random
    for _ in range(10):
        h_sparse = np.zeros(N)
        idx = rng.choice(N, size=rng.randint(5, 15), replace=False)
        h_sparse[idx] = rng.random(len(idx))
        initializations.append(h_sparse)
        
    # Sort initializations by score
    scored_inits = []
    for h_init in initializations:
        h_init = np.maximum(h_init, 0.0)
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        s = _ratio_grad(h_init, beta=100.0)[0]
        scored_inits.append((s, h_init))
        
    scored_inits.sort(key=lambda x: -x[0])
    
    if scored_inits:
        if scored_inits[0][0] > best_score:
            best_score = scored_inits[0][0]
            best_h = scored_inits[0][1].copy()
            
    # Optimize from top initializations
    top_k = min(20, len(scored_inits))
    for i in range(top_k):
        if _time_left(start_time) < 5.0:
            break
            
        s_init, h_init = scored_inits[i]
        
        # Use analytic gradient optimization
        h_opt, s_opt = _local_optimize_analytic(h_init, start_time, max_iter=1000, beta=500.0)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
            
    # Final polish
    if _time_left(start_time) > 10.0:
        for beta in [500.0, 1000.0]:
            if _time_left(start_time) < 5.0:
                break
            h_polish, s_polish = _local_optimize_analytic(best_h, start_time, max_iter=2000, beta=beta)
            if s_polish > best_score:
                best_score = s_polish
                best_h = h_polish.copy()
                
    # Return best_h (normalized)
    return best_h.tolist()
