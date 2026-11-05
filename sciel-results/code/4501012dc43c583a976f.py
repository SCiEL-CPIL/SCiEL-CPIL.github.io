"""Third autocorrelation inequality: minimize abs(2*N*max(conv(h,h))/sum(h)^2)."""
import numpy as np
import time
from scipy import fft

N = 400

def _objective(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _grad_objective(h):
    """Compute gradient of objective w.r.t. h (for h where objective is finite)."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return None, float('inf')
    conv = np.convolve(h, h)  # length 2N-1
    max_idx = np.argmax(conv)
    obj = abs(2.0 * N * conv[max_idx] / (s * s))
    
    # d(conv[max_idx])/dh[j] = sum over k: h[k] * delta(max_idx - j - (N-1-k))
    # conv[c] = sum_{i+j=c} h[i]*h[j], for c in 0..2N-2
    # d(conv[c])/dh[j] = sum_{i: i+j=c, 0<=i<N} h[i] + sum_{j': j'+j=c, 0<=j'<N} h[j']
    # = sum of h[i] for i in [max(0,c-(N-1)), min(N-1,c)] where i+j=c
    # Actually: conv[c] = sum_{j=0}^{N-1} h[j]*h[c-j] (where 0<=c-j<N)
    # d(conv[c])/dh[j] = h[c-j] if 0<=c-j<N, else 0
    # But also the symmetric term... let me be careful.
    # conv[c] = sum_{i=0}^{N-1} h[i] * h[c-i] where 0<=c-i<=N-1
    # d/dc of this w.r.t. h[j]:
    #   from first factor: delta(i=j) * h[c-j] (if valid c-j)
    #   from second factor: h[j] * delta(c-j=j) = h[j] * delta(c=2j) (if valid j)
    # So d(conv[c])/dh[j] = h[c-j] (if 0<=c-j<N) + h[j] (if c==2j)
    
    c = max_idx
    grad_conv = np.zeros(N)
    # h[c-j] for j where 0 <= c-j < N, i.e., c-N+1 <= j <= c
    j_start = max(0, c - (N - 1))
    j_end = min(N - 1, c)
    for j in range(j_start, j_end + 1):
        grad_conv[j] += h[c - j]
    # Plus the symmetric contribution: when j == c/2, there's an extra h[j]
    # Actually this is already counted. Let me reconsider.
    # conv[c] = sum_{j=0}^{N-1} h[j] h[c-j]
    # d/dh[j'] of conv[c] = h[c-j'] (from the h[j] term when j=j') 
    #                        + sum_{j != j'} 0
    # Wait, no. d/dh[j'] of [h[j]*h[c-j]] summed over j:
    # = h[c-j'] (when j=j', derivative of h[j] is 1)
    #   + sum over j where c-j = j', i.e., j = c-j', derivative of h[c-j] is 1
    # = h[c-j'] (if 0<=c-j'<N) + h[j'] (if 0<=j'<N and c-j'=j'... no)
    # Actually: d/dh[j'] sum_j h[j]h[c-j] = h[c-j'] (from j=j') + sum_j h[j] * d/dh[j'] h[c-j]
    # d/dh[j'] h[c-j] = delta(c-j, j') = delta(j, c-j')
    # So the second term = h[c-j'] (from j = c-j', if 0<=c-j'<N)
    # Total: 2*h[c-j'] if c-j' is valid, plus special case j'=c-j' (i.e., 2j'=c)
    # In that case both terms are the same h[j'], so it's still 2*h[j'] but we should count it once.
    # Hmm, let me just use a numerical check approach for the gradient.
    
    # Actually, simpler: use FFT-based gradient computation
    # Let's use a cleaner formula. For the linear part:
    # conv[c] = <h, shift_h(c)> where shift_h(c)[j] = h[c-j] if valid
    # d(conv[c])/dh[j] = shift_h(c)[j] + h[j] * (1 if c-2j in [0,N-1])... 
    # This is getting confusing. Let me use a different approach.
    
    # Use the fact that the gradient of the max can be computed via subgradient.
    # For optimization, we'll use a soft-max approach instead.
    
    return None, obj

def _soft_max_obj(h, beta):
    """Soft approximation of max using log-sum-exp."""
    conv = np.convolve(h, h)
    m = np.max(conv)
    return np.log(np.sum(np.exp(beta * (conv - m)))) / beta + m

def _soft_max_grad(h, beta):
    """Gradient of soft max of conv(h,h) w.r.t. h."""
    conv = np.convolve(h, h)
    m = np.max(conv)
    weights = np.exp(beta * (conv - m))
    weights /= np.sum(weights)
    # grad of soft_max = sum_k weights[k] * d(conv[k])/dh
    grad = np.zeros(N)
    for k in range(len(conv)):
        w = weights[k]
        if w < 1e-15:
            continue
        # d(conv[k])/dh[j] for j in range
        j_start = max(0, k - (N - 1))
        j_end = min(N - 1, k)
        for j in range(j_start, j_end + 1):
            grad[j] += w * h[k - j]
        # Symmetric term
        if k % 2 == 0:
            j_mid = k // 2
            if 0 <= j_mid < N:
                grad[j_mid] += w * h[j_mid]
    return grad

def _objective_and_grad_soft(h, beta):
    """Compute soft-max objective and its gradient."""
    conv = np.convolve(h, h)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return float('inf'), np.zeros(N)
    
    m = np.max(conv)
    logsumexp = m + np.log(np.sum(np.exp(beta * (conv - m))))
    soft_max_val = logsumexp / beta
    
    # Objective: |2*N*soft_max / s^2|
    obj_sign = np.sign(2.0 * N * soft_max_val / (s * s))
    obj_val = abs(2.0 * N * soft_max_val / (s * s))
    
    # Gradient of soft_max w.r.t. h
    weights = np.exp(beta * (conv - m))
    weights /= np.sum(weights)
    
    grad_soft_max = np.zeros(N)
    for k in range(len(conv)):
        w = weights[k]
        if w < 1e-15:
            continue
        j_start = max(0, k - (N - 1))
        j_end = min(N - 1, k)
        for j in range(j_start, j_end + 1):
            grad_soft_max[j] += w * h[k - j]
        if k % 2 == 0:
            j_mid = k // 2
            if 0 <= j_mid < N:
                grad_soft_max[j_mid] += w * h[j_mid]
    
    # Gradient of |2*N*soft_max/s^2|
    # = obj_sign * 2*N * (grad_soft_max * s^2 - soft_max * 2*s*grad_s) / s^4
    # where grad_s = ones(N)
    grad_s = np.ones(N)
    grad_obj = obj_sign * 2.0 * N / (s * s) * grad_soft_max
    grad_obj -= obj_sign * 4.0 * N * soft_max_val / (s * s * s) * grad_s
    
    return obj_val, grad_obj

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = 2.0
    
    rng = np.random.RandomState(42)
    
    # Strategy: Use gradient-based optimization with soft-max approximation
    # to smoothly optimize the max of autoconvolution.
    
    # Initial candidates
    inits = [
        np.ones(N),
        np.sin(2 * np.pi * np.arange(N) / N),
        np.cos(2 * np.pi * np.arange(N) / N),
        np.sin(2 * np.pi * 2 * np.arange(N) / N),
        np.sin(2 * np.pi * 5 * np.arange(N) / N),
        np.cos(2 * np.pi * 5 * np.arange(N) / N),
        rng.randn(N),
    ]
    
    # Add structured candidates with zero or near-zero sum
    # These have low autoconv peaks relative to sum^2
    for freq in [1, 2, 3, 4, 5, 7, 10, 13, 20, 33]:
        inits.append(np.sin(2 * np.pi * freq * np.arange(N) / N))
        inits.append(np.cos(2 * np.pi * freq * np.arange(N) / N))
    
    # Block patterns
    for split in [100, 133, 200, 267, 300]:
        hb = np.zeros(N)
        hb[:split] = 1.0
        hb[split:] = -1.0
        inits.append(hb)
    
    # Sparse patterns
    for _ in range(5):
        hs = np.zeros(N)
        idx = rng.choice(N, 20, replace=False)
        hs[idx] = rng.randn(20)
        inits.append(hs)
    
    # Evaluate initial candidates
    for h in inits:
        if time.time() > deadline:
            break
        val = _objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Gradient-based optimization from best candidate
    current_h = best_h.copy()
    current_val = best_val
    
    # Annealed soft-max gradient descent
    beta_start = 1.0
    beta_end = 50.0
    n_iters = 300
    lr = 0.01
    
    for it in range(n_iters):
        if time.time() > deadline - 5:
            break
        
        # Anneal beta
        beta = beta_start + (beta_end - beta_start) * (it / max(n_iters - 1, 1))
        
        # Get objective and gradient
        obj, grad = _objective_and_grad_soft(current_h, beta)
        
        if not np.isfinite(obj) or not np.all(np.isfinite(grad)):
            # Reset to best known
            current_h = best_h.copy()
            continue
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        step = lr * current_val / (grad_norm + 1e-10)
        step = min(step, 0.5)
        
        new_h = current_h - step * grad
        
        # Ensure finite
        if not np.all(np.isfinite(new_h)):
            continue
        
        new_val = _objective(new_h)
        
        if new_val < current_val:
            current_h = new_h
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = new_h.copy()
        else:
            # Try smaller step
            new_h2 = current_h - step * 0.1 * grad
            new_val2 = _objective(new_h2)
            if new_val2 < current_val:
                current_h = new_h2
                current_val = new_val2
                if new_val2 < best_val:
                    best_val = new_val2
                    best_h = new_h.copy()
    
    # Multiple restarts with random initializations if time permits
    if time.time() < deadline - 20:
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            h_init = rng.randn(N)
            h_init += 0.1 * np.sin(2 * np.pi * rng.uniform(1, 20) * np.arange(N) / N)
            
            # Quick gradient descent from this init
            curr = h_init.copy()
            curr_val = _objective(curr)
            
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                beta = 5.0 + 10.0 * (it / 100.0)
                obj, grad = _objective_and_grad_soft(curr, beta)
                if not np.isfinite(obj) or not np.all(np.isfinite(grad)):
                    break
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-15:
                    break
                step = 0.01 * curr_val / (grad_norm + 1e-10)
                step = min(step, 0.3)
                new_h = curr - step * grad
                if not np.all(np.isfinite(new_h)):
                    break
                new_val = _objective(new_h)
                if new_val < curr_val:
                    curr = new_h
                    curr_val = new_val
                else:
                    break
            
            if curr_val < best_val:
                best_val = curr_val
                best_h = curr.copy()
    
    # Final check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h
