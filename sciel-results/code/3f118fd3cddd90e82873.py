"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac)) / (s * s)

def _spectral_optimize(h_init, time_budget):
    """
    Optimize h to minimize max|H[k]|^2 / |H[0]|^2 using gradient descent
    with log-sum-exp smoothing of the max operator.
    """
    t_start = time.time()
    deadline = t_start + time_budget
    h = h_init.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, 1e10
    # Normalize so sum = 1
    h = h / s
    
    n = N
    # Precompute DFT basis vectors for gradient computation
    # H[k] = sum_i h[i] * exp(-2*pi*i*k/N)
    # d|H[k]|^2/dh[i] = 2*Re(H[k] * conj(exp(-2*pi*i*k/N)))
    
    best_h = h.copy()
    best_val = _score(h)
    
    beta = 50.0  # Temperature for log-sum-exp
    lr = 0.01
    
    for step in range(5000):
        if time.time() > deadline - 2:
            break
        
        # Compute DFT
        H = np.fft.fft(h)
        H0 = H[0]
        if abs(H0) < 1e-9:
            break
        
        # |H[k]|^2 for all k
        mag2 = np.abs(H)**2
        mag2[0] = 0  # Exclude DC
        
        # Log-sum-exp approximation of max
        # f = log(sum(exp(beta * mag2))) / beta approximates max(mag2)
        # Gradient: d f / d h[i] = sum_k (beta * exp(beta*mag2[k]) / sum_j exp(beta*mag2[j])) * d(mag2[k])/dh[i]
        
        # For numerical stability
        max_mag2 = np.max(mag2)
        exp_vals = np.exp(beta * (mag2 - max_mag2))
        sum_exp = np.sum(exp_vals)
        
        if sum_exp < 1e-300:
            weights = np.zeros(n)
            weights[np.argmax(mag2)] = 1.0
        else:
            weights = exp_vals / sum_exp  # Softmax weights over frequency bins
        
        # Gradient of the smoothed objective w.r.t. h
        # d(log(sum_k w_k * mag2[k]))/dh[i] where w_k are softmax weights
        # = sum_k w_k * d(mag2[k])/dh[i]
        # d(mag2[k])/dh[i] = 2*Re(H[k] * conj(exp(-2*pi*i*k/n)))
        
        # Vectorized: grad[i] = 2*Re( sum_k w_k * H[k] * conj(exp(-2*pi*i*k/n)) )
        # = 2*Re( conj( sum_k w_k * conj(H[k]) * exp(-2*pi*i*k/n) ) )
        # Let g[k] = w_k * conj(H[k]), then grad[i] = 2*Re( conj( IDFT(g)[i] ) )
        # Actually: sum_k w_k * H[k] * conj(exp(-2*pi*i*k/n)) = sum_k w_k * H[k] * exp(2*pi*i*k/n)
        # This is the inverse DFT of (w_k * H[k]) but with conjugate phase...
        
        # Let's compute: q[i] = sum_k (w_k * H[k]) * exp(2*pi*i*k/n)
        # This is ifft(w * H) * n (depending on convention)
        # np.fft.ifft(X)[i] = (1/n) sum_k X[k] exp(2*pi*i*k/n)
        # So sum_k X[k] exp(2*pi*i*k/n) = n * ifft(X)[i]
        
        g = weights * H
        q = np.fft.ifft(g) * n  # q[i] = sum_k weights[k]*H[k]*exp(2*pi*i*k/n)
        # grad[i] = 2*Re(conj(q[i]))? Let me verify:
        # d|H[k]|^2/dh[i] = 2*Re(H[k] * conj(exp(-2*pi*i*k/n)))
        # = 2*Re(H[k] * exp(2*pi*i*k/n))
        # sum_k w_k * 2*Re(H[k] * exp(2*pi*i*k/n)) = 2*Re(sum_k w_k * H[k] * exp(2*pi*i*k/n))
        # = 2*Re(q[i])
        
        grad = 2.0 * np.real(q)
        
        # Also need to maintain sum(h) = 1 constraint
        # Project gradient to null space of all-ones vector
        grad_mean = np.mean(grad)
        grad_centered = grad - grad_mean
        
        # Update
        h = h - lr * grad_centered
        
        # Renormalize to sum = 1
        s = np.sum(h)
        if abs(s) < 1e-9:
            h = best_h.copy()
            continue
        h = h / s
        
        # Evaluate
        val = _score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            # Increase learning rate slightly
            lr = min(lr * 1.02, 0.1)
        else:
            h = best_h.copy()
            lr = max(lr * 0.95, 0.001)
        
        # Increase beta over time to sharpen the max approximation
        beta = min(beta * 1.001, 200.0)
    
    return best_h, best_val

def _sign_refine(h_init, time_budget):
    """
    Refine using sign flips with FFT-based scoring.
    """
    t_start = time.time()
    deadline = t_start + time_budget
    h = h_init.copy()
    best_h = h.copy()
    best_val = _score(h)
    
    n = N
    fsize = 1
    while fsize < 2*n - 1:
        fsize *= 2
    
    # Precompute FFT of basis vectors for fast score update
    # When we flip h[i], the change in H is -2*h[i]*exp(-2*pi*i*k/N)
    
    for iteration in range(100):
        if time.time() > deadline - 3:
            break
        
        improved = False
        # Try flipping each element
        for idx in range(n):
            if time.time() > deadline - 1:
                break
            
            old_val = h[idx]
            h[idx] = -old_val
            
            val = _score(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                improved = True
            else:
                h[idx] = old_val
        
        if not improved:
            break
    
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    best_score = 1e10
    best_h = np.ones(N) / N
    
    # Baseline
    base_score = _score(best_h)
    best_score = base_score
    best_h = best_h.copy()
    
    # Generate diverse initial candidates
    i = np.arange(N)
    candidates = []
    
    # Chirp sequences
    candidates.append(np.cos(2.0 * np.pi * i**2 / N))
    candidates.append(np.cos(np.pi * i**2 / N))
    candidates.append(np.cos(2.0 * np.pi * i * (i+1) / N))
    candidates.append(np.sin(2.0 * np.pi * i**2 / N))
    candidates.append(np.sin(np.pi * i**2 / N))
    
    # Random
    rng = np.random.default_rng(42)
    for seed in range(5):
        rng_local = np.random.default_rng(seed)
        candidates.append(rng_local.standard_normal(N))
    
    # Evaluate candidates and pick best few for optimization
    cand_scores = []
    for h_c in candidates:
        s = np.sum(h_c)
        if abs(s) < 1e-6:
            continue
        val = _score(h_c)
        cand_scores.append((val, h_c))
    
    cand_scores.sort(key=lambda x: x[0])
    
    # Take top 3 candidates for spectral optimization
    remaining_time = deadline - time.time()
    per_candidate = remaining_time / 4.0
    
    for rank, (cand_val, h_c) in enumerate(cand_scores[:3]):
        if time.time() > deadline - 15:
            break
        
        t_budget = min(per_candidate, deadline - time.time() - 5)
        if t_budget < 5:
            break
        
        h_opt, val_opt = _spectral_optimize(h_c, t_budget)
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
    
    # Final sign refinement
    if time.time() < deadline - 10:
        h_ref, val_ref = _sign_refine(best_h, deadline - time.time() - 5)
        if val_ref < best_score:
            best_score = val_ref
            best_h = h_ref.copy()
    
    # Ensure we don't return worse than baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
