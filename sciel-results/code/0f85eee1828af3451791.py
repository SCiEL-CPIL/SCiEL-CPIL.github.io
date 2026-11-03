"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Structural change: spectral-parameterized optimization with Nesterov acceleration.
Uses FFT-based overlap computation and optimizes in the frequency domain."""

import numpy as np
import time

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill


def _autocorr(h):
    """Compute autocorrelation r_k = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 using FFT."""
    n = len(h)
    n_fft = 2 * n
    H = np.fft.rfft(h, n_fft)
    acf = np.fft.irfft(H * np.conj(H), n_fft)[:n]
    # Normalize: acf[k] gives sum_{i} h[i]*h[(i-k) mod n_fft] but we want linear
    # Actually for zero-padded FFT, acf[k] = sum_{i=k}^{n-1} h[i]*h[i-k] for k>=0
    # Let me verify: with zero padding, the circular convolution becomes linear
    # r[k] = sum_{j=0}^{n-1-k} h[j+k] * h[j] = sum_{i=k}^{n-1} h[i]*h[i-k]
    # The FFT autocorrelation of zero-padded h gives exactly this
    return acf


def _overlap_values(h):
    """Compute overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h[i]*h[i-k])
    """
    n = len(h)
    acf = _autocorr(h)
    # acf[k] = sum_{i=k}^{n-1} h[i]*h[i-k]
    C = (2.0 / n) * (n / 2.0 - acf)
    return C


def _score(h):
    """Return the maximum overlap value."""
    C = _overlap_values(h)
    return np.max(C)


def _project(h, target_sum):
    """Project h onto {0<=h[i]<=1, sum(h)=target_sum}."""
    h = np.clip(h, 0.0, 1.0)
    lo, hi = -1.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_try = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_try)
        if s > target_sum:
            hi = mid
        else:
            lo = mid
    h_try = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    return h_try


def _smooth_max_and_grad(C, beta):
    """Compute smooth max approximation and its gradient w.r.t. C."""
    max_C = np.max(C)
    exp_vals = np.exp(beta * (C - max_C))
    weights = exp_vals / np.sum(exp_vals)
    smooth_val = max_C + np.log(np.sum(exp_vals)) / beta
    grad_C = weights
    return smooth_val, grad_C


def _smooth_obj_and_grad(h, beta):
    """Compute smooth objective value and gradient w.r.t. h."""
    n = len(h)
    C = _overlap_values(h)
    smooth_val, grad_C = _smooth_max_and_grad(C, beta)
    
    # Gradient of C_k w.r.t. h[i]
    # C_k = (2/n) * (n/2 - sum_{i=k}^{n-1} h[i]*h[i-k])
    # dC_k/dh[j] = (2/n) * (-sum_{i=k}^{n-1} [delta_{j,i}*h[i-k] + delta_{j,i-k}*h[i]])
    # For j in [k, n-1]: dC_k/dh[j] = (2/n) * (-h[j-k] - h[j])  if j>=k
    # Wait, let me be more careful.
    # sum_{i=k}^{n-1} h[i]*h[i-k]
    # d/dh[j] of this sum:
    # If j >= k and j-k >= 0: term h[j]*h[j-k] appears, derivative is h[j-k]
    # Also if j-k is in [k, n-1]... no, the index is i-k, so i = j means i-k = j-k
    # The sum is over i from k to n-1 of h[i]*h[i-k]
    # For a fixed j, h[j] appears when i=j (if j>=k) giving h[j-k], 
    # and when i-k=j, i.e., i=j+k (if j+k <= n-1) giving h[j+k]
    # So d/dh[j] sum = h[j-k] (if j>=k) + h[j+k] (if j+k<n)
    
    grad_h = np.zeros(n)
    for k in range(n):
        gk = grad_C[k] * (2.0 / n)
        # For each j, contribution to dC_k/dh[j]
        # j ranges 0..n-1
        # h[j] appears in sum_{i=k}^{n-1} h[i]*h[i-k] when:
        #   i=j (if j>=k): contributes h[j-k]
        #   i-k=j, i=j+k (if j+k<n): contributes h[j+k]
        # So dC_k/dh[j] = -gk * (h[j-k] if j>=k else 0 + h[j+k] if j+k<n else 0)
        
        # Vectorized: for all j
        # term1: j>=k, value h[j-k]
        # term2: j+k<n, value h[j+k]
        
        # Let's do it with slices
        # For j in [k, n-1]: dC_k/dh[j] includes -gk * h[j-k]
        if k < n:
            # j from k to n-1: h[j-k] where j-k from 0 to n-1-k
            grad_h[k:] -= gk * h[:n-k]
        # For j in [0, n-1-k]: dC_k/dh[j] includes -gk * h[j+k]
        if n - k > 0:
            grad_h[:n-k] -= gk * h[k:]
    
    return smooth_val, grad_h


def _nesterov_projected(h, target_sum, betas, max_iters_total=50000):
    """Nesterov accelerated projected gradient descent in h-space."""
    n = len(h)
    best_h = h.copy()
    best_score = _score(h)
    
    h_curr = h.copy()
    h_prev = h.copy()
    
    elapsed_start = time.time()
    it = 0
    
    for beta in betas:
        if time.time() > _DEADLINE - 15:
            break
        
        # Adaptive step size for this beta
        # Lipschitz constant of gradient is roughly O(N^2) for the quadratic
        # Use a conservative step
        lr = 1.0 / (beta * n)
        
        t_k = 1.0
        h_curr = h_prev.copy()
        
        max_iters_beta = max(1000, max_iters_total // len(betas))
        
        for it_beta in range(max_iters_beta):
            if time.time() > _DEADLINE - 15:
                break
            
            # Nesterov extrapolation
            t_new = (1 + np.sqrt(1 + 4 * t_k * t_k)) / 2
            h_extrap = h_curr + ((t_k - 1) / t_new) * (h_curr - h_prev)
            
            # Compute gradient at extrapolated point
            val, grad = _smooth_obj_and_grad(h_extrap, beta)
            
            # Gradient step
            h_new = h_extrap - lr * grad
            h_new = _project(h_new, target_sum)
            
            # Update
            h_prev = h_curr
            h_curr = h_new
            t_k = t_new
            
            # Check convergence
            if np.max(np.abs(h_new - h_prev)) < 1e-12:
                break
            
            it += 1
            if it >= max_iters_total:
                break
        
        # Evaluate and update best
        score = _score(h_curr)
        if score < best_score:
            best_score = score
            best_h = h_curr.copy()
        
        if time.time() > _DEADLINE - 15:
            break
    
    return best_h, best_score


def _spectral_to_h(coeffs):
    """Convert spectral coefficients to h vector.
    We parameterize h[i] = 0.5 + sum_k a_k * cos(2*pi*k*i/N) + b_k * sin(2*pi*k*i/N)
    But we need 0<=h[i]<=1, so we clip and project.
    """
    n = len(coeffs)
    # coeffs contains [mean_offset, a_1, b_1, a_2, b_2, ...]
    h = np.zeros(n)
    offset = coeffs[0]
    for k in range(1, n):
        idx = 2 * k - 1
        if idx >= len(coeffs):
            break
        a = coeffs[idx]
        b = coeffs[idx + 1] if idx + 1 < len(coeffs) else 0.0
        h += a * np.cos(2 * np.pi * k * np.arange(n) / n) + b * np.sin(2 * np.pi * k * np.arange(n) / n)
    h = h + offset
    return h


def _h_to_spectral(h):
    """Convert h to spectral coefficients."""
    n = len(h)
    offset = np.mean(h) - 0.5
    coeffs = np.array([offset])
    for k in range(1, n):
        a = (2.0 / n) * np.sum(h * np.cos(2 * np.pi * k * np.arange(n) / n))
        b = (2.0 / n) * np.sum(h * np.sin(2 * np.pi * k * np.arange(n) / n))
        coeffs = np.append(coeffs, [a, b])
    return coeffs


def _project_coeffs(coeffs, target_sum):
    """Project spectral coefficients to satisfy constraints."""
    h = _spectral_to_h(coeffs)
    h = _project(h, target_sum)
    return _h_to_spectral(h)


def _nesterov_spectral(h_init, target_sum, betas, max_iters_total=50000):
    """Nesterov optimization in spectral space."""
    n = len(h_init)
    coeffs = _h_to_spectral(h_init)
    best_h = h_init.copy()
    best_score = _score(h_init)
    
    c_curr = coeffs.copy()
    c_prev = coeffs.copy()
    t_k = 1.0
    
    elapsed_start = time.time()
    it = 0
    
    for beta in betas:
        if time.time() > _DEADLINE - 15:
            break
        
        lr = 0.01 / beta
        
        max_iters_beta = max(500, max_iters_total // len(betas))
        
        for it_beta in range(max_iters_beta):
            if time.time() > _DEADLINE - 15:
                break
            
            # Nesterov extrapolation in spectral space
            t_new = (1 + np.sqrt(1 + 4 * t_k * t_k)) / 2
            c_extrap = c_curr + ((t_k - 1) / t_new) * (c_curr - c_prev)
            
            # Convert to h-space for gradient computation
            h_extrap = _spectral_to_h(c_extrap)
            h_extrap = _project(h_extrap, target_sum)
            
            # Compute gradient in h-space
            val, grad_h = _smooth_obj_and_grad(h_extrap, beta)
            
            # Project gradient back to spectral space
            # grad_coeffs[0] = sum(grad_h)  (since offset affects all h[i] equally)
            # grad_coeffs[2k-1] = (2/n) * sum(grad_h * cos(2*pi*k*i/n))
            # grad_coeffs[2k] = (2/n) * sum(grad_h * sin(2*pi*k*i/n))
            grad_coeffs = np.zeros_like(c_extrap)
            grad_coeffs[0] = np.sum(grad_h)
            for k in range(1, n):
                idx = 2 * k - 1
                if idx >= len(grad_coeffs):
                    break
                grad_coeffs[idx] = (2.0 / n) * np.sum(grad_h * np.cos(2 * np.pi * k * np.arange(n) / n))
                if idx + 1 < len(grad_coeffs):
                    grad_coeffs[idx + 1] = (2.0 / n) * np.sum(grad_h * np.sin(2 * np.pi * k * np.arange(n) / n))
            
            # Gradient step in spectral space
            c_new = c_extrap - lr * grad_coeffs
            # Project back
            h_new = _spectral_to_h(c_new)
            h_new = _project(h_new, target_sum)
            c_new = _h_to_spectral(h_new)
            
            c_prev = c_curr
            c_curr = c_new
            t_k = t_new
            
            if np.max(np.abs(c_new - c_prev)) < 1e-12:
                break
            
            it += 1
            if it >= max_iters_total:
                break
        
        # Evaluate
        h_curr = _spectral_to_h(c_curr)
        h_curr = _project(h_curr, target_sum)
        score = _score(h_curr)
        if score < best_score:
            best_score = score
            best_h = h_curr.copy()
        
        if time.time() > _DEADLINE - 15:
            break
    
    return best_h, best_score


def _construct_direct():
    """Direct optimization in h-space with multiple restarts."""
    target_sum = N / 2.0
    betas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    best_h = None
    best_score = np.inf
    
    # Multiple initial guesses
    inits = []
    
    # 1. Alternating blocks: 11110000 pattern
    h1 = np.zeros(N)
    for i in range(N):
        h1[i] = 1.0 if (i % 8) < 4 else 0.0
    inits.append(h1)
    
    # 2. Random
    rng = np.random.default_rng(42)
    h2 = rng.random(N)
    inits.append(h2)
    
    # 3. Sinusoidal
    h3 = 0.5 + 0.3 * np.sin(2 * np.pi * 3 * np.arange(N) / N)
    inits.append(h3)
    
    # 4. Another block pattern: 1100 repeated
    h4 = np.zeros(N)
    for i in range(N):
        h4[i] = 1.0 if (i % 4) < 2 else 0.0
    inits.append(h4)
    
    for h_init in inits:
        if time.time() > _DEADLINE - 20:
            break
        
        h_init = _project(h_init.copy(), target_sum)
        
        # Try direct Nesterov
        h_res, score = _nesterov_projected(h_init, target_sum, betas, max_iters_total=30000)
        if score < best_score:
            best_score = score
            best_h = h_res.copy()
    
    # Also try spectral
    if time.time() < _DEADLINE - 20:
        for h_init in inits[:2]:
            if time.time() > _DEADLINE - 15:
                break
            h_init = _project(h_init.copy(), target_sum)
            h_res, score = _nesterov_spectral(h_init, target_sum, betas[:6], max_iters_total=20000)
            if score < best_score:
                best_score = score
                best_h = h_res.copy()
    
    if best_h is None:
        best_h = inits[0]
        best_h = _project(best_h, target_sum)
    
    # Final local polish
    if time.time() < _DEADLINE - 5:
        for _ in range(100):
            if time.time() > _DEADLINE - 5:
                break
            C = _overlap_values(best_h)
            max_idx = np.argmax(C)
            improved = False
            for i in range(N):
                j = (i + max_idx) % N
                if i == j:
                    continue
                delta = 0.05
                if best_h[i] < best_h[j]:
                    h_try = best_h.copy()
                    h_try[i] = min(1.0, best_h[i] + delta)
                    h_try[j] = max(0.0, best_h[j] - delta)
                else:
                    h_try = best_h.copy()
                    h_try[i] = max(0.0, best_h[i] - delta)
                    h_try[j] = min(1.0, best_h[j] + delta)
                
                if abs(np.sum(h_try) - target_sum) > 1e-6:
                    continue
                
                score_try = _score(h_try)
                if score_try < best_score - 1e-8:
                    best_h = h_try
                    best_score = score_try
                    improved = True
                    break
            if not improved:
                break
    
    return best_h


def construct():
    deadline = time.time() + 110.0
    target_sum = N / 2.0
    
    # Global deadline guard
    global _DEADLINE
    _DEADLINE = deadline
    
    result = _construct_direct()
    
    # Ensure validity
    result = np.clip(result, 0.0, 1.0)
    # Fix sum
    result = _project(result, target_sum)
    
    return result
