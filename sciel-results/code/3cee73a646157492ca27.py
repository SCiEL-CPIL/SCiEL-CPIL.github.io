import numpy as np
import time

N = 600
L = 0.5  # length of [-1/4, 1/4]
dx = L / N

def compute_ratio(f):
    """Compute max(f*f) / (int f)^2 for a nonnegative step function."""
    int_f = np.sum(f) * dx
    if int_f <= 0:
        return np.inf
    F = np.fft.rfft(f, n=2*N)
    autoconv = np.fft.irfft(F * F.conj(), n=2*N) * dx
    peak = np.max(autoconv)
    return peak / (int_f * int_f)

def compute_ratio_and_grad(f, beta=100.0):
    """Compute soft-max objective and its gradient."""
    int_f = np.sum(f) * dx
    if int_f <= 0:
        return np.inf, np.zeros(N)
    
    F = np.fft.rfft(f, n=2*N)
    autoconv = np.fft.irfft(F * F.conj(), n=2*N) * dx
    
    # Soft max: log(sum(exp(beta*x))) / beta
    shifted = autoconv * beta
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    logsumexp = (max_shifted + np.log(np.sum(exp_shifted))) / beta
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient of soft-max autoconv w.r.t. f
    # autoconv[k] = sum_i f[i]*f[k-i]*dx
    # d(autoconv[k])/d(f[j]) = dx*(f[k-j] if valid + f[j-k] if valid)
    # grad_f[j] = sum_k weights[k] * d(autoconv[k])/d(f[j])
    #           = dx * (sum_k weights[k]*f[k-j] + sum_k weights[k]*f[j-k])
    # The first sum is correlation of weights and f at lag j
    # The second sum is correlation of weights and f at lag -j
    
    # Use FFT for correlation
    n_corr = 4 * N  # enough padding
    W = np.fft.rfft(weights, n=n_corr)
    Ff = np.fft.rfft(f, n=n_corr)
    
    # corr1[j] = sum_k weights[k] * f[k-j] for j in [0, N-1]
    # This is the cross-correlation, which via FFT is ifft(conj(W) * Ff)
    corr = np.fft.irfft(np.conj(W) * Ff, n=n_corr)
    
    # For linear correlation: corr_full has length 2*N-1
    # corr_full[m] = sum_k weights[k] * f[k - (m - (N-1))]... 
    # Let me think more carefully.
    # np.correlate(weights, f, 'full')[m] = sum_k weights[k] * f[k - m + (N-1)]
    # Actually, the standard definition: (a * b)[n] = sum_k a[k] b[n-k]
    # Cross-correlation: (a ⋆ b)[n] = sum_k a[k] b[k+n]
    
    # Let's just compute it directly since N=600 is small
    grad_f = np.zeros(N)
    for j in range(N):
        # Term 1: sum_k weights[k] * f[k-j] where 0 <= k-j < N
        # i.e., j <= k < j+N
        k1_start = max(0, j)
        k1_end = min(2*N-1, j+N-1)
        if k1_start <= k1_end:
            k1 = np.arange(k1_start, k1_end+1)
            idx1 = k1 - j
            grad_f[j] += np.sum(weights[k1] * f[idx1])
        
        # Term 2: sum_k weights[k] * f[j-k] where 0 <= j-k < N
        # i.e., j-N < k <= j
        k2_start = max(0, j-N+1)
        k2_end = min(2*N-1, j)
        if k2_start <= k2_end:
            k2 = np.arange(k2_start, k2_end+1)
            idx2 = j - k2
            grad_f[j] += np.sum(weights[k2] * f[idx2])
    
    grad_f *= dx
    
    # Objective: logsumexp / int_f^2
    obj = logsumexp / (int_f * int_f)
    
    # Gradient of obj w.r.t. f
    # d/d(f) [logsumexp / int_f^2] = (grad_logsumexp * int_f^2 - logsumexp * 2*int_f * dx) / int_f^4
    # = grad_logsumexp / int_f^2 - 2*logsumexp*dx / int_f^3
    grad_obj = grad_f / (int_f * int_f) - 2 * logsumexp * dx / (int_f ** 3)
    
    return obj, grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Try multiple initializations and keep the best
    best_f = None
    best_score = np.inf
    
    # Initialization 1: Uniform
    f0 = np.ones(N)
    
    # Initialization 2: Sine wave (first harmonic)
    t = np.linspace(0, 1, N, endpoint=False)
    f1 = 1.0 + 0.5 * np.sin(2 * np.pi * t)
    f1 = np.maximum(f1, 0)
    
    # Initialization 3: Gaussian bump
    t2 = np.linspace(-0.5, 0.5, N)
    f2 = np.exp(-4 * t2**2)
    f2 = f2 / np.max(f2)
    
    # Initialization 4: Two bumps
    f3 = np.exp(-16 * (t2 - 0.25)**2) + np.exp(-16 * (t2 + 0.25)**2)
    f3 = f3 / np.max(f3)
    
    # Initialization 5: Cosine (raised cosine)
    f4 = 0.5 + 0.5 * np.cos(2 * np.pi * t2)
    
    inits = [f0, f1, f2, f3, f4]
    
    for f_init in inits:
        if time.time() > deadline:
            break
        
        f = f_init.copy()
        current_score = compute_ratio(f)
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
        
        # Gradient descent with adaptive step size
        lr = 0.01
        patience = 0
        max_patience = 200
        
        for iteration in range(3000):
            if time.time() > deadline:
                break
            
            obj, grad = compute_ratio_and_grad(f, beta=100.0)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            # Take step
            f_new = f - lr * grad
            f_new = np.maximum(f_new, 0)
            
            # Check if improved
            score_new = compute_ratio(f_new)
            
            if score_new < current_score:
                current_score = score_new
                f = f_new.copy()
                patience = 0
                lr *= 0.99
            else:
                patience += 1
                lr *= 0.9
                if patience >= max_patience:
                    break
            
            # Update best
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
        
        # Add random perturbations around the best found so far
        if best_f is not None and time.time() < deadline:
            for _ in range(10):
                if time.time() > deadline:
                    break
                f_pert = best_f + 0.05 * np.random.randn(N)
                f_pert = np.maximum(f_pert, 0)
                score_pert = compute_ratio(f_pert)
                if score_pert < best_score:
                    best_score = score_pert
                    best_f = f_pert.copy()
    
    # Final refinement on the best solution
    if best_f is not None and time.time() < deadline:
        f = best_f.copy()
        current_score = best_score
        lr = 0.005
        patience = 0
        max_patience = 300
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            obj, grad = compute_ratio_and_grad(f, beta=200.0)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            f_new = f - lr * grad
            f_new = np.maximum(f_new, 0)
            
            score_new = compute_ratio(f_new)
            
            if score_new < current_score:
                current_score = score_new
                f = f_new.copy()
                patience = 0
                lr *= 0.99
            else:
                patience += 1
                lr *= 0.95
                if patience >= max_patience:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
    
    return list(best_f)
