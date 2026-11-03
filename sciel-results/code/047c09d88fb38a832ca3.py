import numpy as np
import time
import math

N = 400
N_FFT = 1024  # Next power of 2 >= 2*N

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=N_FFT)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=N_FFT)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _soft_max_abs_acf(A, beta=100.0):
    """
    Smooth approximation of max(|A[1:]|) using log-sum-exp.
    As beta -> inf, this approaches the true max.
    """
    sidelobes = np.abs(A[1:])
    # Use log-sum-exp for numerical stability
    m = np.max(sidelobes)
    if m == 0:
        return 0.0
    # beta * log(sum(exp(beta * (x - m)))) + m = m + beta * log(sum(exp(beta * (x - m))))
    # But we want max, so: max ≈ m + (1/beta) * log(sum(exp(beta * (x - m))))
    shifted = beta * (sidelobes - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m + log_sum / beta

def _soft_max_abs_acf_grad(s, beta=100.0):
    """
    Compute gradient of soft-max-abs autocorrelation w.r.t. s.
    Uses the chain rule through FFT.
    """
    A = _calc_acf(s)
    sidelobes_abs = np.abs(A[1:])
    
    # Soft-max weights: w_k = exp(beta * |A[k]|) / sum(exp(beta * |A[j]|))
    m = np.max(sidelobes_abs)
    if m == 0:
        return np.zeros(N)
    
    shifted = beta * (sidelobes_abs - m)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient of soft-max w.r.t. A[1:]
    # d(soft_max)/dA[k] = weights[k] * sign(A[k]) for k >= 1
    grad_A = np.zeros(N)
    grad_A[1:] = weights * np.sign(A[1:])
    
    # Now gradient w.r.t. s: 
    # A = ifft(|F|^2) where F = fft(s)
    # dA/ds: Using the fact that for autocorrelation,
    # grad of sum_k g_k * A[k] w.r.t. s = 2 * circ_conv(s, g)
    # where g is the gradient w.r.t. A
    
    F = np.fft.rfft(s.astype(np.float64), n=N_FFT)
    G = np.fft.rfft(grad_A, n=N_FFT)
    grad_s = np.fft.irfft(2 * F * G, n=N_FFT)[:N]
    
    return grad_s

def _gradient_descent_init(n_iter=200, lr=0.05, beta=100.0):
    """
    Use continuous gradient descent on soft-max-abs of autocorrelation sidelobes
    to find a good continuous solution, then threshold to binary.
    """
    x = np.random.uniform(-1, 1, size=N)
    
    for it in range(n_iter):
        # Compute gradient of soft-max-abs objective
        grad = _soft_max_abs_acf_grad(x, beta=beta)
        
        # Adaptive learning rate
        lr_eff = lr / (1.0 + it * 0.01)
        
        # Update
        x -= lr_eff * grad
        
        # Project to [-1, 1]
        x = np.clip(x, -1, 1)
    
    # Threshold to binary
    s = np.where(x >= 0, 1.0, -1.0)
    return s

def _fast_flip_eval(s, i, A_cache, F_cache):
    """
    Evaluate the change in max-abs-ACF when flipping s[i], using cached FFT.
    Returns (new_val, new_A) if flip improves, else (None, None).
    """
    old_si = s[i]
    
    # Flip
    s[i] = -old_si
    
    # Recompute ACF efficiently
    F = np.fft.rfft(s.astype(np.float64), n=N_FFT)
    P = F * np.conj(F)
    A_new = np.fft.irfft(P, n=N_FFT)[:N]
    
    val_new = _max_abs_acf_from_array(A_new)
    
    return val_new, A_new

def _local_search(s, current_val, max_passes=100, time_limit=10.0, deadline=None):
    """Local search by flipping bits with efficient evaluation."""
    start = time.time()
    improved = True
    passes = 0
    no_improve_count = 0
    
    while improved and passes < max_passes and time.time() - start < time_limit:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        passes += 1
        no_improve_count = 0
        
        indices = np.random.permutation(N)
        
        for i in indices:
            if deadline is not None and time.time() > deadline:
                return s, current_val
            
            old_si = s[i]
            s[i] = -old_si
            
            A_new = _calc_acf(s)
            val_new = _max_abs_acf_from_array(A_new)
            
            if val_new < current_val:
                current_val = val_new
                improved = True
                no_improve_count = 0
            else:
                s[i] = old_si
                no_improve_count += 1
        
        # Early termination if no improvement in a full pass
        if not improved:
            break
    
    return s, current_val

def _gradient_polish(s, current_val, n_iter=50, beta=50.0, deadline=None):
    """
    Gradient-based polish on the continuous relaxation, then re-threshold.
    Helps escape shallow local minima.
    """
    x = s.astype(np.float64).copy()
    best_x = x.copy()
    best_val = current_val
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient
        grad = _soft_max_abs_acf_grad(x, beta=beta)
        
        # Adaptive learning rate
        lr = 0.1 / (1.0 + it * 0.05)
        
        # Update
        x -= lr * grad
        x = np.clip(x, -1, 1)
        
        # Evaluate
        s_candidate = np.where(x >= 0, 1.0, -1.0)
        A_candidate = _calc_acf(s_candidate)
        val_candidate = _max_abs_acf_from_array(A_candidate)
        
        if val_candidate < best_val:
            best_val = val_candidate
            best_x = x.copy()
    
    s_final = np.where(best_x >= 0, 1.0, -1.0)
    return s_final, best_val

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Strategy 1: Gradient descent initialization with soft-max objective (multiple times)
    for init_iter in range(8):
        if time.time() > deadline - 10:
            break
        
        s = _gradient_descent_init(n_iter=100, lr=0.08, beta=80.0)
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search refinement
        s, current_val = _local_search(s, current_val, max_passes=50, time_limit=4.0, deadline=deadline)
        
        # Gradient polish to escape local minima
        if time.time() < deadline - 5:
            s, current_val = _gradient_polish(s, current_val, n_iter=30, beta=50.0, deadline=deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # Strategy 2: Random restarts with refinement
    while time.time() < deadline - 3:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search refinement
        s, current_val = _local_search(s, current_val, max_passes=30, time_limit=2.0, deadline=deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # Final polish on best solution
    if time.time() < deadline - 2 and best_val < float('inf'):
        best_seq, best_val = _gradient_polish(best_seq, best_val, n_iter=40, beta=30.0, deadline=deadline)
    
    return best_seq
