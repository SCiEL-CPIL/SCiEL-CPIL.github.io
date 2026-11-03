"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_smooth_max(h, beta=50.0):
    """
    Compute a smooth approximation of max|autoconv| using log-sum-exp in the
    frequency domain. The autoconvolution is IDFT(|H|^2). We use a smooth max
    over the time-domain samples.
    """
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    power_spec = H * np.conj(H)  # |H|^2, real non-negative
    ac = np.fft.irfft(power_spec, fsize)
    # Smooth max of |ac| using log-sum-exp
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    log_sum = np.log(np.sum(np.exp(beta * (abs_ac - m)))) / beta + m
    return log_sum

def _spectral_gradient_descent():
    """
    Optimize h to minimize max|autoconv| / (sum h)^2 using gradient descent
    in the time domain with a smooth max approximation. The gradient of the
    smooth max autoconvolution is computed via FFT.
    """
    t_start = time.time()
    deadline = t_start + 100.0
    
    i = np.arange(N)
    
    # Initialize with a chirp
    h = np.cos(2.0 * np.pi * i**2 / N)
    h = h / np.sum(h)
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Learning rate schedule
    lr = 0.01
    beta = 20.0  # Smoothness parameter
    
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        
        # Normalize to sum = 1
        s = np.sum(h)
        if abs(s) < 1e-9:
            break
        h_norm = h / s
        
        # Compute smooth max autoconvolution and its gradient
        n = len(h_norm)
        size = 2 * n - 1
        fsize = 1
        while fsize < size:
            fsize *= 2
        
        H = np.fft.rfft(h_norm, fsize)
        power_spec = H * np.conj(H)
        ac = np.fft.irfft(power_spec, fsize)
        abs_ac = np.abs(ac)
        m = np.max(abs_ac)
        
        if m < 1e-15:
            break
        
        # Softmax weights for smooth max
        weights = np.exp(beta * (abs_ac - m))
        weights /= np.sum(weights)
        
        # Gradient of smooth max w.r.t. ac
        grad_ac = weights * np.sign(ac)  # d|ac|/dac * weight
        
        # Gradient w.r.t. H: ac = IDFT(|H|^2), so dac/dH_k = 2*conj(H_k)*IDFT basis
        # More directly: grad_H = 2*H * DFT(grad_ac)
        grad_H_full = np.fft.rfft(grad_ac, fsize)
        grad_H = 2.0 * H * grad_H_full
        
        # Gradient w.r.t. h: h = IDFT(H), so grad_h = IDFT(grad_H)
        grad_h = np.fft.irfft(grad_H, fsize)[:n]
        
        # Update
        h = h_norm - lr * grad_h
        
        # Project back to sum=1
        s_new = np.sum(h)
        if abs(s_new) < 1e-9:
            break
        h = h / s_new
        
        # Evaluate
        score = _score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Decay learning rate
        lr *= 0.995
        beta = min(beta * 1.02, 100.0)
    
    return best_h, best_score

def _eigenvalue_minimization():
    """
    Directly minimize the largest eigenvalue ratio using spectral methods.
    The circulant matrix C_h has eigenvalues H[k] = DFT(h)[k].
    We minimize max_k |H[k]|^2 subject to sum(h) = 1, which is equivalent
    to minimizing the condition number of the spectral distribution.
    """
    t_start = time.time()
    deadline = t_start + 80.0
    
    i = np.arange(N)
    
    # Try various initializations
    initializations = [
        np.cos(2.0 * np.pi * i**2 / N),
        np.cos(np.pi * i * (i + 1) / N),
        np.sin(2.0 * np.pi * i**2 / N),
        np.ones(N),
    ]
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    for h_init in initializations:
        if time.time() > deadline - 10:
            break
        
        h = h_init.copy()
        s = np.sum(h)
        if abs(s) < 1e-9:
            continue
        h = h / s
        
        # Gradient descent on max|H[k]|^2 / (sum h)^2
        lr = 0.005
        for iteration in range(300):
            if time.time() > deadline - 5:
                break
            
            s = np.sum(h)
            if abs(s) < 1e-9:
                break
            h_norm = h / s
            
            # Compute H
            H = np.fft.rfft(h_norm)
            H_abs_sq = np.abs(H)**2
            
            # Find peak frequency
            k_max = np.argmax(H_abs_sq)
            
            # Smooth approximation: use top few peaks
            top_k = np.argsort(H_abs_sq)[-5:]
            smooth_max = np.mean(H_abs_sq[top_k])
            
            # Gradient of smooth_max w.r.t. H
            weights = np.zeros_like(H_abs_sq)
            weights[top_k] = 1.0 / len(top_k)
            grad_H = 2.0 * weights * H
            
            # Gradient w.r.t. h
            grad_h = np.fft.irfft(grad_H)
            
            # Update
            h = h_norm - lr * grad_h
            s_new = np.sum(h)
            if abs(s_new) < 1e-9:
                break
            h = h / s_new
            
            score = _score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
            
            lr *= 0.99
    
    return best_h, best_score

def _local_search(h, deadline):
    """Local search via sign flips and small perturbations."""
    h_refined = h.copy()
    best_local = _score(h_refined)
    
    for _ in range(10):
        improved = False
        for idx in range(N):
            if time.time() > deadline - 2:
                break
            h_refined[idx] = -h_refined[idx]
            new_val = _score(h_refined)
            if new_val < best_local:
                best_local = new_val
                improved = True
            else:
                h_refined[idx] = -h_refined[idx]
        if not improved:
            break
    
    return h_refined, best_local

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Baseline
    base_score = _score(best_h)
    best_score = min(best_score, base_score)
    
    # Strategy 1: Spectral gradient descent with smooth max
    if time.time() < deadline - 50:
        h_spec, score_spec = _spectral_gradient_descent()
        if score_spec < best_score:
            best_score = score_spec
            best_h = h_spec.copy()
    
    # Strategy 2: Eigenvalue minimization
    if time.time() < deadline - 25:
        h_eig, score_eig = _eigenvalue_minimization()
        if score_eig < best_score:
            best_score = score_eig
            best_h = h_eig.copy()
    
    # Final local search
    if time.time() < deadline - 5:
        h_refined, score_local = _local_search(best_h, deadline)
        if score_local < best_score:
            best_score = score_local
            best_h = h_refined.copy()
    
    return best_h

# EVOLVE-BLOCK-END
