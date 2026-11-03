"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    # Initial guess: raised cosine / Hann window shape (smooth, nonnegative)
    # This gives a good starting point with low sidelobes
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Use a smooth bump: product of cosines or Gaussian
    # Gaussian-like: exp(-a * t^2)
    sigma = 0.1
    f = np.exp(-0.5 * (x / sigma) ** 2)
    f = np.maximum(f, 0.0)
    
    # Normalize so that integral = 1 (discrete approximation)
    dx = 0.5 / N
    f = f / (np.sum(f) * dx)
    
    best_f = f.copy()
    best_score = np.inf
    
    # Helper: compute autoconvolution peak and score
    def compute_score(f_arr):
        # Autoconvolution via FFT
        n_fft = 2048  # power of 2 >= 2*N
        F = np.fft.fft(f_arr, n=n_fft)
        autoconv = np.fft.ifft(F * F)
        autoconv = np.real(autoconv[:N])  # take real part, first N samples
        peak = np.max(np.abs(autoconv))
        # Normalize: score is the peak of autoconvolution
        # With ∫f = 1, the peak of f*f is the value we minimize
        return peak, autoconv
    
    # Initial score
    score, _ = compute_score(f)
    best_score = score
    best_f = f.copy()
    
    # Gradient-based optimization using soft-max surrogate
    # Objective: minimize log-sum-exp(beta * autoconv) as smooth approximation to max(autoconv)
    # Then project to nonnegative cone
    
    def objective_and_grad(f_arr):
        """Compute soft-max objective and gradient."""
        n_fft = 2048
        F = np.fft.fft(f_arr, n=n_fft)
        autoconv = np.fft.ifft(F * F)
        autoconv = np.real(autoconv[:N])
        
        beta = 50.0  # temperature for soft-max
        # Soft-max: (1/beta) * log(sum(exp(beta * x)))
        # For numerical stability
        x_max = np.max(autoconv)
        log_sum_exp = x_max + np.log(np.sum(np.exp(beta * (autoconv - x_max))))
        obj = log_sum_exp / beta
        
        # Gradient of soft-max w.r.t. autoconv: softmax(autoconv * beta)
        weights = np.exp(beta * (autoconv - x_max))
        weights = weights / np.sum(weights)
        
        # Gradient of autoconv w.r.t. f: autoconv is f*f (circular via FFT)
        # d(autoconv)/df = convolution with f (but circular)
        # In frequency domain: autoconv = ifft(fft(f)^2)
        # d(autoconv)/df = ifft(2 * fft(f) * conj(fft(f))) ... no
        # Actually: autoconv[k] = sum_j f[j] f[(k-j) mod N]
        # d(autoconv)/df[m] = sum_k weights[k] * d(autoconv[k])/df[m]
        # d(autoconv[k])/df[m] = f[(k-m) mod N]
        
        # So gradient[m] = sum_k weights[k] * f[(k-m) mod N]
        # This is the circular cross-correlation of weights with f
        
        # Compute via FFT: grad = ifft(fft(weights) * conj(fft(f)))
        Fw = np.fft.fft(weights, n=n_fft)
        Ff = np.fft.fft(f_arr, n=n_fft)
        grad_autoconv = np.fft.ifft(Fw * np.conj(Ff))
        grad_autoconv = np.real(grad_autoconv[:N])
        
        # The objective gradient is the same since we're directly optimizing
        grad = grad_autoconv
        
        return obj, grad
    
    # Iterative optimization with projected gradient
    lr = 0.01
    beta = 50.0
    iterations = 0
    max_iterations = 5000
    
    while iterations < max_iterations:
        if time.time() > deadline:
            break
        
        # Compute objective and gradient
        obj, grad = objective_and_grad(f)
        
        # Update: f_new = f - lr * grad
        f_new = f - lr * grad
        
        # Project onto nonnegative cone
        f_new = np.maximum(f_new, 0.0)
        
        # Renormalize: ensure integral = 1
        integral = np.sum(f_new) * dx
        if integral > 1e-10:
            f_new = f_new / integral
        else:
            f_new = f.copy()
        
        # Check convergence
        if np.max(np.abs(f_new - f)) < 1e-8:
            f = f_new
            break
        
        f = f_new
        iterations += 1
        
        # Track best
        cur_score, _ = compute_score(f)
        if cur_score < best_score:
            best_score = cur_score
            best_f = f.copy()
    
    # Final polish: a few more iterations with smaller learning rate
    if time.time() < deadline - 10.0:
        lr2 = 0.001
        for _ in range(200):
            if time.time() > deadline:
                break
            obj, grad = objective_and_grad(f)
            f_new = f - lr2 * grad
            f_new = np.maximum(f_new, 0.0)
            integral = np.sum(f_new) * dx
            if integral > 1e-10:
                f_new = f_new / integral
            f = f_new
            cur_score, _ = compute_score(f)
            if cur_score < best_score:
                best_score = cur_score
                best_f = f.copy()
    
    return list(best_f)
# EVOLVE-BLOCK-END
