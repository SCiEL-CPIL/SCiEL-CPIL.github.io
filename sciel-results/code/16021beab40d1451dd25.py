import numpy as np
from scipy.optimize import minimize
import time

def compute_score(f):
    """
    Compute the score for the given step function f (array of nonnegative values).
    The score is the normalized maximum of the circular autoconvolution.
    score = max_k c_k / mean_k c_k = n * max(c_k) / (sum f)^2
    """
    n = len(f)
    if n == 0 or np.sum(f) == 0:
        return 0.0
    # Circular autoconvolution via FFT
    F = np.fft.fft(f)
    P = np.abs(F) ** 2
    c = np.fft.ifft(P).real
    # Numerical cleanup
    c = np.clip(c, 0, None)
    max_c = np.max(c)
    mean_c = np.mean(c)  # equals (sum f)^2 / n
    if mean_c == 0:
        return 0.0
    return n * max_c / (np.sum(f) ** 2)

def fft_autoconv(f):
    """Compute circular autoconvolution via FFT."""
    n = len(f)
    F = np.fft.fft(f)
    P = np.abs(F) ** 2
    c = np.fft.ifft(P).real
    return np.clip(c, 0, None)

def smooth_max_objective(x, beta=100.0, n=600):
    """
    Objective function for optimization.
    x is the parameter vector for softmax.
    f = softmax(x).
    Returns smooth max of autoconvolution.
    """
    # Softmax
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    f = exp_x / np.sum(exp_x)
    
    # Autoconvolution
    c = fft_autoconv(f)
    
    # Smooth max: (1/beta) * log(sum(exp(beta * c)))
    # We want to minimize this.
    # Note: c is already scaled such that sum(c) = (sum(f))^2 = 1.
    # So mean(c) = 1/n.
    # The score is n * max(c).
    # Smooth score approx: (n/beta) * log(sum(exp(beta * c)))
    
    # To avoid overflow, shift c
    c_max = np.max(c)
    smooth_val = (1.0 / beta) * np.log(np.sum(np.exp(beta * (c - c_max)))) + c_max
    return n * smooth_val

def smooth_max_gradient(x, beta=100.0, n=600):
    """
    Gradient of the smooth max objective with respect to x.
    """
    # Softmax
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    f = exp_x / np.sum(exp_x)
    
    # Autoconvolution
    c = fft_autoconv(f)
    
    # Weights for smooth max
    c_max = np.max(c)
    weights = np.exp(beta * (c - c_max))
    weights /= np.sum(weights)
    
    # Gradient of J = sum(weights * c) w.r.t f
    # dJ/df_j = sum_k weights_k * (2 * f[(k-j) % n])
    # This is 2 * circular_conv(weights, f)
    W = np.fft.fft(weights)
    F = np.fft.fft(f)
    G = 2.0 * np.real(np.fft.ifft(W * F))
    
    # Gradient w.r.t x
    # dx_j = f_j * (G_j - sum(G_i * f_i))
    dot_Gf = np.dot(G, f)
    grad_x = f * (G - dot_Gf)
    
    return grad_x

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: constant
    f_init = np.ones(n) / n
    # Convert to x for softmax
    # f = exp(x) / sum(exp(x))
    # If f is constant, x is constant.
    x_init = np.zeros(n)
    
    best_score = compute_score(f_init)
    best_f = f_init.copy()
    
    # Multi-start
    np.random.seed(42)
    n_starts = 5
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
            
        if i == 0:
            x_cur = x_init.copy()
        elif i == 1:
            # Bump in middle -> higher x in middle
            x_cur = np.zeros(n)
            x_cur[n//4:3*n//4] = 1.0
        elif i == 2:
            # Two bumps
            x_cur = np.zeros(n)
            x_cur[n//6:n//3] = 1.0
            x_cur[2*n//3:5*n//6] = 1.0
        elif i == 3:
            # Random
            x_cur = np.random.normal(0, 0.5, n)
        else:
            # Smoothed random
            x_cur = np.random.normal(0, 0.5, n)
            # Simple smoothing
            x_cur = np.convolve(x_cur, np.ones(10)/10, mode='same')
        
        # Optimize
        try:
            res = minimize(
                smooth_max_objective,
                x_cur,
                method='L-BFGS-B',
                jac=smooth_max_gradient,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Extract best f
            x_opt = res.x
            x_shifted = x_opt - np.max(x_opt)
            f_opt = np.exp(x_shifted)
            f_opt /= np.sum(f_opt)
            
            score_opt = compute_score(f_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_f = f_opt.copy()
                
        except Exception as e:
            pass
    
    # Final polish
    if time.time() < deadline - 10:
        # Convert best_f back to x
        # f = exp(x) / sum(exp(x))
        # x = log(f) + constant
        # We can just use log(f) as initial x, adding a small epsilon to avoid log(0)
        f_safe = np.maximum(best_f, 1e-10)
        x_polish = np.log(f_safe)
        
        try:
            res = minimize(
                smooth_max_objective,
                x_polish,
                method='L-BFGS-B',
                jac=smooth_max_gradient,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x_opt = res.x
            x_shifted = x_opt - np.max(x_opt)
            f_opt = np.exp(x_shifted)
            f_opt /= np.sum(f_opt)
            
            score_opt = compute_score(f_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_f = f_opt.copy()
        except Exception as e:
            pass

    # Ensure non-negativity and normalize
    best_f = np.maximum(best_f, 0.0)
    s = best_f.sum()
    if s > 0:
        best_f = best_f / s
        
    return list(best_f)

if __name__ == "__main__":
    # Quick test
    f = construct()
    print(f"Score: {compute_score(np.array(f))}")
