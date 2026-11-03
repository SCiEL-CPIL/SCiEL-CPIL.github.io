"""Program for minimizing the normalized maximum of autoconvolution of a step function."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Minimizes the normalized maximum of the autoconvolution of a nonnegative step function.
    Uses Gradient Descent with a Log-Sum-Exp approximation to the max function for smoothness.
    """
    
    # Objective function and gradient
    # h is normalized to sum to 1.
    # We minimize max(autoconv(h)).
    # Using LogSumExp(beta, C) where C = autoconv(h).
    
    def logsumexp(beta, C):
        # Numerical stability: subtract max
        max_C = np.max(C)
        if max_C == 0:
            return 0.0
        # Avoid overflow
        scaled = beta * (C - max_C)
        # Clip to prevent exp overflow, though beta*(C-max) <= 0
        # If beta is large, exp(0)=1, exp(-large)~0
        vals = np.exp(scaled)
        return max_C + np.log(np.sum(vals)) / beta

    def objective_and_grad(h_flat, beta):
        # h_flat is 1D array of length N
        h = h_flat.copy()
        s = np.sum(h)
        if s < 1e-10:
            h = np.ones(N) / N
            s = 1.0
        else:
            h = h / s
            
        # Compute autoconvolution
        H = np.fft.fft(h)
        C = np.fft.ifft(H * H).real
        
        # Compute LogSumExp objective
        obj = logsumexp(beta, C)
        
        # Compute Gradient
        # d/dh_i (1/beta * log(sum_k exp(beta C_k)))
        # = (1/beta) * sum_k (beta exp(beta C_k) / sum_j exp(beta C_j)) * dC_k/dh_i
        # = sum_k w_k * h_{k-i}
        # where w_k = exp(beta C_k) / sum_j exp(beta C_j)
        
        max_C = np.max(C)
        scaled_w = beta * (C - max_C)
        # Clip scaled_w to avoid underflow issues in sum, though exp(-large) is fine
        w_unnorm = np.exp(scaled_w)
        w = w_unnorm / np.sum(w_unnorm)
        
        # Gradient g_i = sum_k w_k h_{k-i}
        # This is the circular convolution of w and h_reversed?
        # Let's check indices.
        # g_i = \sum_{k=0}^{N-1} w_k h_{(k-i) \bmod N}
        # Let j = k-i => k = i+j.
        # g_i = \sum_j w_{i+j} h_j.
        # This is the cross-correlation of w and h.
        # In FFT:
        # FFT(w * h_rev) = FFT(w) * FFT(h_rev)
        # h_rev[i] = h[-i] = h[N-i] (for i!=0).
        # Since h is real, FFT(h_rev) = conj(FFT(h)).
        # So G_fft = FFT(w) * conj(FFT(h)).
        # g = IFFT(G_fft).real
        
        W = np.fft.fft(w)
        H = np.fft.fft(h)
        G = W * np.conj(H)
        grad = np.fft.ifft(G).real
        
        return obj, grad

    # Optimization Strategy
    # 1. Start with a Gaussian initialization
    # 2. Use L-BFGS-B or manual gradient descent?
    # L-BFGS-B requires a smooth objective. LogSumExp is smooth.
    # However, L-BFGS-B might be slow for 600 vars if not careful.
    # Manual gradient descent with adaptive step is robust.
    
    def optimize(h0, iterations=500, beta_start=50.0, beta_end=500.0):
        h = h0.copy()
        s = np.sum(h)
        if s > 0:
            h = h / s
        else:
            h = np.ones(N) / N
            
        best_h = h.copy()
        best_score = np.inf
        
        # We will increase beta over time to sharpen the max approximation
        for it in range(iterations):
            # Interpolate beta
            t = it / max(iterations - 1, 1)
            beta = beta_start * (beta_end / beta_start) ** t
            
            obj, grad = objective_and_grad(h, beta)
            
            # Normalize gradient? The gradient scale depends on h (sum 1) and beta.
            # Step size should be small.
            # Use a simple adaptive step: step = 1.0 / (np.max(np.abs(grad)) + 1e-6) * 0.1
            # Or fixed small step with projection.
            
            # Let's use a fixed step size scaled by the inverse of the gradient norm for stability
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                break
                
            step_size = 0.1 / grad_norm
            
            h_new = h - step_size * grad
            
            # Project to non-negative
            h_new = np.clip(h_new, 0, None)
            
            # Normalize to sum 1
            s_new = np.sum(h_new)
            if s_new < 1e-10:
                h_new = h # Fallback
            else:
                h_new = h_new / s_new
                
            # Check improvement on the actual max (not logsumexp) to ensure we are minimizing the right thing
            H_new = np.fft.fft(h_new)
            C_new = np.fft.ifft(H_new * H_new).real
            current_max = np.max(C_new)
            
            if current_max < best_score:
                best_score = current_max
                best_h = h_new.copy()
                
            h = h_new
            
        return best_h, best_score

    # Initializations
    # 1. Gaussian
    x = np.linspace(-1, 1, N)
    sigma = 0.2
    gauss = np.exp(-x**2 / (2 * sigma**2))
    
    # 2. Cosine bump (smooth, zero at edges)
    cos_bump = np.maximum(0, np.cos(np.pi * x)) ** 2
    
    # 3. Uniform (baseline)
    uniform = np.ones(N)
    
    candidates = [gauss, cos_bump, uniform]
    
    best_h_global = None
    best_score_global = np.inf
    
    for h0 in candidates:
        h_opt, score = optimize(h0, iterations=300)
        if score < best_score_global:
            best_score_global = score
            best_h_global = h_opt
            
    return list(best_h_global)
# EVOLVE-BLOCK-END
