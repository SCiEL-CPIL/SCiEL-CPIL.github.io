import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    time_limit = 140.0  # 10s margin
    
    # Initialize with ones, normalized to sum 1
    y = np.ones(N) / N
    
    # Learning rate
    lr = 0.01
    # Softmax temperature
    beta = 100.0
    
    def autocorr(y):
        # Linear autocorrelation
        # Using FFT for speed
        Y = np.fft.fft(y, n=2*N)
        C = np.fft.ifft(Y * np.conj(Y))[:N]
        return C.real

    def objective_and_grad(y):
        C = autocorr(y)
        abs_C = np.abs(C)
        # Softmax approximation of max
        # log-sum-exp trick for stability
        m = np.max(abs_C)
        soft_max_val = m + np.log(np.sum(np.exp(beta * (abs_C - m)))) / beta
        # Gradient of soft_max
        # d/dx log(sum(exp(beta*(x-m)))) = sum(beta*exp(...)*sign(x)) / sum(exp(...))
        # Note: m is constant w.r.t y for gradient calculation of the term inside?
        # Actually, derivative of log(sum(exp(beta*(|C|-m)))) w.r.t C_k is 
        # beta * sign(C_k) * exp(beta*(|C_k|-m)) / sum(exp(beta*(|C_j|-m)))
        exp_terms = np.exp(beta * (abs_C - m))
        weights = exp_terms / np.sum(exp_terms)
        # Gradient of soft_max w.r.t C
        grad_C = weights * beta * np.sign(C)
        
        # Gradient of C w.r.t y
        # C_k = sum y_i y_{i+k}
        # grad_y = sum_k grad_C_k * (shift(y, -k) + shift(y, k))
        # This is equivalent to convolving grad_C with y?
        # Let's use FFT for gradient of quadratic form.
        # C = ifft(|fft(y)|^2)
        # grad_y = 2 * ifft(fft(grad_C) * conj(fft(y)))? 
        # Let's derive:
        # C_k = sum_i y_i y_{i+k}
        # dC_k/dy_j = y_{j+k} + y_{j-k}
        # sum_k grad_C_k (y_{j+k} + y_{j-k})
        # = sum_k grad_C_k y_{j+k} + sum_k grad_C_k y_{j-k}
        # Let l = j+k => k = l-j. sum_l grad_C_{l-j} y_l
        # This is a convolution.
        # grad_y = convolve(grad_C, y) + convolve(grad_C[::-1], y)?
        # Actually, since C is real and even (autocorr), grad_C is even?
        # If C is even, |C| is even, weights are even, sign(C) is even. So grad_C is even.
        # Then grad_y = 2 * convolve(grad_C, y) ?
        # Let's just compute it directly or via FFT.
        # FFT method:
        # C = ifft(|Y|^2)
        # dC/dy = 2 * Re(ifft(Y * dC/dy? No.
        # Let's use the explicit convolution via FFT.
        # grad_y_j = sum_k grad_C_k (y_{j+k} + y_{j-k})
        # This is the convolution of grad_C with y, plus the convolution of grad_C with y reversed?
        # Since grad_C is symmetric (even), conv(grad_C, y) covers both terms if we handle indices correctly.
        # Let's use np.convolve with mode='same' or FFT.
        
        # FFT approach for gradient of sum_k g_k C_k
        # C = ifft(Y * conj(Y))
        # g = grad_C
        # Objective L = sum g_k C_k
        # dL/dy = 2 * Re(ifft(fft(g) * conj(Y))) ? 
        # Let's check.
        # C_k = sum_i y_i y_{i+k}
        # L = sum_k g_k sum_i y_i y_{i+k}
        # = sum_i y_i (sum_k g_k y_{i+k})
        # = sum_i y_i (g * y)_i  (convolution)
        # So L = y . (g * y)
        # dL/dy = 2 (g * y) if g is symmetric?
        # (g * y)_i = sum_k g_k y_{i+k}
        # dL/dy_i = (g*y)_i + sum_k g_k y_{k-i} ?
        # Since g is even, g_k = g_{-k}.
        # sum_k g_k y_{k-i} = sum_l g_{-l} y_{i-l} = sum_l g_l y_{i-l} = (g * y)_i.
        # So grad_y = 2 * convolve(g, y).
        
        G = np.fft.fft(grad_C, n=2*N)
        Y = np.fft.fft(y, n=2*N)
        grad_y = np.fft.ifft(G * Y)[:N].real * 2.0
        
        return soft_max_val, grad_y

    best_val = np.inf
    best_y = y.copy()
    
    # Calculate initial value
    C_init = autocorr(best_y)
    # The metric is max |C_k| / (sum y)^2. Since sum y = 1, it's max |C_k|.
    # But we must return the sequence x such that the ratio is minimized.
    # The score is seed/value. Seed value is 1/N.
    # If we minimize max |C_k| for sum y = 1, the value is max |C_k|.
    # We want max |C_k| < 1/N.
    
    for step in range(500):
        if time.time() - start_time > time_limit:
            break
            
        val, grad = objective_and_grad(y)
        
        # Project gradient to satisfy sum y = 1
        # grad_proj = grad - mean(grad)
        grad_proj = grad - np.mean(grad)
        
        y_new = y - lr * grad_proj
        
        # Project y_new to sum 1
        y_new = y_new / np.sum(y_new)
        
        # Optional: Project to non-negative? 
        # If we allow negative, we might get better results.
        # But "signed step heights" usually implies we can choose signs.
        # However, if we don't constrain sign, the solution might have large oscillations.
        # Let's try without non-negativity constraint first.
        
        y = y_new
        
        # Evaluate actual max
        C = autocorr(y)
        actual_max = np.max(np.abs(C))
        if actual_max < best_val:
            best_val = actual_max
            best_y = y.copy()
            
    # Construct final sequence
    # The problem asks for "sequence of signed step heights".
    # The score is based on the ratio.
    # We can scale best_y to any scale.
    # Let's return best_y scaled to have integer values or just the floats?
    # "construct() -> sequence of signed step heights"
    # Usually implies the values returned are the heights.
    # If we return best_y, sum is 1.
    # The evaluator calculates the ratio.
    # Ratio = max |autoconv(best_y)| / (sum best_y)^2 = best_val / 1 = best_val.
    # Seed value = 1/N = 0.0025.
    # If best_val < 0.0025, score > 1.
    
    return best_y
