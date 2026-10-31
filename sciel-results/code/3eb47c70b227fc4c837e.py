import numpy as np
import time
import random
from scipy.optimize import minimize

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def compute_autoconv(h):
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv

    def objective_and_grad(x, beta):
        h = np.exp(x)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10, np.zeros(N)
        h_norm = h / s
        
        conv = compute_autoconv(h_norm)
        m = np.max(conv)
        shifted = (conv - m) * beta
        shifted = np.clip(shifted, -50, 50)
        exp_conv = np.exp(shifted)
        sum_exp = np.sum(exp_conv)
        
        if sum_exp < 1e-30:
            val = m
            weights = np.zeros_like(conv)
            weights[np.argmax(conv)] = 1.0
        else:
            val = m + np.log(sum_exp) / beta
            weights = exp_conv / sum_exp
            
        size = 2 * N
        w_pad = np.zeros(size)
        w_pad[:N] = weights[:N] # Weights are symmetric, but conv is 2N. 
        # Actually, conv indices 0..N-1 are the positive lags. 
        # The autoconvolution is symmetric. 
        # The max is over all lags. 
        # We should use the full weights vector of size 2N? 
        # No, h is only N. 
        # The gradient formula grad_h[k] = 2 beta sum_j w_j h[k+j] assumes w is defined for all j. 
        # Let's just use the first N weights? 
        # If we use w_pad of size 2N with w_pad[:N] = weights[:N], 
        # the convolution will be correct for the first N elements of grad_h.
        
        w_fft = np.fft.fft(w_pad, n=size)
        h_fft = np.fft.fft(h_norm, n=size)
        grad_h_fft = w_fft * h_fft
        grad_h = np.fft.ifft(grad_h_fft).real[:N]
        grad_h = 2 * beta * grad_h
        
        # Gradient w.r.t x
        # h = exp(x)
        # dh/dx = h
        # But we have h_norm = h/s.
        # Approximation: grad_x = grad_h * h_norm * s? No.
        # Let's just return grad_h * h (unnormalized) and let the optimizer handle it?
        # No, the objective is defined on h_norm.
        # If we use L-BFGS-B on x, it expects grad of f(x).
        # f(x) = F(h_norm(x)).
        # df/dx_i = sum_j (df/dh_norm_j) * (dh_norm_j/dx_i).
        # dh_norm_j/dx_i = (h_j/s) * (delta_{ij} - h_i/s) = h_norm_j * (delta_{ij} - h_norm_i).
        # So grad_x_i = sum_j grad_h_norm_j * h_norm_j * (delta_{ij} - h_norm_i)
        # = h_norm_i * grad_h_norm_i - h_norm_i * sum_j (grad_h_norm_j * h_norm_j).
        
        # We have grad_h which is grad w.r.t h_norm? 
        # My grad_h calculation above was for grad w.r.t h_norm? 
        # No, I calculated grad of Conv(h_norm) w.r.t h_norm.
        # So grad_h_norm = grad_h.
        
        dot = np.dot(grad_h, h_norm)
        grad_x = h_norm * (grad_h - dot * h_norm)
        
        return val, grad_x

    # Inits
    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    
    inits = []
    inits.append(np.ones(N))
    
    for sigma in [N/10, N/8, N/6]:
        for alpha in [0.1, 0.2, 0.3]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2)
            
    u_shape = 1.0 + 2.0 * ((i - center) / (N/2))**2
    inits.append(u_shape)
    
    best_h = None
    best_val = np.inf
    
    for init in inits:
        if time_left() < 5: break
        h = init / np.sum(init)
        val = compute_autoconv(h).max()
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv(best_h).max()
        
    # Optimize in log-space
    if time_left() > 10:
        current_x = np.log(best_h + 1e-10)
        
        betas = [5.0, 10.0, 20.0, 50.0]
        
        for beta in betas:
            if time_left() < 5: break
            
            # Define wrapper for minimize
            def obj_wrapper(x):
                val, _ = objective_and_grad(x, beta)
                return val
            def grad_wrapper(x):
                _, grad = objective_and_grad(x, beta)
                return grad
                
            try:
                res = minimize(
                    obj_wrapper, 
                    current_x, 
                    jac=grad_wrapper, 
                    method='L-BFGS-B', 
                    options={'maxiter': 50, 'gtol': 1e-5}
                )
                if res.success or res.fun < best_val:
                    new_h = np.exp(res.x)
                    new_h /= np.sum(new_h)
                    new_val = compute_autoconv(new_h).max()
                    if new_val < best_val:
                        best_val = new_val
                        best_h = new_h
                        current_x = res.x
            except Exception:
                pass
                
    # Final CD
    if time_left() > 5:
        current_h = best_h.copy()
        current_val = best_val
        for _ in range(5):
            if time_left() < 2: break
            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 1: break
                for delta in [0.05, -0.05]:
                    new_h = current_h.copy()
                    new_h[idx] = max(0, current_h[idx] + delta * current_h[idx])
                    s = np.sum(new_h)
                    if s < 1e-10: continue
                    new_h /= s
                    new_val = compute_autoconv(new_h).max()
                    if new_val < current_val - 1e-12:
                        current_h = new_h
                        current_val = new_val
                        improved = True
                        break
            if not improved: break
            
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
            
    return best_h.tolist()
