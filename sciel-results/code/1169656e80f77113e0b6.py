import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 120  # seconds

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def autoconv_max(h):
        """Compute max of linear autoconvolution of h (assumed sum=1)."""
        # Use FFT for linear convolution
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)
    
    def smooth_max(h, beta=100.0):
        """Smooth approximation of max using log-sum-exp."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        # Log-sum-exp for numerical stability
        m = np.max(conv)
        if m < -1e10:
            return 0.0
        s = np.sum(np.exp((conv - m) * beta))
        return m + np.log(s) / beta
    
    def objective(params, beta):
        """Objective function for scipy optimizer."""
        h = np.exp(params)  # Ensure positivity via exp
        h = h / np.sum(h)   # Normalize to sum=1
        return smooth_max(h, beta)
    
    def gradient(params, beta):
        """Gradient of the smoothed objective."""
        h = np.exp(params)
        s = np.sum(h)
        h_norm = h / s
        
        # Compute autoconv
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h_norm
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        
        m = np.max(conv)
        exp_conv = np.exp((conv - m) * beta)
        weights = exp_conv / np.sum(exp_conv)
        
        # Gradient of max w.r.t conv
        # d(conv)/d(h_norm[i]) 
        # conv[k] = sum_j h_norm[j] * h_norm[k-j] (linear)
        # d(conv[k])/d(h_norm[i]) = h_norm[k-i] + h_norm[i-k] (if in range)
        
        grad_conv = np.zeros(N)
        idx_max_region = np.where(weights > 1e-10)[0]
        
        # This is expensive to do exactly, use FFT-based gradient
        # The gradient of sum_k w_k * conv[k] w.r.t h is:
        # (w * h) + (w_rev * h) where w is the weight vector
        
        w_full = np.zeros(size)
        w_full[:len(conv)] = weights
        
        # Gradient w.r.t h_pad
        # conv = ifft(fft(h_pad)^2)
        # d(conv)/d(h_pad) = 2 * ifft(fft(h_pad) * fft(w_full))
        # Actually, for real signals: d(conv)/d(h_pad[i]) = (w_full * h_pad) + (w_full_rev * h_pad)
        # where * is circular convolution
        
        # Simpler: use the fact that grad = 2 * ifft(fft(h_pad) * fft(w_full))
        W = np.fft.fft(w_full, n=size)
        grad_h_pad = 2.0 * np.fft.ifft(H * W).real[:N]
        
        # Chain rule: params -> h = exp(params) -> h_norm = h/sum(h)
        # d(h_norm[i])/d(params[j]) = h_norm[i] * (delta_ij - h_norm[j])
        
        grad_params = np.zeros(N)
        for i in range(N):
            # grad_params[i] = sum_j grad_conv[j] * d(h_norm[j])/d(params[i])
            # = grad_conv[i] * h_norm[i] * (1 - sum_j grad_conv[j] * h_norm[j])
            # Wait, that's not right. Let me think again.
            
            # h_norm[j] = exp(params[j]) / sum_k exp(params[k])
            # d(h_norm[j])/d(params[i]) = h_norm[j] * (delta_{ij} - h_norm[i])
            
            # So grad_params[i] = sum_j grad_conv[j] * h_norm[j] * (delta_{ij} - h_norm[i])
            # = grad_conv[i] * h_norm[i] - h_norm[i] * sum_j grad_conv[j] * h_norm[j]
            pass
        
        # Vectorized:
        dot = np.dot(grad_h_pad, h_norm)
        grad_params = grad_h_pad * h_norm - h_norm * dot
        
        # Also account for the normalization in objective? No, we normalized h.
        # But we have the exp transformation, so:
        # params -> h_raw = exp(params) -> h_norm = h_raw / sum(h_raw)
        # The gradient above is d(objective)/d(h_norm)
        # We need d(objective)/d(params)
        
        # Let g = grad_h_pad (gradient w.r.t h_norm)
        # h_norm = exp(params) / sum(exp(params))
        # d(h_norm[j])/d(params[i]) = h_norm[j] * (delta_ij - h_norm[i])
        # grad_params[i] = sum_j g[j] * h_norm[j] * (delta_ij - h_norm[i])
        # = g[i] * h_norm[i] - h_norm[i] * sum_j g[j] * h_norm[j]
        
        dot_g = np.dot(grad_h_pad, h_norm)
        grad_params = grad_h_pad * h_norm - h_norm * dot_g
        
        return grad_params
    
    # Initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Raised cosine
    i = np.arange(N)
    rc = 0.5 * (1 + np.cos(2 * np.pi * (i - N/2) / N))
    rc = np.maximum(rc, 0.01)
    inits.append(rc)
    
    # Gaussian
    center = N / 2.0
    for sigma in [N/4, N/3, N/5]:
        gauss = np.exp(-((i - center) ** 2) / (2 * sigma ** 2))
        inits.append(gauss)
    
    # U-shaped (mass at edges)
    u_shape = 1.0 + 3.0 * (1 - np.sin(np.pi * i / N))
    inits.append(u_shape)
    
    # Triangular
    tri = 1 - np.abs(i - N/2) / (N/2)
    tri = np.maximum(tri, 0.01)
    inits.append(tri)
    
    best_h = None
    best_val = np.inf
    
    for init in inits:
        if time_left() < 15:
            break
        
        init = init / np.sum(init)
        
        # Try different beta values for smoothing
        betas = [50.0, 100.0, 200.0]
        
        for beta in betas:
            if time_left() < 10:
                break
            
            # Convert to log-space for scipy
            h_init = init.copy()
            # Ensure positive
            h_init = np.maximum(h_init, 1e-10)
            params_init = np.log(h_init)
            
            try:
                res = minimize(
                    objective,
                    params_init,
                    args=(beta,),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 50,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxfun': 100,
                        'maxls': 20,
                        'maxcor': 10,
                        'maxls': 10,
                        'maxfun': 50
                    }
                )
                
                if res.success or res.fun < best_val:
                    h_opt = np.exp(res.x)
                    h_opt = h_opt / np.sum(h_opt)
                    val = autoconv_max(h_opt)
                    
                    if val < best_val:
                        best_val = val
                        best_h = h_opt.copy()
                        
            except Exception as e:
                continue
    
    # Fallback: if optimization failed, use the best initial
    if best_h is None:
        for init in inits:
            h = init / np.sum(init)
            val = autoconv_max(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # Final refinement with exact max (not smoothed)
    if best_h is not None and time_left() > 10:
        # Do a few steps of projected gradient with exact max
        current_h = best_h.copy()
        current_val = best_val
        eta = 0.05
        
        for k in range(100):
            if time_left() < 5:
                break
            
            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = current_h
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real
            
            c_val = np.max(conv)
            idx_max = np.argmax(conv)
            
            # Gradient
            idx = np.arange(N)
            pos1 = idx_max - idx
            pos2 = idx - idx_max
            
            grad = np.zeros(N)
            mask1 = (pos1 >= 0) & (pos1 < N)
            if np.any(mask1):
                grad[mask1] += current_h[pos1[mask1]]
            mask2 = (pos2 >= 0) & (pos2 < N)
            if np.any(mask2):
                grad[mask2] += current_h[pos2[mask2]]
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            
            new_h = current_h - eta * grad
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s > 0:
                new_h /= s
            
            new_val = autoconv_max(new_h)
            
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                eta *= 0.8
                if eta < 1e-6:
                    break
    
    if best_h is None:
        best_h = np.ones(N) / N
    
    return best_h.tolist()
