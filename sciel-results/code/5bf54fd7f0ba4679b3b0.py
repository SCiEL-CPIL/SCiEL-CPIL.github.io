import numpy as np
import time
import scipy.optimize as opt

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(f*f) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        # Autoconvolution via FFT
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def smooth_objective(x, alpha):
        """
        Objective function for optimization.
        x: log-heights
        alpha: sharpness parameter for max approximation
        Returns: (value, gradient)
        """
        h = softmax(x)
        # Compute auto-correlation
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        power = np.abs(H) ** 2
        acorr = np.fft.irfft(power, n=size)
        
        # Smooth max: (1/alpha) * log(sum(exp(alpha * acorr)))
        # To avoid overflow, shift by max
        max_acorr = np.max(acorr)
        log_sum_exp = max_acorr + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (acorr - max_acorr))))
        
        value = log_sum_exp
        # Note: We are minimizing the peak. The normalization (sum h)^2 is 1 because h is softmax.
        # So we just minimize the peak.
        
        # Gradient calculation
        # d(value)/d(acorr_k) = exp(alpha * (acorr_k - max_acorr)) / sum(exp(...))
        # Let w_k = exp(alpha * (acorr_k - max_acorr)) / sum(exp(...))
        # sum(w_k) = 1
        w = np.exp(alpha * (acorr - max_acorr))
        w /= np.sum(w)
        
        # d(acorr_k)/d(h_i)
        # acorr_k = sum_i h_i h_{i+k}
        # grad_h = 2 * (h * h) ? No, gradient of auto-correlation is the convolution with h, but symmetric.
        # Actually, acorr is the inverse FFT of |H|^2.
        # d(acorr)/d(h) = 2 * (h * h) ?
        # Let's use the property that grad(acorr_k) wrt h_i is h_{i+k} + h_{i-k}.
        # This is equivalent to 2 * (h * h) if we consider the full convolution structure, 
        # but we need to be careful with the FFT domain.
        
        # Simpler: Use the chain rule via FFT?
        # d(LSE)/d(h) = 2 * (w * h) ?
        # Let's verify.
        # acorr = IFFT(|FFT(h)|^2).
        # d(acorr_k)/d(h_i) = h_{(i+k) % size} + h_{(i-k) % size}.
        # This is 2 * (circular_conv(h, h))_i ?
        # Yes, the gradient of the auto-correlation vector with respect to h is 2 * (h * h) (circular).
        
        grad_acorr_h = 2.0 * np.fft.irfft(np.fft.rfft(h, n=size) * np.fft.rfft(w, n=size), n=size)[:N]
        # Wait, w is defined on the acorr domain (time domain).
        # The gradient of sum(w_k * acorr_k) wrt h is 2 * (w * h)?
        # Let's check:
        # sum_k w_k acorr_k = sum_k w_k sum_i h_i h_{i+k}
        # = sum_i h_i sum_k w_k h_{i+k}
        # = sum_i h_i (w * h)_i  (circular conv)
        # Gradient wrt h_i is (w * h)_i + (w * h)_{-i}?
        # Since w and h are real, and acorr is even, the gradient is 2 * (w * h) if w is even?
        # w is even because acorr is even.
        # So grad_h = 2 * (w * h).
        
        # Compute w * h via FFT
        W = np.fft.rfft(w, n=size)
        H_h = np.fft.rfft(h, n=size)
        conv_wh = np.fft.irfft(W * H_h, n=size)[:N]
        
        grad_h = 2.0 * conv_wh
        
        # Gradient wrt x: h = softmax(x)
        # d(h_i)/d(x_j) = h_i (delta_ij - h_j)
        # grad_x_j = sum_i grad_h_i h_i (delta_ij - h_j)
        # = h_j (sum_i grad_h_i h_i) - h_j (sum_i grad_h_i h_i) ??
        # Let S = sum_i grad_h_i h_i
        # grad_x_j = h_j S - h_j S = 0?
        # No.
        # sum_i grad_h_i h_i (delta_ij - h_j)
        # = h_j grad_h_j - h_j sum_i grad_h_i h_i
        
        S = np.dot(grad_h, h)
        grad_x = h * (grad_h - S)
        
        return value, grad_x

    def optimize_single(alpha_seq, init_x):
        best_h = None
        best_val = float('inf')
        
        x = init_x.copy()
        
        for alpha in alpha_seq:
            if time.time() > deadline:
                break
            
            # Optimize with current alpha
            res = opt.minimize(
                smooth_objective, 
                x, 
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            x = res.x
            h = softmax(x)
            val = compute_metric(h)
            
            if val < best_val:
                best_val = val
                best_h = h.copy()
                
        return best_h, best_val

    # Multiple initializations
    np.random.seed(42)
    inits = [
        np.zeros(N), # Uniform
        np.random.randn(N),
        np.random.rand(N),
        np.sin(np.linspace(0, 2*np.pi, N)),
        np.exp(-np.linspace(0, 1, N)**2)
    ]
    
    best_h = None
    best_m = float('inf')
    
    alphas = [10, 50, 200, 1000]
    
    for init in inits:
        if time.time() > deadline:
            break
        h_opt, m_opt = optimize_single(alphas, init)
        if h_opt is not None and m_opt < best_m:
            best_m = m_opt
            best_h = h_opt
    
    # Fallback to uniform if optimization fails or is worse
    uniform_h = np.ones(N) / N
    uniform_m = compute_metric(uniform_h)
    
    if best_h is None or uniform_m < best_m:
        return list(uniform_h)
        
    return list(best_h)

# EVOLVE-BLOCK-END
