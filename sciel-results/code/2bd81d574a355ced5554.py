"""Model-and-optimize bootstrap: parameterize step heights and optimize the objective."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Strategy:
    1. Represent h as real part of sparse Fourier series (first M harmonics).
    2. Optimize complex amplitudes using projected gradient on log-sum-exp surrogate.
    3. Final polish with coordinate descent on full space.
    """
    start_time = time.time()
    DEADLINE = 142  # 8s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    rng = np.random.RandomState(42)
    
    # Fourier series representation:
    # h[i] = c0 + sum_{k=1}^{M} (a_k * cos(2*pi*k*i/N) + b_k * sin(2*pi*k*i/N))
    # This is equivalent to real part of complex Fourier series
    
    M = 20  # Number of harmonics
    
    def fourier_to_h(params):
        """Convert Fourier parameters to step heights.
        params[0] = DC offset, params[1::2] = cos amplitudes, params[2::2] = sin amplitudes
        """
        h = np.full(N, params[0])
        idx = 1
        for k in range(1, M + 1):
            a = params[idx]
            b = params[idx + 1]
            if a != 0:
                h += a * np.cos(2.0 * np.pi * k * np.arange(N) / N)
            if b != 0:
                h += b * np.sin(2.0 * np.pi * k * np.arange(N) / N)
            idx += 2
        return h
    
    n_params = 1 + 2 * M
    
    def fourier_objective(params):
        h = fourier_to_h(params)
        val, valid = evaluate(h)
        if not valid:
            return 1e10
        return val
    
    # Phase 1: Random restarts with gradient-based optimization in Fourier space
    def lse_surrogate_and_grad(h, beta):
        """
        Log-sum-exp surrogate for max(conv(h,h)) with gradient.
        Returns (surrogate_value, gradient_wrt_h)
        """
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        shifted = beta * (conv - max_conv)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp < 1e-300:
            soft_max = max_conv
            weights = np.zeros_like(conv)
            weights[np.argmax(conv)] = 1.0
        else:
            logsumexp = max_conv + np.log(sum_exp) / beta
            soft_max = logsumexp
            weights = exp_shifted / sum_exp
        
        # Gradient of soft_max w.r.t. h:
        # conv[k] = sum_j h[j]*h[k-j]
        # d(conv[k])/d(h[i]) = sum_j [delta(j,i)*h[k-j] + h[j]*delta(k-j,i)]
        #                      = h[k-i] + h[k-i] (with appropriate bounds)
        # Actually: d(conv[k])/d(h[i]) = h[k-i] + h[k-i] when indices valid
        # More carefully: conv[k] = sum_j h[j] * h[k-j]
        # d(conv[k])/d(h[i]) = h[k-i] (from first factor when j=i) + h[k-i] (from second factor when k-j=i, i.e., j=k-i)
        # But we need to be careful about bounds. Using FFT is cleaner.
        
        # Use FFT for gradient computation
        n_fft = 1 << (2 * N - 1).bit_length()
        W = np.fft.rfft(weights, n=n_fft)
        H = np.fft.rfft(h, n=n_fft)
        # grad_soft_max[i] = sum_k weights[k] * d(conv[k])/d(h[i])
        # = sum_k weights[k] * (h[k-i] + h[k-i]) with proper handling
        # In frequency domain: the gradient is 2 * ifft(weights_fft * conj(h_fft)) truncated
        corr = np.fft.irfft(W * np.conj(H), n=n_fft)
        grad_soft_max = 2.0 * corr[:N]
        
        # f = 2*N*soft_max / s^2
        f = 2.0 * N * soft_max / (s * s)
        
        # Gradient of f
        grad_f = (2.0 * N / (s * s)) * grad_soft_max - (4.0 * N * soft_max / (s ** 3))
        
        return f, grad_f
    
    # Gradient in Fourier parameter space
    def fourier_grad(params, beta):
        h = fourier_to_h(params)
        f_h, grad_h = lse_surrogate_and_grad(h, beta)
        
        # grad_params[k] = sum_i grad_h[i] * d(h[i])/d(params[k])
        grad_params = np.zeros(n_params)
        grad_params[0] = np.sum(grad_h)  # DC term
        
        idx = 1
        for k in range(1, M + 1):
            cos_k = np.cos(2.0 * np.pi * k * np.arange(N) / N)
            sin_k = np.sin(2.0 * np.pi * k * np.arange(N) / N)
            grad_params[idx] = np.dot(grad_h, cos_k)
            grad_params[idx + 1] = np.dot(grad_h, sin_k)
            idx += 2
        
        return f_h, grad_params
    
    # Projected gradient descent in Fourier space
    def optimize_fourier(params_init, beta, lr, n_iter, momentum=0.9):
        params = params_init.copy()
        v = np.zeros_like(params)
        
        for it in range(n_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            f, grad = fourier_grad(params, beta)
            
            if not np.isfinite(f):
                break
            
            v = momentum * v + (1 - momentum) * grad
            params_new = params - lr * v
            
            # Simple bounds: keep params in [-10, 10]
            params_new = np.clip(params_new, -10.0, 10.0)
            
            h_new = fourier_to_h(params_new)
            val_new, valid_new = evaluate(h_new)
            
            if valid_new and np.isfinite(val_new):
                params = params_new
            else:
                # Revert and reduce learning rate
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        return params
    
    # Phase 1: Multiple random restarts in Fourier space
    n_restarts = 15
    best_fourier_params = None
    
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Random initialization
        if r == 0:
            # Start near constant (DC dominant)
            params_init = np.zeros(n_params)
            params_init[0] = 1.0
        else:
            params_init = rng.uniform(-2, 2, n_params)
            params_init[0] += 1.0  # Ensure non-zero DC
        
        # Adaptive beta schedule
        for beta in [50, 100, 300, 1000]:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            lr = 0.01
            n_iter = 300
            params_opt = optimize_fourier(params_init, beta, lr, n_iter)
            
            h_opt = fourier_to_h(params_opt)
            val_opt, valid_opt = evaluate(h_opt)
            
            if valid_opt and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                best_fourier_params = params_opt.copy()
            
            # Use optimized params as init for next beta
            params_init = params_opt.copy()
        
        if best_value < 1.2:
            break
    
    # Phase 2: Full-space gradient optimization from best Fourier result
    if time.time() - start_time < DEADLINE - 15:
        h_curr = best_h.copy()
        
        for beta in [100, 300, 1000, 3000]:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Gradient descent with adaptive step size
            lr = 0.005
            momentum = 0.9
            v = np.zeros_like(h_curr)
            
            for it in range(500):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                f, grad = lse_surrogate_and_grad(h_curr, beta)
                
                if not np.isfinite(f):
                    break
                
                v = momentum * v + (1 - momentum) * grad
                h_new = h_curr - lr * v
                
                val_new, valid_new = evaluate(h_new)
                
                if valid_new and np.isfinite(val_new) and val_new < best_value:
                    best_value = val_new
                    best_h = h_new.copy()
                    h_curr = h_new
                else:
                    lr *= 0.7
                    if lr < 1e-9:
                        break
    
    # Phase 3: Coordinate descent polish
    if time.time() - start_time < DEADLINE - 3:
        h_curr = best_h.copy()
        lr = 0.01
        stagnation = 0
        
        for iteration in range(100):
            if time.time() - start_time > DEADLINE - 3:
                break
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE - 3:
                    break
                for step_mult in [1.0, 0.5, 0.2, 0.1]:
                    h_try = h_curr.copy()
                    h_try[coord] += lr * step_mult * rng.uniform(-1, 1)
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_value:
                        best_value = val_try
                        best_h = h_try.copy()
                        h_curr = h_try
                        improved = True
                        break
                if best_value < 1.0:
                    break
            if best_value < 1.0:
                break
            if not improved:
                stagnation += 1
                lr *= 0.6
                if stagnation > 8:
                    break
            else:
                stagnation = 0
    
    return best_h
# EVOLVE-BLOCK-END
