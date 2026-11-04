import numpy as np
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Initialize with random signs
    x = np.random.choice([-1, 1], size=N).astype(float)
    
    # Normalize to unit energy
    x = x / np.sqrt(np.sum(x**2))
    
    best_x = x.copy()
    best_val = np.inf
    
    # Soft-max parameter
    beta = 10.0
    
    # Gradient Descent
    lr = 0.01
    for i in range(500):
        if time.time() > deadline:
            break
            
        # Compute autoconvolution
        c = np.fft.ifft(np.fft.fft(x, n=2*N) ** 2).real[:N]
        
        # Exclude lag 0
        c_sidelobes = np.concatenate([c[1:], c[-N+1:]]) # Circular? No, linear autoconv
        # Linear autoconvolution sidelobes:
        # c[k] for k=1..N-1 and k=-(N-1)..-1
        # Using FFT gives circular. We need linear.
        # Let's use np.convolve for accuracy if N is small, but N=400 is fine for FFT if zero-padded.
        # Zero padding to 2*N-1
        c_lin = np.fft.ifft(np.fft.fft(x, n=2*N-1) ** 2).real[:2*N-1]
        
        # Sidelobes are indices 1..N-1 and N+1..2N-2 (symmetric)
        # Actually, c_lin[k] corresponds to lag k.
        # Lags: 0, 1, ..., N-1, -(N-1), ..., -1
        # In the array of length 2N-1:
        # Index 0: lag 0
        # Index 1: lag 1
        # ...
        # Index N-1: lag N-1
        # Index N: lag -(N-1) ?? No.
        # FFT autoconv of length 2N-1:
        # c[k] = sum x[n] x[n-k]
        # Indices 0 to 2N-2.
        # Lag 0 is at index N-1? No.
        # Standard correlation:
        # Let's just use np.correlate or manual.
        
        # Simpler:
        c_full = np.correlate(x, x, mode='full')
        # c_full has length 2N-1.
        # Center is lag 0.
        # Sidelobes are everything except the center.
        center_idx = N - 1
        sidelobes = np.concatenate([c_full[:center_idx], c_full[center_idx+1:]])
        
        abs_sidelobes = np.abs(sidelobes)
        
        # Soft max
        # L = log(sum(exp(beta * abs_sidelobes))) / beta
        # To avoid overflow, subtract max
        m = np.max(abs_sidelobes)
        if m == 0:
            val = 0
            grad = 0
        else:
            log_sum_exp = np.log(np.sum(np.exp(beta * (abs_sidelobes - m)))) + beta * m
            val = log_sum_exp / beta
            
            # Gradient of soft max
            # d/dx (log sum exp(beta * |c_k|) / beta)
            # = (1/beta) * (sum exp(beta|c_k|) * beta * sign(c_k) * dc_k/dx) / (sum exp(beta|c_k|))
            # = sum( w_k * sign(c_k) * dc_k/dx )
            # where w_k = exp(beta(|c_k|-m)) / sum(exp(...))
            
            weights = np.exp(beta * (abs_sidelobes - m))
            weights /= np.sum(weights)
            
            # Gradient of c_k wrt x
            # c_k = sum x[n] x[n-k]
            # dc_k/dx[n] = x[n-k] + x[n+k] (for linear correlation)
            # This is equivalent to convolution with the sequence of weights.
            
            # Construct gradient
            grad = np.zeros(N)
            
            # This is expensive to do loop by loop.
            # Use FFT for gradient of correlation.
            # c = ifft(fft(x) * fft(x))
            # dc/dx = ifft(fft(grad_c) * fft(x)) + ifft(fft(x) * fft(grad_c)) ?
            # Actually, gradient of soft-max of abs(c) is:
            # sum_k w_k sign(c_k) (dc_k/dx)
            
            # Let's build the vector g_c such that g_c[k] = w_k sign(c_k) for sidelobes, 0 for center.
            g_c = np.zeros(2*N-1)
            g_c[:center_idx] = weights[:center_idx] * np.sign(c_full[:center_idx])
            g_c[center_idx+1:] = weights[center_idx+1:] * np.sign(c_full[center_idx+1:])
            
            # grad_x = ifft(fft(g_c) * fft(x)) + ifft(fft(x) * fft(g_c)) ?
            # No, correlation c_k = <x, shift_k(x)>.
            # grad = shift_{-k}(g_c) + shift_k(g_c)?
            
            # Efficient way:
            # grad = np.fft.ifft(np.fft.fft(g_c, n=2*N-1) * np.fft.fft(x, n=2*N-1)).real[:N]
            # This gives one part. The other part is symmetric.
            # Actually, for real x, c is symmetric.
            # Let's just use the property:
            # grad = np.convolve(g_c, x, mode='same')? No.
            
            # Let's stick to a simpler, albeit slower, gradient if N is small, or use the FFT trick correctly.
            # The gradient of the sum of weighted correlations is:
            # grad = ifft( fft(g_c) * fft(x) ) + ifft( fft(x) * fft(g_c) ) -- wait, g_c is in lag domain.
            
            # Let's use the following:
            # grad_x = np.fft.ifft(np.fft.fft(g_c, n=2*N) * np.fft.fft(x, n=2*N)).real[:N]
            # + np.fft.ifft(np.fft.fft(g_c[::-1], n=2*N) * np.fft.fft(x, n=2*N)).real[:N]
            # This is getting complicated.
            
            # Alternative: Use the fact that we are minimizing a smooth function.
            # Let's just use finite differences for the gradient? No, too slow.
            
            # Let's use the analytical gradient for the soft max of the autoconvolution.
            # L = (1/beta) log sum exp(beta |c_k|)
            # dL/dx = (1/beta) * (1/S) * sum_k exp(beta |c_k|) * beta * sign(c_k) * dc_k/dx
            # = sum_k p_k sign(c_k) dc_k/dx
            # where p_k = exp(beta |c_k|) / S.
            
            # dc_k/dx = shift_k(x) + shift_{-k}(x) (with appropriate boundary handling)
            
            # We can compute this by convolving the weighted sequence.
            # Let h_k = p_k sign(c_k).
            # Then grad = h * x + h_rev * x (convolution)
            
            h = np.zeros(2*N-1)
            h[:center_idx] = weights[:center_idx] * np.sign(c_full[:center_idx])
            h[center_idx+1:] = weights[center_idx+1:] * np.sign(c_full[center_idx+1:])
            
            # grad = np.convolve(h, x, mode='full') + np.convolve(h[::-1], x, mode='full')
            # Need to align indices.
            
            # Simpler:
            # grad = np.fft.ifft(np.fft.fft(h, n=2*N) * np.fft.fft(x, n=2*N)).real[:N]
            # This corresponds to one cross-correlation.
            # The total gradient is the sum of correlations with h and h_reversed.
            
            grad_part1 = np.fft.ifft(np.fft.fft(h, n=2*N) * np.fft.fft(x, n=2*N)).real[:N]
            grad_part2 = np.fft.ifft(np.fft.fft(h[::-1], n=2*N) * np.fft.fft(x, n=2*N)).real[:N]
            grad = grad_part1 + grad_part2
            
            # Normalize gradient
            norm = np.linalg.norm(grad)
            if norm > 0:
                x -= lr * grad / norm
                
                # Renormalize x to unit energy
                x_norm = np.linalg.norm(x)
                if x_norm > 1e-6:
                    x /= x_norm
                else:
                    x = np.random.choice([-1, 1], size=N).astype(float)
                    x /= np.sqrt(N)
        
        # Track best
        # Compute exact max for tracking
        c_check = np.correlate(x, x, mode='full')
        max_side = np.max(np.abs(np.concatenate([c_check[:N-1], c_check[N:]])))
        if max_side < best_val:
            best_val = max_side
            best_x = x.copy()
            
    # Return best_x
    # The evaluator will compute the score.
    # We return the sequence.
    return best_x
