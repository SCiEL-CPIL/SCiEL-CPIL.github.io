import numpy as np
import time

N = 600
h = 1.0 / 1200.0
DEADLINE = 140.0  # Safety margin for 150s limit

def construct():
    start_time = time.time()
    
    # Initialize with a Gaussian-like bump
    x = np.linspace(-0.25, 0.25, N)
    # Wide Gaussian to start
    sigma = 0.1
    f = np.exp(-x**2 / (2 * sigma**2))
    
    # Ensure non-negative and max=1
    f = np.maximum(f, 0)
    if np.max(f) > 0:
        f = f / np.max(f)
    
    best_score = np.inf
    best_f = f.copy()
    
    # Simple Gradient Descent with adaptive step size
    lr = 0.01
    momentum = 0.9
    v = np.zeros(N)
    
    while time.time() - start_time < DEADLINE:
        # Compute autoconvolution
        # Use FFT for speed
        # Pad to next power of 2 or sufficient length for linear convolution
        # Linear conv of two N-length signals has length 2N-1
        # We only care about the max, which is likely near the center
        # FFT length should be at least 2N-1
        size = 2 * N - 1
        # Next power of 2 for FFT efficiency
        fft_size = 1
        while fft_size < size:
            fft_size *= 2
            
        F = np.fft.rfft(f, n=fft_size)
        C = np.fft.irfft(F * F.conjugate(), n=fft_size)
        
        # Extract the relevant part of the convolution
        # The convolution of two signals of length N centered at 0
        # Our f is defined on indices 0..N-1 corresponding to x in [-0.25, 0.25]
        # The convolution result C is circular if we don't shift.
        # We need linear convolution.
        # np.fft.irfft gives circular convolution.
        # To get linear convolution from FFT:
        # 1. Zero pad to fft_size >= 2N-1.
        # 2. The linear convolution is contained in the first 2N-1 samples of the result?
        # Actually, if we use rfft of length fft_size, the result is circular.
        # But since we zero-padded, the "wrap around" is zero.
        # So C[0:size] is the linear convolution.
        
        c_lin = C[:size]
        
        # Find max
        max_val = np.max(c_lin)
        max_idx = np.argmax(c_lin)
        
        # Calculate Integral
        integral = np.sum(f) * h
        
        # Score
        score = max_val / integral
        
        if score < best_score:
            best_score = score
            best_f = f.copy()
            
        # Compute Gradient
        # d/d f_m (c_lin[max_idx])
        # c_lin[k] = sum_j f_j f_{k-j} (with indices adjusted for the offset)
        # In the linear convolution array c_lin, index k corresponds to shift k.
        # f is indexed 0..N-1.
        # c_lin[k] = sum_{j} f_j f_{k-j} where valid indices are 0<=j<N and 0<=k-j<N.
        # Gradient w.r.t f_m:
        # term 1: f_{k-m} if 0 <= k-m < N
        # term 2: f_{m-k} if 0 <= m-k < N (from the other term in the sum? No, c is symmetric)
        # Actually, c_lin[k] = sum_j f_j f_{k-j}.
        # d c_lin[k] / d f_m = f_{k-m} (if valid) + f_{m-k} (if valid)?
        # No, the sum is over j.
        # If j=m, term is f_{k-m}.
        # If k-j=m => j=k-m, term is f_m f_{k-(k-m)} = f_m f_m? No.
        # Let's use the property that c is the correlation of f and f.
        # c[k] = <f, shift_k(f)>.
        # Gradient is shift_k(f) + shift_{-k}(f).
        
        grad = np.zeros(N)
        # shift_k(f) means f shifted by k.
        # c[k] = sum_j f_j f_{j-k} (using correlation definition where shift is on the second term?)
        # Let's stick to: c[k] = sum_j f_j f_{k-j}.
        # d/d f_m:
        # 1. j=m -> f_{k-m}
        # 2. k-j=m -> j=k-m -> f_{k-m} f_m ? No.
        # The term in the sum involving f_m is:
        # If j=m: f_m f_{k-m}
        # If k-j=m: f_{k-m} f_m (same term if we consider the product)
        # So derivative is 2 * f_{k-m} if 0 <= k-m < N.
        # Wait, is it 2?
        # c[k] = sum_j f_j f_{k-j}.
        # If we differentiate w.r.t f_m:
        # d/d f_m (f_m f_{k-m}) = f_{k-m} + f_m * d/d f_m (f_{k-m})?
        # No, f_{k-m} does not depend on f_m unless k-m=m.
        # So d/d f_m (f_m f_{k-m}) = f_{k-m}.
        # Also, if there is a term where the second factor is f_m:
        # f_{k-j} = f_m => k-j=m => j=k-m.
        # Term is f_{k-m} f_m.
        # d/d f_m (f_{k-m} f_m) = f_{k-m}.
        # So total derivative is 2 * f_{k-m} if the indices are valid and distinct?
        # If k-m = m (i.e., k=2m), then it's d/d f_m (f_m^2) = 2 f_m.
        # So yes, grad[m] = 2 * f[k-m] if 0 <= k-m < N, else 0.
        
        shift = max_idx
        # f[k-m] is f shifted by k.
        # Let's construct the vector g where g[m] = f[shift-m] if valid.
        # This is a reversed and shifted version of f.
        
        # Efficient calculation:
        # grad = 0
        # For m in 0..N-1:
        #   idx = shift - m
        #   if 0 <= idx < N:
        #      grad[m] = 2 * f[idx]
        
        # Vectorized:
        # We want grad[m] = 2 * f[shift-m]
        # This is 2 * f reversed and shifted.
        
        # Let's create a full array of zeros of length 2N
        # Place f in it.
        
        # Simpler:
        # indices for m: 0..N-1
        # indices for f: shift - m
        # We need shift-m to be in [0, N-1]
        
        m_vals = np.arange(N)
        f_indices = shift - m_vals
        valid = (f_indices >= 0) & (f_indices < N)
        
        grad[valid] = 2.0 * f[f_indices[valid]]
        
        # Gradient of Score S = c_k / I
        # dS = (I * dc - c * dI) / I^2
        # dI = sum(1) = N (in terms of sum, but we used integral = sum*h)
        # Let's use sum for gradient to avoid h factors cancelling out
        # S = c_k / (sum(f) * h)
        # dS/df = (1/h) * (I_sum * grad_c - c_k * 1) / I_sum^2
        # where I_sum = sum(f)
        
        I_sum = np.sum(f)
        
        # grad_c is 'grad' calculated above
        # grad_S = (I_sum * grad - c_k) / (I_sum**2)
        # Note: c_k is max_val
        
        grad_S = (I_sum * grad - max_val) / (I_sum**2)
        
        # Update with momentum
        v = momentum * v - lr * grad_S
        f_new = f + v
        
        # Project to non-negative
        f_new = np.maximum(f_new, 0)
        
        # Normalize max to 1
        max_f = np.max(f_new)
        if max_f > 0:
            f_new = f_new / max_f
        else:
            # Fallback
            f_new = f.copy()
            
        f = f_new
        
        # Adaptive learning rate
        # If score improves, keep lr or increase slightly?
        # If score worsens, decrease lr.
        # We don't track previous score easily in this loop without storing.
        # Let's just use a fixed small lr and momentum.
        
    return best_f.tolist()

