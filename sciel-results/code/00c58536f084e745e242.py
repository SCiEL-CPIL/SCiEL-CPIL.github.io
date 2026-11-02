import numpy as np
import time

N = 600

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Initialize with a raised cosine or similar non-flat shape
    # x is from -0.5 to 0.5 (normalized indices)
    x = np.linspace(-0.5, 0.5, N, endpoint=False)
    # Raised cosine: 0.5 * (1 + cos(2*pi*x))
    f = 0.5 * (1 + np.cos(2 * np.pi * x))
    # Ensure non-negative
    f = np.maximum(f, 0)
    
    # Normalize to sum 1
    f = f / np.sum(f)
    
    # Optimization loop
    # Use log-sum-exp to approximate max
    beta = 10.0  # Smoothness parameter
    lr = 0.01
    steps = 5000
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        # Compute autoconvolution
        # Use FFT for speed
        F = np.fft.fft(f, n=N*2)
        conv = np.fft.ifft(F * F).real[:N]
        
        # Log-sum-exp
        max_conv = np.max(conv)
        log_sum_exp = max_conv + np.log(np.sum(np.exp(conv - max_conv)))
        
        # Gradient of log-sum-exp w.r.t conv
        # d(L)/d(conv_k) = exp(conv_k - L)
        grad_conv = np.exp(conv - log_sum_exp)
        
        # Gradient w.r.t f
        # conv_k = sum_j f_j f_{j+k}
        # d(conv)/df is the autoconvolution matrix
        # We can compute grad_f = 2 * (f * grad_conv) in frequency domain?
        # No, d(conv)/df is a Toeplitz matrix.
        # grad_f = 2 * (f * grad_conv) is only true if conv is circular and symmetric?
        # Actually, d(df*f)/df = 2 * (f * grad_conv) where * is convolution.
        # Let's use the property: grad_f = 2 * np.fft.ifft(np.fft.fft(f) * np.fft.fft(grad_conv, n=N*2)).real[:N]
        # Wait, the gradient of the quadratic form x^T A x is (A + A^T) x.
        # Here A is the autoconvolution matrix.
        # The gradient of sum_k g_k (f*f)_k is 2 * (f * g) where g is the gradient vector.
        
        grad_f = 2 * np.fft.ifft(np.fft.fft(f, n=N*2) * np.fft.fft(grad_conv, n=N*2)).real[:N]
        
        # Update
        f = f - lr * grad_f
        
        # Project to non-negative
        f = np.maximum(f, 0)
        
        # Renormalize
        f = f / np.sum(f)
        
        # Adaptive beta?
        if i % 100 == 0:
            beta *= 1.1
            lr *= 0.9

    # Final normalization
    f = f / np.sum(f)
    
    return list(f)

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Length: {len(res)}")
    print(f"Sum: {sum(res)}")
