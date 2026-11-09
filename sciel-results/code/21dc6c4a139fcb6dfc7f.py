import numpy as np
import time

N = 600
DEADLINE = 140.0

def _autoconv(h):
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    H = np.fft.rfft(h, nfft)
    conv = np.fft.irfft(H * H, nfft)
    return conv[:2 * N - 1]

def _objective_and_grad(h, temp=1.0):
    conv = _autoconv(h)
    s = np.sum(h)
    if s < 1e-12:
        s = 1e-12
    norm = s * s
    
    z = conv / norm
    z_max = np.max(z)
    z_shifted = z - z_max
    lse = z_max + temp * np.log(np.sum(np.exp(z_shifted / temp)))
    value = lse
    
    weights = np.exp(z_shifted / temp)
    weights /= np.sum(weights)
    
    # Gradient via FFT
    # grad_h_i = (1/norm) * sum_k weights[k] * (h[i+k-N+1] + h[i-k+N+1])
    # = (1/norm) * [ (w * h)_i + (w * h)_i ] where w is centered
    # Let's compute: term1_i = sum_k weights[k] h[i+k-(N-1)]
    # term2_i = sum_k weights[k] h[i-k+(N-1)]
    # 
    # term1: sum_k w[k] h[i+k-(N-1)] 
    # Let's use FFT: create w_full of length 2N-1 centered at N-1
    # term1_i = sum_k w[k] h[i-(N-1)+k]
    # This is the cross-correlation of w and h shifted
    # 
    # Using FFT: 
    # A = fft(w, nfft), B = fft(h, nfft)
    # term1 = ifft(A * conj(B)) gives sum_k w[k] h[i-k] ... not quite
    # 
    # Let me just use np.convolve for correctness, N=600 is small enough
    
    # term1_i = sum_k weights[k] * h[i + k - (N-1)]
    # = sum_k weights[k] * h[(i - (N-1)) + k]
    # This is (weights * h)[i - (N-1)] if we define conv as sum_k a[k] b[n-k]
    # No: sum_k a[k] b[n-k] with n = i-(N-1) gives sum_k a[k] b[i-(N-1)-k]
    # But we want sum_k a[k] b[i-(N-1)+k]
    # = sum_k a[k] b[i-(N-1)-(-k)] = (a * b_rev)[i-(N-1)] where b_rev[j] = b[-j]
    # 
    # Easier: just flip weights
    # sum_k w[k] h[i+k-(N-1)] = sum_k w[k] h[(i-(N-1))-(-k)]
    # Let m = -k, k = -m: sum_m w[-m] h[i-(N-1)-m]
    # = sum_m w_flipped[m] h[i-(N-1)-m] where w_flipped[m] = w[-m]
    # = (w_flipped * h)[i-(N-1)] in standard conv notation sum_m a[m] b[n-m]
    
    w_flipped = weights[::-1]  # w_flipped[m] = w[2N-2-m] = w[-m] in centered sense
    # Actually weights has indices 0..2N-2. w_flipped[k] = weights[2N-2-k]
    
    # term1_i = sum_k weights[k] h[i+k-(N-1)]
    # Let's compute via convolve: 
    # conv_w_h = np.convolve(w_flipped, h, mode='full')
    # conv_w_h[n] = sum_m w_flipped[m] h[n-m]
    # We want sum_k weights[k] h[i+k-(N-1)] = sum_k weights[k] h[i-(N-1)+k]
    # = sum_m w_flipped[2N-2-m] h[i-(N-1)+m]  ... this is getting confusing
    
    # Let me just do it directly with a vectorized approach
    # For each i, term1_i = sum_k weights[k] h[i+k-(N-1)]
    # Create matrix M[i,k] = h[i+k-(N-1)] if valid, else 0
    # term1 = M @ weights
    
    # But that's O(N^2) memory. Let me use the fact that this is a Toeplitz matrix-vector product.
    # I'll use FFT for the convolution.
    
    # term1_i = sum_k weights[k] h[i+k-(N-1)]
    # Define a[k] = weights[k] for k=0..2N-2
    # Define b[j] = h[j] for j=0..N-1
    # term1_i = sum_k a[k] b[i+k-(N-1)]
    # Let n = i - (N-1), then term1_i = sum_k a[k] b[n+k]
    # This is the cross-correlation: (a ⋆ b)[n] = sum_k a[k] b[n+k]
    # In FFT: (a ⋆ b) = ifft(fft(a) * conj(fft(b))) ... let me verify
    # Actually, cross-correlation (a ⋆ b)[n] = sum_k a[k] b[n+k]
    # FFT of cross-correlation: F(a ⋆ b) = F(a) * conj(F(b))
    # So (a ⋆ b) = ifft(F(a) * conj(F(b)))
    
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    
    A = np.fft.rfft(weights, nfft)
    B = np.fft.rfft(h, nfft)
    
    # term1 = ifft(A * conj(B))
    term1_full = np.fft.irfft(A * np.conj(B), nfft)
    # term1_full[n] = sum_k weights[k] h[n+k] for appropriate n
    # We want term1_i = term1_full[i - (N-1)]
    term1 = term1_full[N-1:N-1+N]  # indices N-1 to 2N-2
    
    # term2_i = sum_k weights[k] h[i-k+(N-1)]
    # = sum_k weights[k] h[i+(N-1)-k]
    # Let n = i + (N-1), term2_i = sum_k weights[k] h[n-k]
    # This is standard convolution: (weights * h)[n] = sum_k weights[k] h[n-k]
    # = convolve(weights, h)[n]
    
    term2_full = np.fft.irfft(A * B, nfft)
    # term2_full[n] = sum_k weights[k] h[n-k]
    # We want term2_i = term2_full[i + (N-1)]
    term2 = term2_full[N-1:N-1+N]
    
    grad_h = (term1 + term2) / norm
    
    return value, grad_h

def construct():
    start_time = time.time()
    
    # Initialize with a smooth bump
    x = np.linspace(-1/4, 1/4, N)
    sigma = 0.12
    h = np.exp(-x**2 / (2 * sigma**2))
    h = np.maximum(h, 0.0)
    h = h / np.sum(h)
    
    best_h = h.copy()
    best_val = _objective_and_grad(h)[0]
    
    lr = 0.01
    temp = 1.0
    min_temp = 0.01
    
    while time.time() - start_time < DEADLINE:
        val, grad = _objective_and_grad(h, temp)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        h = h - lr * grad
        h = np.maximum(h, 0.0)
        
        # Adaptive parameters
        if val > best_val * 1.005:
            lr *= 0.95
            temp *= 0.98
            temp = max(temp, min_temp)
        else:
            lr *= 1.005
            temp *= 0.995
            temp = max(temp, min_temp)
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Final polish
    lr = 0.001
    temp = min_temp
    for _ in range(100):
        if time.time() - start_time > DEADLINE - 1:
            break
        val, grad = _objective_and_grad(best_h, temp)
        if val < best_val:
            best_val = val
        h_new = best_h - lr * grad
        h_new = np.maximum(h_new, 0.0)
        val_new = _objective_and_grad(h_new, temp)[0]
        if val_new < best_val:
            best_h = h_new
            best_val = val_new
    
    return list(best_h)
