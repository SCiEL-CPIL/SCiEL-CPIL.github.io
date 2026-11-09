import numpy as np
from scipy import signal
import time

N = 600

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 120.0  # Safety margin
    
    best_h = np.ones(N)
    best_score = np.inf
    
    # Try a few random initializations
    for _ in range(5):
        if time.time() - start_time > max_time:
            break
            
        # Initialize u (h = u^2)
        # Start with noise around 1
        u = np.random.randn(N) * 0.1 + 1.0
        
        # Gradient descent
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > max_time:
                break
                
            h = u**2
            
            # Compute autoconvolution
            # Use FFT for speed
            # Zero pad to 2*N to avoid circular convolution artifacts
            h_fft = np.fft.rfft(h, n=2*N)
            c = np.fft.irfft(h_fft * np.conj(h_fft), n=2*N)
            
            # The peak is likely in the middle
            # Find max
            max_val = np.max(c)
            
            # Normalize by L2 norm squared
            norm_sq = np.sum(h**2)
            
            score = max_val / norm_sq
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
            # Compute gradient
            # d(score)/du
            # score = M / S
            # dM/du: M is max(c). Assume max is at index k*.
            # c_k = sum_i h_i h_{k-i}
            # dc_k/du_j = 2 u_j (h_{k-j} + h_{j-k})
            
            # Find index of max
            k_star = np.argmax(c)
            
            # Compute gradient of c_k_star with respect to h
            # This is a convolution of h with a delta at k_star?
            # c_k = h * h. dc_k/dh = h * delta_k + delta_k * h = h(k-.) + h(. - k)
            
            # Efficiently:
            # grad_h = np.zeros(N)
            # We need indices i such that k_star - i is in [0, N-1]
            # and i is in [0, N-1]
            
            # Let's compute the gradient term for the max index
            # Note: if there are multiple maxes, we should sum gradients? 
            # For now, use the single max index.
            
            # grad_c = np.zeros(2*N)
            # grad_c[k_star] = 1.0
            # grad_h = signal.convolve(h, grad_c, mode='valid') + signal.convolve(grad_c, h, mode='valid')
            # This is slow.
            
            # Vectorized approach for grad_h:
            # h is length N.
            # We want g_j = 2 u_j (h_{k-j} + h_{j-k})
            
            # Create index arrays
            j = np.arange(N)
            # Indices for h_{k-j}
            idx1 = k_star - j
            # Indices for h_{j-k}
            idx2 = j - k_star
            
            # Get values, 0 if out of bounds
            # Since h is defined on [0, N-1], we use 0 for out of bounds
            # But wait, the convolution is linear.
            # h is zero outside [0, N-1].
            
            h_vals1 = np.zeros(N)
            mask1 = (idx1 >= 0) & (idx1 < N)
            h_vals1[mask1] = h[idx1[mask1]]
            
            h_vals2 = np.zeros(N)
            mask2 = (idx2 >= 0) & (idx2 < N)
            h_vals2[mask2] = h[idx2[mask2]]
            
            grad_h = 2.0 * u * (h_vals1 + h_vals2)
            
            # Gradient of S = sum h^2 = sum u^4
            # dS/du = 4 u^3
            grad_S = 4.0 * u**3
            
            # d(score)/du = (1/S) * dM/du - (M/S^2) * dS/du
            # dM/du = grad_h (assuming k* fixed)
            grad_score = (1.0 / norm_sq) * grad_h - (max_val / (norm_sq**2)) * grad_S
            
            # Update u
            u -= lr * grad_score
            
            # Project to positive? u can be negative, h=u^2 is positive.
            # But large u might cause issues.
            # Clip u to avoid explosion
            u = np.clip(u, -10, 10)
            
            # Adaptive learning rate?
            if step % 50 == 0:
                lr *= 0.9

    return list(best_h)
