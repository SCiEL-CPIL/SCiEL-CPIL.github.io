"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600
DX = 1.0 / N  # interval width

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Initial guess: flat function
    h = np.ones(N)
    
    def autoconv_values(h):
        """Compute autoconvolution at grid centers using FFT-based linear convolution."""
        # For step function, autoconv at point x_k = -1/4 + (k+0.5)*DX
        # (f*f)(x) = integral f(t) f(x-t) dt
        # For discrete step functions, we can compute via convolution of the height array
        # with appropriate weighting
        # Linear convolution of h with h, then sample at appropriate points
        # The autoconvolution of a piecewise constant function on [-1/4, 1/4]
        # at grid points can be computed as:
        # conv[i] = sum_j h[j] * overlap_length(i, j)
        # where overlap is the length of intersection of intervals
        
        # Simpler: use np.convolve for linear convolution, then interpret
        # The autoconvolution at the grid centers
        # Let's compute the full linear convolution of the discrete signal
        # h represents values on intervals [x_i, x_{i+1}]
        # The autoconvolution sampled at grid centers
        
        # Using the fact that for step functions on uniform grid:
        # (f*f)(x_k) = DX^2 * sum_{i,j: x_i + x_j ≈ x_k} h[i]*h[j] approximately
        # More precisely, use convolution
        
        # Compute linear convolution
        conv = np.convolve(h, h, mode='full')
        # conv has length 2N-1
        # The autoconvolution at grid center k corresponds to conv index k + N - 1 + 0.5? 
        # Let's think: the support of f is [-1/4, 1/4], so f*f has support [-1/2, 1/2]
        # Grid centers for f*f would be at -1/2 + (m+0.5)*DX for m=0,...,2N-1
        # But we only care about the values, and we want to minimize the max
        
        # Actually, let's just compute the autoconvolution properly.
        # For a step function with heights h[i] on intervals [a_i, a_{i+1}],
        # the autoconvolution at point s is:
        # (f*f)(s) = integral f(t) f(s-t) dt
        # = sum_i h[i] * integral over t in [a_i, a_{i+1}] of f(s-t) dt
        # f(s-t) is h[j] when s-t in [a_j, a_{j+1}], i.e., t in [s - a_{j+1}, s - a_j]
        # The overlap of [a_i, a_{i+1}] and [s - a_{j+1}, s - a_j] gives the length
        
        # This is O(N^2) per evaluation which is 360000 ops, fine for gradient descent
        # But let's use FFT for speed
        
        # Use FFT-based linear convolution
        nfft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        conv_fft = np.fft.irfft(H * H, n=nfft)
        # conv_fft[0:2*N-1] is the linear convolution
        # The autoconvolution at grid center k of the original domain corresponds to
        # a specific index in the convolution
        
        # Let's just return the full convolution values that are relevant
        # The peak will be somewhere in the middle
        return conv_fft[:2*N-1]
    
    def objective_and_grad(h, temp=0.1):
        """Compute smooth max of autoconvolution and its gradient."""
        conv = autoconv_values(h)
        # Normalize: divide by (sum h * DX)^2 to get normalized max
        integral = np.sum(h) * DX
        norm = integral * integral
        conv_norm = conv / norm
        
        # Soft max via log-sum-exp
        m = np.max(conv_norm)
        logsumexp = m + np.log(np.sum(np.exp(conv_norm - m)))
        # Gradient of logsumexp w.r.t. conv_norm
        weights = np.exp(conv_norm - m) / np.sum(np.exp(conv_norm - m))
        
        # Now we need gradient w.r.t. h
        # conv = autoconv(h) is quadratic in h
        # d(conv[k])/d(h[i]) = 2 * (convolution of delta_i with h)[k] approximately
        # For linear convolution: conv[k] = sum_j h[j] * h[k-j] (with proper indexing)
        # d(conv[k])/d(h[i]) = h[k-i] + h[i-k] if in bounds
        
        # This is getting complex. Let's use a simpler approach:
        # Just do gradient descent on the peak value using subgradient
        # Or use the fact that we can compute the gradient of the autoconv
        
        # Simpler: return the max value and use subgradient (the index where max occurs)
        idx = np.argmax(conv_norm)
        # Gradient of conv[idx] w.r.t. h
        # conv[idx] = sum_{i,j} h[i]*h[j] * overlap(i,j,idx)
        # For our discrete approximation, conv from np.convolve:
        # conv[k] = sum_j h[j] * h[k-j] for k=0..2N-2
        # d(conv[idx])/d(h[i]) = h[idx-i] + h[i-idx] if valid indices
        
        grad_h = np.zeros(N)
        for i in range(N):
            j1 = idx - i
            j2 = i - idx
            if 0 <= j1 < N:
                grad_h[i] += h[j1]
            if 0 <= j2 < N:
                grad_h[i] += h[j2]
        
        # Normalize gradient
        grad_h = grad_h / norm
        
        return conv_norm[idx], grad_h
    
    def compute_score(h):
        conv = autoconv_values(h)
        integral = np.sum(h) * DX
        norm = integral * integral
        return np.max(conv) / norm
    
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Projected gradient descent
    lr = 0.01
    for it in range(5000):
        if time.time() > deadline:
            break
        
        score, grad = objective_and_grad(h)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Gradient step
        h_new = h - lr * grad
        # Project onto nonnegative orthant
        h_new = np.maximum(h_new, 0.0)
        
        # Normalize to keep integral constant (optional, but helps)
        # Actually, the problem says nonnegative step function, no constraint on integral
        # But scaling matters. Let's keep it simple.
        
        h = h_new
    
    # Final polish: try to reduce the peak by adjusting only the heights at the peak region
    # Return the best found
    return list(best_h)
# EVOLVE-BLOCK-END
