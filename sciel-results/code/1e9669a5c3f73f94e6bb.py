import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    # Initialize with random signs
    # Using random signs is a strong start for low peak autocorrelation
    h = np.random.choice([-1.0, 1.0], size=N)
    
    best_h = h.copy()
    best_score = np.inf
    
    # Optimization parameters
    # Start with a relatively high temperature for smoothness
    epsilon = 1.0
    lr = 0.01
    max_iters = 5000
    
    # Pre-compute indices or use np.convolve
    # We need to minimize max(|h*h|) subject to ||h||_1 = 1
    # Since scale invariant, we can fix ||h||_1 = 1
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Compute autoconvolution
        g = np.convolve(h, h, mode='full')
        
        # Compute current score (peak / ||h||_1^2)
        l1_norm = np.sum(np.abs(h))
        if l1_norm < 1e-6:
            # Re-initialize if it collapsed
            h = np.random.choice([-1.0, 1.0], size=N)
            continue
            
        current_peak = np.max(np.abs(g))
        current_score = current_peak / (l1_norm ** 2)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        # Compute gradient of soft-max approximation of max(|g|)
        # L = epsilon * log(sum(exp(|g|/epsilon)))
        # dL/dg_k = (1/epsilon) * exp(|g_k|/epsilon) / sum(exp(|g|/epsilon)) * sign(g_k)
        
        # To avoid overflow in exp, subtract max
        abs_g = np.abs(g)
        max_abs_g = np.max(abs_g)
        
        # If epsilon is small, this approximates the gradient of the max
        # We want to minimize the peak, so we push down the values contributing to the max
        # The gradient of the loss w.r.t h will tell us how to change h to reduce L
        
        # Calculate weights for the soft-max
        # weights_k proportional to exp((|g_k| - max_abs_g) / epsilon)
        exp_term = np.exp((abs_g - max_abs_g) / epsilon)
        sum_exp = np.sum(exp_term)
        
        if sum_exp < 1e-9:
            break
            
        weights = exp_term / sum_exp
        
        # dL/dg = (1/epsilon) * weights * sign(g)
        dL_dg = (1.0 / epsilon) * weights * np.sign(g)
        
        # Gradient w.r.t h
        # grad_h = 2 * conv(dL_dg, h)
        # Note: conv(dL_dg, h) aligns correctly?
        # g_k = sum_j h_j h_{k-j}
        # dL/dh_j = sum_k dL/dg_k (h_{k-j} + h_{j-k})
        # This is indeed 2 * conv(dL_dg, h) if we handle the linear convolution padding correctly.
        # np.convolve(dL_dg, h, mode='full') gives length 2N-1.
        # We need to extract the part corresponding to indices 0..N-1?
        # No, the gradient vector has the same size as h (N).
        # Let's verify the mapping.
        # dL/dh_j depends on g_k where k is related to j.
        # The convolution `conv(dL_dg, h)` produces a vector of length 2N-1.
        # The term `conv(dL_dg, h)_j` is sum_k dL_dg_k h_{j-k}.
        # This matches the first part of the derivative.
        # So we take the central N elements? Or the first N?
        # Let's check indices.
        # g has length 2N-1. h has length N.
        # dL_dg has length 2N-1.
        # conv(dL_dg, h) has length 2N-1 + N - 1 = 3N-2? No.
        # Convolution of length M and K is M+K-1.
        # Here M = 2N-1, K = N. Result length = 3N-2.
        # This is not right. The gradient of h*h w.r.t h is a vector of length N.
        
        # Let's re-derive or use a simpler numerical check or standard library.
        # Actually, `np.convolve` with mode='full' is for linear convolution.
        # The gradient of $L(g)$ where $g = h*h$ w.r.t $h$ is:
        # grad_h = conv(dL_dg, h) + conv(h, dL_dg) ?
        # If we use `mode='full'`, the output is larger.
        # We need to sum the contributions to each $h_j$.
        
        # Let's use the property:
        # grad_h_j = sum_k dL_dg_k * (h_{k-j} + h_{j-k})
        # This is the cross-correlation of dL_dg and h?
        # Actually, it is simpler to compute it via:
        # grad_h = np.convolve(dL_dg, h, mode='full')
        # This gives a vector of length 3N-2.
        # The term corresponding to h_j is at index j + (N-1)?
        # Let's test with small N.
        
        # Alternative: Use FFT for speed and simplicity of gradient?
        # Or just use the fact that we can compute the gradient numerically for a few steps? No, too slow.
        
        # Let's stick to the analytic gradient but be careful with indexing.
        # The convolution `conv(dL_dg, h)` computes:
        # y_m = sum_k dL_dg_k h_{m-k}
        # We want grad_h_j = sum_k dL_dg_k h_{k-j} + sum_k dL_dg_k h_{j-k}
        # Note that h is not zero-padded in the middle for the derivative sum?
        # The indices k run from 0 to 2N-2.
        # The term h_{k-j} is non-zero when 0 <= k-j < N, i.e., j <= k < j+N.
        # The term h_{j-k} is non-zero when 0 <= j-k < N, i.e., j-N < k <= j.
        
        # This is effectively the cross-correlation.
        # Let's use `np.correlate`?
        # np.correlate(dL_dg, h, mode='full')
        # c_m = sum_k dL_dg_k h_{k+m}
        
        # Let's try a different approach:
        # grad_h = 2 * np.correlate(dL_dg, h, mode='full')
        # And then take the appropriate slice?
        
        # Actually, there is a known identity:
        # If g = h * h, then grad_h L = 2 * (grad_g L) * h  (where * is cross-correlation? No, convolution?)
        # In signal processing, the gradient of the energy of a convolution is related to the convolution of the gradient with the signal.
        
        # Let's use a numerical gradient for the first step to verify, or just trust the `2 * conv` structure but handle the size.
        # Wait, `np.convolve` is commutative.
        # Let's assume the gradient vector `grad` is computed as:
        # `grad_full = np.convolve(dL_dg, h, mode='full')`
        # The length is `len(dL_dg) + len(h) - 1 = (2N-1) + N - 1 = 3N-2`.
        # The valid gradient for `h` (length N) should be the central part?
        # Let's check the index of h_0.
        # grad_h_0 = sum_k dL_dg_k h_{k} (from h_{k-0}) + sum_k dL_dg_k h_{-k} (from h_{0-k}, which is 0 for k>0).
        # So grad_h_0 = sum_{k=0}^{N-1} dL_dg_k h_k.
        # This is the dot product of the first N elements of dL_dg and h.
        # In `conv(dL_dg, h)`, index 0 is dL_dg_0 h_0.
        # Index 1 is dL_dg_1 h_0 + dL_dg_0 h_1.
        # This doesn't look like the dot product.
        
        # Let's use `np.correlate`.
        # `np.correlate(dL_dg, h, mode='full')`
        # c_m = sum_k dL_dg_k h_{k+m}
        # For m=0, c_0 = sum_k dL_dg_k h_k.
        # This matches the first part of grad_h_0.
        # Due to symmetry, the gradient is `2 * correlate(dL_dg, h, mode='full')` sliced appropriately.
        # The correlation output length is 3N-2.
        # The "valid" part for the gradient of h (length N) corresponds to the lags where the overlap is complete?
        # No, the gradient sum is over all k.
        # Let's just compute it numerically for the specific indices to be safe, or use the FFT method which is cleaner for gradients.
        
        # FFT Method:
        # g = ifft(fft(h)^2)
        # dL_dg is known.
        # grad_h = ifft(2 * fft(h) * fft(dL_dg)) ?
        # Let's check.
        # L = f(g). dL/dh = conv(dL_dg, h) + conv(h, dL_dg).
        # In frequency domain:
        # FFT(conv(a, b)) = FFT(a) * FFT(b).
        # So FFT(grad_h) = FFT(dL_dg) * FFT(h) + FFT(h) * FFT(dL_dg) = 2 * FFT(dL_dg) * FFT(h).
        # So grad_h = ifft(2 * FFT(dL_dg) * FFT(h)).
        # This is much cleaner and handles the padding automatically if we use zero-padding to length >= 2N-1.
        
        # Let's implement the FFT gradient.
        
        # Pad h and dL_dg to length 2N-1 (or power of 2 for speed)
        size_fft = 2 * N
        H = np.fft.fft(h, n=size_fft)
        DL = np.fft.fft(dL_dg, n=size_fft)
        
        Grad_H = 2 * DL * H
        grad_h = np.fft.ifft(Grad_H).real[:N]
        
        # Update h
        h = h - lr * grad_h
        
        # Normalize h to maintain ||h||_1 = 1
        l1_norm = np.sum(np.abs(h))
        if l1_norm > 1e-9:
            h = h / l1_norm
        else:
            h = np.random.choice([-1.0, 1.0], size=N)
            
        # Decay epsilon and lr
        epsilon *= 0.99
        if i > 100:
            lr *= 0.999
        if epsilon < 0.01:
            epsilon = 0.01
            
    # Final check and return
    # Ensure we return the best one found
    if best_score < np.inf:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
