import numpy as np
from scipy.optimize import minimize
import time

N = 600
DELTA = (1.0/2.0) / N  # interval width = 1/1200

def compute_autoconv(h):
    """Compute the autoconvolution of the step function at interval centers.
    h: array of N nonnegative heights.
    Returns array of length 2*N-1 representing (f*f) at shifted positions.
    """
    # Discrete convolution of h with itself, scaled by DELTA
    conv = np.convolve(h, h, mode='full') * DELTA
    return conv

def compute_score(h):
    """Compute normalized max autoconvolution: max(f*f) / (int f)^2"""
    conv = compute_autoconv(h)
    max_conv = np.max(conv)
    integral = np.sum(h) * DELTA
    if integral <= 0:
        return 1e10
    return max_conv / (integral * integral)

def smooth_max_surrogate(h, beta=100.0):
    """Smooth surrogate for max(f*f)/(int f)^2 using log-sum-exp."""
    conv = compute_autoconv(h)
    integral = np.sum(h) * DELTA
    if integral <= 1e-15:
        return 1e10
    # log-sum-exp for stability
    m = np.max(conv)
    if m < 0:
        m = 0
    lse = m + np.log(np.sum(np.exp((conv - m) * beta)) / beta)
    return lse / (integral * integral)

def grad_surrogate(h, beta=100.0):
    """Gradient of the smooth surrogate."""
    conv = compute_autoconv(h)
    integral = np.sum(h) * DELTA
    if integral <= 1e-15:
        return np.ones_like(h), 1e10
    
    m = np.max(conv)
    if m < 0:
        m = 0
    # Soft-max weights
    w = np.exp((conv - m) * beta)
    w_sum = np.sum(w)
    if w_sum < 1e-300:
        w = np.zeros_like(conv)
        w[np.argmax(conv)] = 1.0
        w_sum = 1.0
    w = w / w_sum
    
    # Gradient of max_conv w.r.t. h: d/dh_i sum_j w_j conv_j
    # conv_j = DELTA * sum_k h_k h_{j-k} (with appropriate indexing)
    # d conv_j / d h_i = DELTA * (h_{j-i} + h_{j-i+N} ... ) -- use convolution gradient
    # For full convolution, conv[j] = sum_k h[k]*h[j-k] * DELTA for valid ranges
    # d conv[j]/d h[i] = DELTA * (h[j-i] if 0<=j-i<N else 0 + h[i-j+N] if ... )
    # Simpler: use the fact that d/dh (h*h) in convolution sense
    # Let's compute gradient numerically-free via:
    # conv = DELTA * (h * h) where * is full convolution
    # d conv[j] / d h[i] = DELTA * (h[j-i] + h[i-(j-N+1)]...) 
    # Actually for full conv: conv[j] = sum_{k} h[k] h[j-k] * DELTA, k from max(0,j-N+1) to min(N-1,j)
    # d conv[j]/d h[i] = DELTA * (h[j-i] + h[i-j+N-1] if in range)
    # Let's just do: grad_conv = DELTA * (w * h convolved appropriately)
    # The gradient of (h*h)_full[j] w.r.t h[i] = h[j-i] + h[i-j+N-1] (with bounds)
    # This is equivalent to: grad = DELTA * (w_conv_full with h) where we do two convolutions
    
    # d/dh_i [sum_j w_j DELTA sum_k h[k]h[j-k]] = DELTA sum_j w_j [h[j-i] + h[i-j+N-1]] (with bounds)
    # = DELTA * (w * h)_full evaluated at i, plus the symmetric term
    # Actually: sum_j w_j h[j-i] = (w * h_rev)[i] where h_rev is reversed... 
    # Let me just compute it as:
    # term1 = sum_j w_j h[j-i] for valid j,i -> this is cross-correlation
    # Use: grad = DELTA * (np.convolve(w, h, 'full')[i] + np.convolve(w, h[::-1], 'full')[N-1-i] ...)
    
    # Simpler approach: the Jacobian of full convolution
    # conv = DELTA * convolve(h, h, 'full')
    # d conv[j] / d h[i] = DELTA * (h[j-i] if 0 <= j-i < N else 0) + DELTA * (h[i-j+N-1] if 0 <= i-j+N-1 < N else 0)
    # But note: conv[j] = sum_k h[k] h[j-k] DELTA, so d/dh[i] = DELTA * (h[j-i] + h[i-j+N-1]) with bounds
    # Let's compute:
    # G[j,i] = DELTA * (h[j-i] if 0<=j-i<N else 0) + DELTA * (h[i-j+N-1] if 0<=i-j+N-1<N else 0)
    # grad_h[i] = sum_j w[j] * G[j,i]
    
    # Efficient: 
    # sum_j w[j] h[j-i] = (w * h_flip)[i] where h_flip[k] = h[N-1-k]
    # Actually sum_j w[j] h[j-i] for j from max(0,i) to min(N-1, N-1+i) 
    # = correlation of w and h shifted
    # Let's use: c1 = np.correlate(w, h, mode='full') -> c1[i] = sum_j w[j] h[j-i+N-1]... 
    # This is getting complex. Let me just use a simpler gradient computation.
    
    # Alternative: since N=600, we can compute the gradient more directly
    # grad[i] = DELTA * sum_j w[j] * (h[j-i] + h[i-j+N-1]) with bounds
    # = DELTA * (sum_j w[j] h[j-i] + sum_j w[j] h[i-j+N-1])
    # First term: for fixed i, sum over j where 0<=j-i<N, i.e., i<=j<i+N, j in [0,N-1]
    #   = sum_{j=max(0,i)}^{min(N-1,i+N-1)} w[j] h[j-i]
    # Second term: for fixed i, sum over j where 0<=i-j+N-1<N, i.e., i-N+1<=j<=i, j in [0,N-1]
    #   = sum_{j=max(0,i-N+1)}^{min(N-1,i)} w[j] h[i-j+N-1]
    
    # Use FFT for efficiency
    # First term: (w * h)[i] where * is convolution but we need h[j-i] which is correlation
    # sum_j w[j] h[j-i] = (w corr h)[i] = conv(w, h_flip, 'full')[N-1-i]... 
    # Let me just use:
    # A = np.convolve(w, h, mode='full')  # A[k] = sum_j w[j] h[k-j]
    # We want sum_j w[j] h[j-i] = sum_j w[j] h[-(i-j)] = A[N-1-i] if we flip h
    # h_flip = h[::-1]
    # sum_j w[j] h[j-i] = sum_j w[j] h_flip[N-1-(j-i)] = sum_j w[j] h_flip[N-1-j+i]
    # = conv(w, h_flip, 'full')[i] ... let me verify with small example
    
    # Actually, let me just compute it directly for correctness, N=600 is small enough
    # But 600*600 = 360000 ops per gradient, times iterations... might be slow.
    # Let me use the FFT approach properly.
    
    # The gradient of the objective f(h) = (1/int^2) * sum_j w_j conv_j
    # where conv_j = DELTA * sum_k h[k] h[j-k]
    # df/dh_i = (1/int^2) * sum_j w_j DELTA (h[j-i] + h[i-j+N-1]) - 2*(1/int^3) * (sum_j w_j conv_j) * DELTA * h_i
    # Wait, we also need to account for the denominator (int f)^2
    
    obj_val = np.sum(w * conv)  # this is the soft-max approx of max_conv
    int_val = integral
    
    # d obj / d h_i = (1/int^2) * d obj_val / d h_i - 2*obj_val/int^3 * d(int)/d h_i
    # d obj_val / d h_i = DELTA * sum_j w_j (h[j-i] + h[i-j+N-1]) with bounds
    # d(int)/d h_i = DELTA
    
    # Compute sum_j w_j h[j-i] using FFT
    # Let's define: for the first sum S1[i] = sum_j w[j] h[j-i] (with bounds)
    # This is the cross-correlation of w and h
    # Using: S1[i] = sum_j w[j] h[j-i] = (w * h_rev)[i] where appropriate
    # h_rev[k] = h[N-1-k]
    # S1[i] = sum_j w[j] h_rev[N-1-j+i] = conv(w, h_rev, 'full')[i + N-1]... 
    
    # Let me just use np.correlate
    # np.correlate(w, h, mode='full')[k] = sum_j w[j] h[j + k - (N-1)]... 
    # This is confusing. Let me use a different approach.
    
    # Since N=600, let me just compute the gradient in O(N^2) but vectorized
    # S1[i] = sum_{j} w[j] * h[j-i] where 0<=j-i<N
    # Create matrix M where M[j,i] = h[j-i] if 0<=j-i<N else 0
    # Then S1 = w @ M
    # M is a Toeplitz-like matrix
    
    # Actually, let me just use the simple approach with np.roll or direct indexing
    # For efficiency, use FFT:
    # sum_j w[j] h[j-i] = IFFT(FFT(w) * FFT(h)) with appropriate shifts
    # This is the circular cross-correlation, but we need linear.
    
    # Let me just do it the direct way but vectorized:
    # S1[i] = sum_{j=max(0,i)}^{min(N-1, i+N-1)} w[j] h[j-i]
    # Let d = j-i, so d ranges from max(0,i)-i to min(N-1,i+N-1)-i = max(-i,0) to min(N-1-i, N-1)
    # d from max(-i, 0) to min(N-1-i, N-1)
    
    # I'll use the convolution approach:
    # S1 = np.correlate(w, h, mode='full')[N-1:]  # This gives S1[i] = sum_j w[j] h[j+i-(N-1)]... 
    # Let me test mentally: correlate(w,h,'full')[k] = sum_j w[j] h[j+k-(N-1)] for k=0..2N-2
    # We want sum_j w[j] h[j-i] = sum_j w[j] h[j+(-i)] 
    # = correlate(w,h,'full')[N-1-i] for i=0..N-1
    
    corr_wh = np.correlate(w, h, mode='full')
    # S1[i] = corr_wh[N-1-i] for i in 0..N-1
    S1 = corr_wh[N-1::-1][:N]  # S1[i] = corr_wh[N-1-i]
    
    # S2[i] = sum_j w[j] h[i-j+N-1] where 0<=i-j+N-1<N, i.e., i-N+1<=j<=i
    # = sum_j w[j] h[(i-j)+(N-1)] = sum_j w[j] h_rev[j-i] where h_rev[k]=h[N-1-k]
    # h[i-j+N-1] = h_rev[j-i]
    # So S2[i] = sum_j w[j] h_rev[j-i] = same as S1 but with h replaced by h_rev
    h_rev = h[::-1]
    corr_wh_rev = np.correlate(w, h_rev, mode='full')
    S2 = corr_wh_rev[N-1::-1][:N]
    
    d_obj_val = DELTA * (S1 + S2)
    
    grad_obj = (1.0 / (int_val * int_val)) * d_obj_val - (2.0 * obj_val / (int_val ** 3)) * DELTA * np.ones(N)
    
    return grad_obj, obj_val / (int_val * int_val)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with a smooth initial guess (Gaussian-like)
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA/2
    h = np.exp(-100 * x**2)
    h = np.maximum(h, 0)
    h = h / np.sum(h) * 1.0  # normalize
    
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Try a few initial conditions
    initial_guesses = [
        h,
        np.ones(N) / N,  # uniform
        np.exp(-50 * x**2),  # wider Gaussian
        np.exp(-200 * x**2),  # narrower Gaussian
        0.5 + 0.5*np.cos(2*np.pi*x),  # cosine bump
    ]
    
    for h0 in initial_guesses:
        if time.time() > deadline - 30:
            break
        h0 = np.maximum(h0, 0)
        if np.sum(h0) < 1e-15:
            continue
        h0 = h0 / np.sum(h0)
        
        def obj_func(h_flat):
            h = np.maximum(h_flat, 0)
            return compute_score(h)
        
        def obj_grad(h_flat):
            h = np.maximum(h_flat, 0)
            g, val = grad_surrogate(h, beta=50.0)
            # Mask gradient where h=0 (subgradient)
            g = g * (h_flat > 0)
            return g, val
        
        # Use L-BFGS-B with bounds
        bounds = [(0, None)] * N
        
        try:
            result = minimize(
                lambda h: compute_score(h),
                h0.copy(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            if time.time() > deadline - 20:
                break
            if result.fun < best_score:
                best_h = np.maximum(result.x, 0)
                best_score = result.fun
        except Exception:
            pass
    
    # Polish with more iterations if time allows
    if time.time() < deadline - 15:
        try:
            result = minimize(
                lambda h: compute_score(h),
                best_h.copy(),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
            )
            if result.fun < best_score:
                best_h = np.maximum(result.x, 0)
                best_score = result.fun
        except Exception:
            pass
    
    # Final check
    final_score = compute_score(best_h)
    if final_score < best_score:
        best_score = final_score
    
    return list(best_h)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with a smooth initial guess (Gaussian-like)
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA/2
    h = np.exp(-100 * x**2)
    h = np.maximum(h, 0)
    h = h / np.sum(h) * 1.0
    
    best_h = h.copy()
    best_score = compute_score(h)
    
    initial_guesses = [
        h,
        np.ones(N) / N,
        np.exp(-50 * x**2),
        np.exp(-200 * x**2),
        0.5 + 0.5*np.cos(2*np.pi*x),
    ]
    
    for h0 in initial_guesses:
        if time.time() > deadline - 30:
            break
        h0 = np.maximum(h0, 0)
        if np.sum(h0) < 1e-15:
            continue
        h0 = h0 / np.sum(h0)
        
        try:
            result = minimize(
                lambda h: compute_score(h),
                h0.copy(),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            if time.time() > deadline - 20:
                break
            if result.fun < best_score:
                best_h = np.maximum(result.x, 0)
                best_score = result.fun
        except Exception:
            pass
    
    if time.time() < deadline - 15:
        try:
            result = minimize(
                lambda h: compute_score(h),
                best_h.copy(),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
            )
            if result.fun < best_score:
                best_h = np.maximum(result.x, 0)
                best_score = result.fun
        except Exception:
            pass
    
    final_score = compute_score(best_h)
    if final_score < best_score:
        best_score = final_score
    
    return list(best_h)
# EVOLVE-BLOCK-END
