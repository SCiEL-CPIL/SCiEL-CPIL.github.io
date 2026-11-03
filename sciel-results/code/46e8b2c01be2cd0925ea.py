"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
import time

N = 600

# EVOLVE-BLOCK-START
def _autoconv_peak(x):
    """Compute the peak of the autoconvolution of x (length N)."""
    # x is nonnegative. Autoconvolution c[k] = sum_i x[i] x[k-i] for appropriate k.
    # Using FFT: c = ifft(fft(x)^2)
    n = len(x)
    # Use a larger FFT size to avoid circular convolution effects
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    c = np.fft.irfft(X * X, size)[:2 * n - 1]
    return np.max(c)

def _score(x):
    """The normalized maximum of autoconvolution.
    
    For a step function on [-1/4, 1/4] with N equal intervals, the autoconvolution
    is computed in continuous time. The discrete approximation:
    - Each bin has width h = (1/4 - (-1/4)) / N = (1/2) / N = 1/(2N)
    - The continuous autoconvolution at point t is approximately sum_i x_i * x_{i+k} * h
      where k is the bin offset corresponding to t.
    
    Actually, let's think more carefully. If f(x) = x_i for x in [a_i, a_{i+1}),
    then (f*f)(t) = ∫ f(s) f(t-s) ds.
    
    For step functions with equal intervals, the autoconvolution is a piecewise
    quadratic function. The discrete approximation using the trapezoidal rule or
    simple Riemann sum:
    
    (f*f)(t_k) ≈ h * sum_i x_i x_{i+k}
    
    where h = 1/(2N) is the bin width, and the sum is over valid i.
    
    The "normalized maximum" likely refers to max_t (f*f)(t) / (some normalization).
    
    Looking at the task: "minimize the normalized maximum of its autoconvolution; 
    this gives an upper bound on C1."
    
    C1 is likely the first autocorrelation constant. For a function f on [-1/4, 1/4],
    the first autocorrelation inequality involves:
    
    C1 >= max_{|t| <= 1/4} (f*f)(t) / (∫f)^2
    
    or similar. Let me assume the score is:
    
    score = max_t (f*f)(t) / (∫f dx)^2
    
    where ∫f dx ≈ h * sum(x_i).
    """
    h = 1.0 / (2 * N)
    integral = h * np.sum(x)
    if integral <= 0:
        return np.inf
    peak = _autoconv_peak(x) * h  # The discrete sum approximates (1/h) * (f*f), so multiply by h
    # Actually, let's reconsider. The discrete autoconvolution sum S[k] = sum_i x_i x_{i+k}
    # approximates (1/h) * (f*f)(t_k) because:
    # (f*f)(t) = ∫ f(s) f(t-s) ds ≈ h * sum_i f(s_i) f(t - s_i) = h * sum_i x_i x_{i+k}
    # So (f*f)(t_k) ≈ h * S[k]
    # And (∫f)^2 ≈ (h * sum(x))^2
    # So the ratio is: h * max(S) / (h * sum(x))^2 = max(S) / (h * (sum(x))^2)
    # Wait, that doesn't seem right either. Let me just compute it directly.
    
    # (f*f)(t) ≈ h * S[k] where S[k] = sum_i x_i x_{i+k}
    # (∫f dx)^2 = (h * sum(x))^2
    # ratio = h * max(S) / (h^2 * (sum(x))^2) = max(S) / (h * (sum(x))^2)
    
    return peak / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial guess: uniform
    x = np.ones(N)
    
    # Normalize so that integral = 1 (this doesn't change the ratio since it's homogeneous of degree 0)
    # Actually, the score is scale-invariant: if we multiply x by c, both numerator and denominator
    # scale by c^2, so the ratio is unchanged. Let's just work with the raw values.
    
    best_x = x.copy()
    best_score = _score(x)
    
    # Subgradient method: minimize the peak autoconvolution
    # The peak is at some lag k*. The subgradient with respect to x[i] is:
    # d/dx[i] (h * S[k*]) / (h * sum(x))^2 * ... 
    # This is complex. Let's use a simpler approach.
    
    # Alternative: Use coordinate descent or projected gradient.
    # Since we want to minimize the max of a convex function (autoconvolution is convex in x
    # for nonnegative x), we can use:
    # 1. Find the peak lag
    # 2. Adjust x to reduce that peak
    
    # Actually, let's try a different approach: iterative water-filling / balancing.
    # The idea: if the autoconvolution peak is at lag k, we want to reduce x[i] x[i+k]
    # for the pairs that contribute most to that peak.
    
    # Let's use a simple iterative method:
    # 1. Compute autoconvolution
    # 2. Find the peak
    # 3. For the peak lag k, reduce x[i] and x[i+k] proportionally to their product
    # 4. Renormalize
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Compute autoconvolution
        size = 1
        while size < 2 * N - 1:
            size *= 2
        X = np.fft.rfft(x, size)
        c = np.fft.irfft(X * X, size)[:2 * N - 1]
        
        # Find peak
        peak_idx = np.argmax(c)
        peak_val = c[peak_idx]
        
        if peak_val <= 0:
            break
        
        # The peak is at lag peak_idx. For lag k, the contribution is sum_i x[i] x[i+k]
        # We want to reduce this. A simple approach: for each i, reduce x[i] and x[i+k]
        # proportional to x[i] x[i+k].
        
        k = peak_idx
        if k == 0:
            # Peak at lag 0: sum x[i]^2. Reduce x[i] proportional to x[i]^2.
            # This is like minimizing sum x[i]^2 subject to sum x[i] = const,
            # which gives uniform x.
            x = np.ones(N)
        else:
            # For lag k, contributions are x[i] x[i+k] for i = 0 to N-1-k
            # and x[i+k] x[i] for i = 0 to N-1-k (but these are the same terms)
            # Actually, the autoconvolution at lag k is sum_{i=0}^{N-1-k} x[i] x[i+k]
            # for k > 0, and for k < 0 it's sum_{i=|k|}^{N-1} x[i] x[i-|k|]
            
            # Let's handle positive lags
            if k > 0:
                # Contributions: x[i] x[i+k] for i = 0, ..., N-1-k
                # Gradient with respect to x[i]:
                # d/dx[i] sum_j x[j] x[j+k] = x[i+k] (from term j=i) + x[i-k] (from term j=i-k, if valid)
                # So for i in [k, N-1]: gradient = x[i+k] + x[i-k]
                # For i in [0, k-1]: gradient = x[i+k]
                # For i in [N-k, N-1]: gradient = x[i-k]
                
                grad = np.zeros(N)
                for i in range(N - k):
                    grad[i] += x[i + k]
                    grad[i + k] += x[i]
                
                # Projected gradient step: x_new = max(0, x - alpha * grad)
                # Choose alpha to reduce the peak
                # Simple heuristic: alpha = peak_val / (max(grad) + 1e-10) * 0.5
                max_grad = np.max(grad)
                if max_grad > 0:
                    alpha = 0.5 * peak_val / (max_grad * N)
                    x = np.maximum(0, x - alpha * grad)
                
                # Renormalize to keep sum(x) constant
                s = np.sum(x)
                if s > 0:
                    x = x * (np.sum(best_x) / s)
            else:
                # Negative lag, handle similarly
                k_abs = -k
                grad = np.zeros(N)
                for i in range(N - k_abs):
                    grad[i] += x[i + k_abs]
                    grad[i + k_abs] += x[i]
                
                max_grad = np.max(grad)
                if max_grad > 0:
                    alpha = 0.5 * peak_val / (max_grad * N)
                    x = np.maximum(0, x - alpha * grad)
                
                s = np.sum(x)
                if s > 0:
                    x = x * (np.sum(best_x) / s)
        
        # Check improvement
        current_score = _score(x)
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
    
    # Final polish: coordinate descent
    for coord_pass in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Try to optimize x[i] while keeping others fixed
            # The autoconvolution is a quadratic in x[i]:
            # c[k] = sum_j x[j] x[j+k]
            # Terms involving x[i]:
            # - For k=0: x[i]^2
            # - For k>0: x[i] x[i+k] + x[i-k] x[i] (if valid)
            # So c is quadratic in x[i].
            
            # Find the optimal x[i] that minimizes the peak
            # This is complex. Let's just try a few values.
            old_val = x[i]
            # Try reducing and increasing
            for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                x[i] = old_val * factor
                s = np.sum(x)
                if s > 0:
                    # Renormalize
                    x = x * (np.sum(best_x) / s)
                
                sc = _score(x)
                if sc < best_score:
                    best_score = sc
                    best_x = x.copy()
                    improved = True
                else:
                    x[i] = old_val
        
        if not improved:
            break
    
    return list(best_x)
# EVOLVE-BLOCK-END
