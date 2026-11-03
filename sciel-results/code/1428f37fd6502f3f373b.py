"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Direct optimization of the 600-bin vector using FFT-accelerated convolution.
2. Use a projected subgradient method (or quasi-Newton if available, but subgradient is robust)
   to minimize the maximum of the autocorrelation.
3. Use a log-sum-exp smooth approximation of the max for better gradients.
4. Multiple restarts with different initial shapes (Box, Gaussian, Cosine, Random).
5. Deadline-guarded.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill
X_GRID = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for speed
    # Pad to avoid circular convolution artifacts
    size = 2 * N
    F = np.fft.rfft(f, size)
    C = np.fft.irfft(F * F.conjugate(), size)
    # The linear convolution is in the first N + N - 1 elements, but due to padding it's spread.
    # Actually, for linear conv of two length N vectors, result length 2N-1.
    # np.fft.rfft with size=2N gives circular conv of length 2N.
    # Since support of linear conv is 2N-1 < 2N, the first 2N-1 elements are correct, 
    # and the last one is 0 (or wrap-around if not padded enough, but 2N is enough for linear conv of two N-vectors).
    # Wait, linear conv length is 2N-1. Circular conv length 2N.
    # The linear conv is embedded in the circular conv.
    # Elements 0 to 2N-2 are the linear conv. Element 2N-1 is 0.
    max_c = np.max(C)
    return max_c / (s * s)

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x - x_max
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max + log_sum

def get_gradient(f, beta=10.0):
    """
    Compute the gradient of the smooth max of the autocorrelation.
    Returns the gradient vector of length N.
    """
    s = np.sum(f)
    if s < 1e-15:
        return np.zeros(N)
    
    size = 2 * N
    F = np.fft.rfft(f, size)
    C = np.fft.irfft(F * F.conjugate(), size)
    
    # Normalize C by s^2 for the score gradient?
    # Score = max(C) / s^2.
    # Let's optimize the score directly.
    # d/d f_j ( max(C) / s^2 )
    # This is complex. 
    # Alternative: Normalize f to sum=1 at each step.
    # Then Score = max(C_norm) where C_norm is autocorr of normalized f.
    # If we fix sum(f)=1, then s=1. Score = max(C).
    # Gradient of max(C) w.r.t f_j is C_{m-j} at the peak m.
    # For smooth max: 
    # S = (1/beta) log(sum(exp(beta * C_k)))
    # dS/df_j = sum_k (exp(beta*C_k)/sum(exp(beta*C_l))) * dC_k/df_j
    # dC_k/df_j = f_{k-j} (for linear conv, with appropriate indexing)
    
    # Weights for the soft max
    C_max = np.max(C)
    weights = np.exp(beta * (C - C_max))
    weights /= np.sum(weights)
    
    # Gradient of C w.r.t f
    # C_k = sum_i f_i f_{k-i}
    # dC_k/df_j = f_{k-j} + f_{k-j} ? No, derivative of f_i f_{k-i} w.r.t f_j:
    # If i=j, term is f_{k-j}. If k-i=j => i=k-j, term is f_j.
    # Since f is real, C is symmetric? No, C is autocorrelation, so C_k = C_{-k}.
    # In linear conv indexing:
    # C_m = sum_{j} f_j f_{m-j}
    # dC_m/df_j = f_{m-j} + f_{m-j} = 2 f_{m-j}? 
    # Wait. C_m = sum_i f_i f_{m-i}.
    # d/d f_j ( sum_i f_i f_{m-i} )
    # Term i=j: f_j f_{m-j} -> derivative f_{m-j}
    # Term i=m-j: f_{m-j} f_j -> derivative f_j
    # If j != m-j, both terms exist.
    # If j = m-j (i.e., 2j=m), there is only one term in the sum? No, the sum is over i.
    # If we differentiate w.r.t f_j, we hit the term where i=j and the term where m-i=j.
    # So dC_m/df_j = f_{m-j} + f_{m-j} = 2 f_{m-j}.
    # EXCEPT if we are dealing with the diagonal? No, it's always 2 f_{m-j} for linear convolution of f with itself.
    # Wait, is it?
    # Let f = [a, b]. C = [a^2, ab+ba, b^2] = [a^2, 2ab, b^2].
    # dC_0/da = 2a. Formula 2 f_{0-0} = 2 f_0 = 2a. Correct.
    # dC_1/da = 2b. Formula 2 f_{1-0} = 2 f_1 = 2b. Correct.
    # dC_1/db = 2a. Formula 2 f_{1-1} = 2 f_0 = 2a. Correct.
    # So yes, gradient of autocorrelation peak w.r.t f_j is 2 f_{m-j}.
    
    # We need to compute G_j = sum_m weights_m * 2 f_{m-j}
    # This is a convolution of weights with f?
    # G_j = 2 sum_m weights_m f_{m-j}
    # Let W = weights. This is W * f?
    # (W * f)_j = sum_m W_m f_{j-m}.
    # We have f_{m-j} = f_{-(j-m)}.
    # So G_j = 2 (W * f_{reversed})_j?
    
    # Simpler: Just compute it directly for the peak indices if the support of weights is small?
    # With log-sum-exp, weights are spread out.
    # But N=600. We can do an O(N^2) operation once per iteration if it's fast enough in numpy.
    # Or use FFT.
    # G = 2 * Convolve(weights, f) with appropriate indexing.
    # Let's just do the O(N^2) loop in numpy vectorized?
    # G = 2 * np.array([np.sum(weights * np.roll(f, m)) for m in range(N)])? No.
    
    # Let's use the FFT property.
    # G_j = 2 sum_m W_m f_{m-j}
    # This is the cross-correlation of W and f.
    # Cross-correlation can be done via FFT.
    # corr(W, f)_j = sum_m W_m f_{m-j}
    # np.correlate(W, f, mode='same')?
    # Let's check dimensions.
    # W is length 2N (from C). f is length N.
    # We need the result to be length N.
    
    # Actually, since C is computed via FFT, we can compute the gradient via FFT too.
    # But the weights depend on C.
    
    # Let's try a simpler approach:
    # The gradient is dominated by the peaks.
    # If we use a hard max, the gradient is just 2 f_{m*-j}.
    # This is sparse if we only take the top few peaks.
    # Let's use the hard max gradient for speed and robustness, with a small step size.
    
    m_star = np.argmax(C)
    # Gradient of max(C) at m_star is 2 f_{m_star - j}
    # We need to map m_star - j to valid indices in f.
    # f is defined on 0..N-1.
    # The convolution C is linear.
    # C_m = sum_{i=0}^{N-1} f_i f_{m-i} (where f_k=0 if k<0 or k>=N)
    # dC_m/df_j = 2 f_{m-j}
    
    # Construct the gradient vector
    grad = np.zeros(N)
    # m_star is in [0, 2N-2]
    # For each j in 0..N-1:
    # grad[j] = 2 * f[m_star - j] if 0 <= m_star - j < N else 0
    
    # Vectorized:
    # indices = m_star - np.arange(N)
    # valid = (indices >= 0) & (indices < N)
    # grad[valid] = 2 * f[indices[valid]]
    
    indices = m_star - np.arange(N)
    valid = (indices >= 0) & (indices < N)
    grad[valid] = 2.0 * f[indices[valid]]
    
    return grad

def optimize_direct(f_init, start_time, max_time, beta=10.0, lr_init=0.01):
    """
    Optimize f using projected subgradient descent on the bins.
    """
    f = f_init.copy()
    s = np.sum(f)
    if s > 0:
        f = f / s # Normalize to sum 1
    
    best_f = f.copy()
    best_score = compute_score(f)
    
    lr = lr_init
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        # Compute gradient
        grad = get_gradient(f, beta)
        
        # Update
        f_new = f - lr * grad
        
        # Project to non-negative
        f_new = np.maximum(f_new, 0.0)
        
        # Normalize to sum 1
        s_new = np.sum(f_new)
        if s_new < 1e-15:
            break
        f_new = f_new / s_new
        
        # Evaluate
        score_new = compute_score(f_new)
        
        if score_new < best_score:
            best_score = score_new
            best_f = f_new.copy()
            no_improve_count = 0
            # Increase learning rate slightly if improving
            lr = min(lr * 1.05, 0.1)
        else:
            no_improve_count += 1
            lr *= 0.9
            if no_improve_count > 50:
                break
        
        f = f_new
        
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Constant (Box)
    f_box = np.ones(N)
    candidates.append(f_box)
    
    # 2. Gaussian
    x = X_GRID
    sigma = 0.1
    f_gauss = np.exp(-0.5 * (x / sigma) ** 2)
    candidates.append(f_gauss)
    
    # 3. Cosine bump
    f_cos = np.zeros(N)
    mask = np.abs(x) <= 0.25
    f_cos[mask] = 0.5 * (1 + np.cos(2 * np.pi * x[mask] / 0.5))
    candidates.append(f_cos)
    
    # 4. Random smooth
    rng = np.random.RandomState(42)
    coeffs = rng.normal(size=10)
    f_rand = np.zeros(N)
    for k, c in enumerate(coeffs, start=1):
        f_rand += c * np.cos(2 * np.pi * k * (x + 0.25) / 0.5)
    f_rand = np.abs(f_rand) + 0.1
    candidates.append(f_rand)
    
    best_f = candidates[0]
    best_score = compute_score(best_f)
    
    # Optimize each candidate
    for i, f_init in enumerate(candidates):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Allocate time proportionally
        remaining = TIME_LIMIT - (time.time() - start_time)
        time_per_candidate = remaining / (len(candidates) - i)
        
        f_opt, score_opt = optimize_direct(f_init, start_time, time_per_candidate, lr_init=0.05)
        
        if score_opt < best_score:
            best_score = score_opt
            best_f = f_opt
    
    # Final polish with very small steps on the best found
    if time.time() - start_time < TIME_LIMIT - 2:
        f_polished, score_polished = optimize_direct(best_f, start_time, TIME_LIMIT - 1, lr_init=0.001)
        if score_polished < best_score:
            best_score = score_polished
            best_f = f_polished

    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
