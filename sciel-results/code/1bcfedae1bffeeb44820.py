"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _soft_max_abs(x, temp=0.01):
    """
    Differentiable approximation of max(|x|).
    Uses the log-sum-exp trick for stability.
    """
    abs_x = np.abs(x)
    max_val = np.max(abs_x)
    if max_val < 1e-12:
        return 0.0
    # Shift to prevent overflow in exp
    shifted = abs_x - max_val
    exp_sum = np.sum(np.exp(shifted / temp))
    return max_val + temp * np.log(exp_sum)

def _gradient_step(x, fx, size, n, lr):
    """
    Compute gradient of soft_max(|autoconv|) w.r.t x.
    autoconv = x * x (linear)
    c_k = sum x_i x_{k-i}
    d c_k / d x_j = x_{k-j} + x_{j-k} (if within bounds)
    
    We want to minimize sum_{k} soft_max(|c_k|) approx.
    Actually, we just want to minimize the peak.
    Let M = max |c_k|.
    Gradient of soft_max(|c|) w.r.t c is:
    d/dc_k = sign(c_k) * exp(|c_k|/T) / sum_j exp(|c_j|/T)
    
    Then chain rule to x.
    This is expensive to compute exactly. 
    Alternative: Use the gradient of the actual max if we identify the peak indices, 
    or use a simpler heuristic: 
    Move x in the direction that reduces the largest peaks.
    
    Let's use a simpler, robust approach: 
    1. Compute autoconv c.
    2. Find indices k where |c_k| is close to max.
    3. Compute gradient of |c_k| w.r.t x.
    4. Update x.
    """
    # Compute autoconv
    conv = np.fft.irfft(fx * fx, size)
    conv = conv[:2 * n - 1]
    
    # Find peak indices
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    
    # Identify indices contributing to the peak (soft mask)
    # Use a threshold to include top few peaks
    threshold = max_val * 0.9
    peak_mask = abs_conv >= threshold
    
    if not np.any(peak_mask):
        return x, 0.0
        
    # Gradient of sum_{k in peaks} |c_k| w.r.t x
    # c_k = sum_i x_i x_{k-i}
    # d c_k / d x_j = x_{k-j} + x_{j-k} (with boundary checks)
    
    grad = np.zeros(n)
    
    # For each peak k, add contribution to grad
    # This loop is O(N * N_peaks). N=400, N_peaks is small.
    peak_indices = np.where(peak_mask)[0]
    
    for k in peak_indices:
        # Contribution of x_j to c_k
        # c_k = sum_{i=0}^{N-1} x_i x_{k-i}
        # Term with x_j: x_j x_{k-j} (if 0 <= k-j < N)
        # Term with x_{k-j}: x_{k-j} x_j (if 0 <= j < N, which is true)
        # So d c_k / d x_j = x_{k-j} + x_{j-k} ? No.
        # c_k = x_0 x_k + x_1 x_{k-1} + ... + x_j x_{k-j} + ... + x_k x_0
        # d/dx_j (x_j x_{k-j}) = x_{k-j}
        # d/dx_j (x_{k-j} x_j) = x_{k-j}
        # So if k-j is in [0, N-1], contribution is x_{k-j} + x_{k-j} = 2*x_{k-j}? 
        # Wait, if i=j, term is x_j x_{k-j}. Deriv is x_{k-j}.
        # If i=k-j, term is x_{k-j} x_j. Deriv is x_j.
        # These are the same term if j = k-j => 2j=k.
        # Generally, d c_k / d x_j = x_{k-j} (if valid) + x_{j-k} (if valid, which is same index if we consider symmetry? No.)
        
        # Let's just compute the gradient numerically or via FFT?
        # FFT gradient:
        # C = X^2
        # dC/dX = 2X
        # dC/dx = IFFT(2X) * x ? No.
        # C_k = sum x_i x_{k-i}
        # This is linear convolution.
        # Gradient of C_k w.r.t x is a circulant-like matrix.
        
        # Simpler: Just use the fact that we want to reduce |c_k|.
        # If c_k > 0, we want to decrease c_k.
        # If c_k < 0, we want to increase c_k.
        # d c_k / d x_j = x_{k-j} (if 0<=k-j<N) + x_{j-k} (if 0<=j-k<N)
        # Note: x_{j-k} is valid if j-k >= 0 and < N.
        
        # Let's compute the full gradient vector for the peak k.
        # g_j = d c_k / d x_j
        # g_j = x_{k-j} (if 0 <= k-j < N) + x_{j-k} (if 0 <= j-k < N)
        
        # We can compute this efficiently.
        # term1: x shifted by k.
        # term2: x shifted by -k.
        
        # Create a zero array of size 2N
        # x is length N.
        
        # Let's just do it directly for simplicity since N=400 is small.
        # But we have multiple peaks.
        
        pass # We will implement a vectorized version below.
        
    # Vectorized Gradient Calculation for all peaks at once is hard.
    # Let's use a simpler heuristic: 
    # Randomly pick one peak index k_peak.
    # Compute grad of |c_k_peak| w.r.t x.
    # Update x.
    
    k_peak = peak_indices[np.argmax(abs_conv[peak_indices])]
    
    # Compute d c_k_peak / d x
    g = np.zeros(n)
    # c_k = sum_i x_i x_{k-i}
    # For each j in 0..N-1:
    # if 0 <= k-j < N: g[j] += x[k-j]
    # if 0 <= j-k < N: g[j] += x[j-k]
    
    # Range of i such that k-i is in [0, N-1]
    # i_min = max(0, k - (N-1))
    # i_max = min(N-1, k)
    
    # Contribution from x_i x_{k-i} where i=j: term is x_{k-j}. Valid if 0 <= k-j < N.
    # Contribution from x_{k-i} x_i where k-i=j => i=k-j. Term is x_j. Valid if 0 <= k-j < N.
    # So for each j:
    # if 0 <= k-j < N: g[j] += x[k-j] + x[j]
    # if 0 <= j-k < N: g[j] += x[j-k] + x[k-j] ? No.
    
    # Let's re-derive.
    # c_k = sum_{i=0}^{N-1} x_i x_{k-i}
    # d c_k / d x_j:
    # Only terms involving x_j contribute.
    # Term 1: i=j. x_j x_{k-j}. Deriv: x_{k-j}. Valid if 0 <= k-j < N.
    # Term 2: k-i=j => i=k-j. x_{k-j} x_j. Deriv: x_j. Valid if 0 <= k-j < N.
    # So if 0 <= k-j < N, g[j] = x_{k-j} + x_j.
    # If k-j < 0 or k-j >= N, the term doesn't exist.
    # What if i != j and k-i != j? Then x_j is not in the term.
    # So g[j] is non-zero only if 0 <= k-j < N.
    # In that case, g[j] = x_{k-j} + x_j.
    
    # Wait, is that correct?
    # c_k = ... + x_j x_{k-j} + ... + x_{k-j} x_j + ...
    # If j != k-j, these are two distinct terms in the sum.
    # d/dx_j (x_j x_{k-j}) = x_{k-j}
    # d/dx_j (x_{k-j} x_j) = x_{k-j}
    # So yes, g[j] = 2 * x_{k-j} if j != k-j? No.
    # d/dx_j (x_j x_{k-j}) = x_{k-j}.
    # d/dx_j (x_{k-j} x_j) = x_{k-j}.
    # Sum = 2 * x_{k-j}.
    
    # But if j = k-j (i.e., k=2j), then it's the same term x_j^2.
    # d/dx_j (x_j^2) = 2 x_j.
    # So g[j] = 2 x_j.
    
    # Let's check the condition 0 <= k-j < N.
    
    valid_j = np.arange(n)
    # Condition: 0 <= k - j < N
    # j <= k and j > k - N
    j_min = max(0, k_peak - (N - 1))
    j_max = min(N - 1, k_peak)
    
    if j_min <= j_max:
        js = np.arange(j_min, j_max + 1)
        # g[js] = 2 * x[k_peak - js]
        # Check if k_peak - js == js for any?
        # If k_peak - j == j => k_peak = 2j.
        # In that case, the term is x_j^2, deriv is 2x_j.
        # The formula 2 * x[k_peak - j] gives 2 * x[j]. Same.
        g[js] = 2.0 * x[k_peak - js]
        
    # Now we have gradient of c_k_peak w.r.t x.
    # We want to minimize |c_k_peak|.
    # If c_k_peak > 0, we want to decrease c_k_peak. Update: x -= lr * grad.
    # If c_k_peak < 0, we want to increase c_k_peak. Update: x += lr * grad.
    # So x -= lr * sign(c_k_peak) * grad.
    
    sign_c = np.sign(conv[k_peak])
    x_new = x - lr * sign_c * g
    
    return x_new, conv[k_peak]

def construct():
    deadline = time.time() + 140.0
    
    rng = np.random.default_rng(2024)
    
    # Initialize with random noise
    x = rng.normal(0, 1, N)
    
    # Normalize sum to 1
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x / s
    
    best_x = x.copy()
    best_val = _metric(best_x)
    
    size = 1
    while size < 2 * N - 1:
        size *= 2
        
    # Adam Optimizer State
    m = np.zeros(N)
    v = np.zeros(N)
    t = 0
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 0.1
    
    # We will use a simple gradient step on the peak autoconvolution
    # To avoid getting stuck in local minima of the peak, we can occasionally
    # randomize the starting point or use a larger step.
    
    no_improve = 0
    
    while time.time() < deadline:
        # Compute FFT
        fx = np.fft.rfft(x, size)
        
        # Get gradient step
        x_new, peak_val = _gradient_step(x, fx, size, N, lr)
        
        # Normalize to maintain sum=1 constraint
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            # Fallback to random restart if normalization fails
            x = rng.normal(0, 1, N)
            s = np.sum(x)
            x = x / s
            continue
            
        x_new = x_new / s_new
        
        # Update Adam moments
        t += 1
        m = beta1 * m + (1 - beta1) * (x_new - x)
        v = beta2 * v + (1 - beta2) * (x_new - x)**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Adam update
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Re-normalize after Adam step
        s = np.sum(x)
        if abs(s) > 1e-12:
            x = x / s
        else:
            x = rng.normal(0, 1, N)
            x = x / np.sum(x)
            
        # Evaluate
        val = _metric(x)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
            no_improve = 0
            # Increase lr slightly if improving? Or keep constant.
        else:
            no_improve += 1
            if no_improve > 100:
                # Reduce lr
                lr *= 0.9
                no_improve = 0
                if lr < 1e-6:
                    # Random restart
                    x = rng.normal(0, 1, N)
                    s = np.sum(x)
                    x = x / s
                    lr = 0.1
                    m = np.zeros(N)
                    v = np.zeros(N)
                    t = 0

    return best_x
# EVOLVE-BLOCK-END
