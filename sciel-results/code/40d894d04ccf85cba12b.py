import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill
M_FFT = 2048  # Next power of 2 >= 2*600-1 = 1199

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _autoconv(heights):
    """Compute autoconvolution using FFT."""
    n = len(heights)
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    return conv

def _score_and_grad(heights):
    """Compute score and gradient of score w.r.t. heights using FFT.
    The score is scale-invariant, so we normalize heights to have sum=1 for gradient computation.
    Returns (score, grad) where grad is the gradient of score w.r.t. heights (before normalization).
    """
    heights = np.asarray(heights, dtype=np.float64)
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf, np.zeros(n)
    
    # Normalize to sum=1 for scale-invariant computation
    h_norm = heights / total
    w = 0.5 / n
    
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = h_norm
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    
    # Score for normalized heights: max_conv * n / (1 * 1 * w) = max_conv * n / w
    # But original score = max_conv_orig * n / (total^2 * w)
    # max_conv_orig = total^2 * max_conv (since h_orig = total * h_norm)
    # So score = total^2 * max_conv * n / (total^2 * w) = max_conv * n / w
    score = max_conv * n / w
    
    # Gradient: We need d(score)/d(heights)
    # score = max_conv * n / w where max_conv = max_k (h_norm * h_norm)[k]
    # Let k* be the argmax
    # d(max_conv)/d(h_norm[i]) = 2 * (h_norm * delta_{k*})[i] where delta_{k*} is the indicator of position k*
    # Actually, conv[i] = sum_j h_norm[j] * h_norm[i-j]
    # d(conv[k*])/d(h_norm[i]) = h_norm[i-k*] + h_norm[k*-i] (with appropriate indexing)
    
    # For linear convolution: conv[k] = sum_j h[j] * h[k-j]
    # d(conv[k*])/d(h[i]) = h[k*-i] + h[i-k*] (if k*-i >= 0 and i-k* >= 0)
    
    # Since h is defined on [0, n-1], and conv is on [0, 2n-2]
    # Let's compute the gradient properly
    
    # The convolution at lag k* is: sum_{j=max(0,k*-n+1)}^{min(n-1, k*)} h_norm[j] * h_norm[k*-j]
    # Gradient w.r.t. h_norm[i]:
    # If 0 <= i < n:
    #   term1: h_norm[k*-i] if 0 <= k*-i < n
    #   term2: h_norm[i-k*] if 0 <= i-k* < n
    
    k_star = np.argmax(conv)
    grad_conv = np.zeros(n)
    
    # h_norm[k*-i] for i in [0, n-1]
    idx1 = k_star - np.arange(n)
    valid1 = (idx1 >= 0) & (idx1 < n)
    grad_conv[valid1] += h_norm[idx1[valid1]]
    
    # h_norm[i-k*] for i in [0, n-1]
    idx2 = np.arange(n) - k_star
    valid2 = (idx2 >= 0) & (idx2 < n)
    grad_conv[valid2] += h_norm[idx2[valid2]]
    
    # d(score)/d(h_norm) = (n/w) * grad_conv
    grad_norm = (n / w) * grad_conv
    
    # Chain rule: h_norm = heights / total
    # d(h_norm[i])/d(heights[j]) = delta_{ij}/total - heights[i]*heights[j]/total^2
    # So d(score)/d(heights[j]) = sum_i d(score)/d(h_norm[i]) * (delta_{ij}/total - heights[i]*heights[j]/total^2)
    # = grad_norm[j]/total - heights[j]/total^2 * sum_i grad_norm[i] * heights[i]
    
    sum_grad_h = np.dot(grad_norm, heights)
    grad = grad_norm / total - heights / (total * total) * sum_grad_h
    
    return score, grad

def _smooth_max_surrogate(conv, beta):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    Returns (surrogate_value, gradient_wrt_conv)
    """
    # Numerical stability
    m = np.max(conv)
    shifted = conv - m
    exp_shifted = np.exp(shifted / beta)
    sum_exp = np.sum(exp_shifted)
    
    # surrogate = m + beta * log(sum_exp)
    surrogate = m + beta * np.log(sum_exp)
    
    # gradient w.r.t. conv[k] = exp_shifted[k] / sum_exp
    grad_conv = exp_shifted / sum_exp
    
    return surrogate, grad_conv

def _log_domain_obj_grad(x, beta=50.0):
    """
    Objective and gradient in log-domain: h = exp(x).
    We minimize the smooth-max surrogate of autoconvolution / (sum h)^2.
    This naturally enforces non-negativity.
    """
    h = np.exp(x)
    n = len(h)
    total = np.sum(h)
    
    if total <= 0:
        return np.inf, np.zeros_like(x)
    
    # Normalize for computation (score is scale-invariant, but we need to track it properly)
    # Actually, let's work with the actual h to get the correct gradient chain rule
    w = 0.5 / n
    
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = h
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    # Use smooth max surrogate
    surrogate, grad_conv = _smooth_max_surrogate(conv, beta)
    
    # Score = surrogate * n / (total^2 * w)
    score = surrogate * n / (total * total * w)
    
    # Gradient w.r.t. h:
    # d(score)/d(h[i]) = n/(total^2 * w) * d(surrogate)/d(h[i]) - 2*surrogate*n/(total^3 * w) * h[i]
    # d(surrogate)/d(h[i]) = sum_k grad_conv[k] * d(conv[k])/d(h[i])
    # d(conv[k])/d(h[i]) = h[k-i] (if 0<=k-i<n) + h[i-k] (if 0<=i-k<n)
    
    # Compute d(surrogate)/d(h)
    # term1[i] = sum_k grad_conv[k] * h[k-i] (valid k-i)
    # term2[i] = sum_k grad_conv[k] * h[i-k] (valid i-k)
    
    # This is a convolution-like operation. Let's compute it efficiently.
    # For each i, d(surrogate)/d(h[i]) = sum_{k: 0<=k-i<n} grad_conv[k]*h[k-i] + sum_{k: 0<=i-k<n} grad_conv[k]*h[i-k]
    
    # Let's compute this using FFT or direct summation. For N=600, direct is O(N^2) = 360k, which is fine.
    # But let's try to be smarter.
    
    # Actually, let's compute grad_h directly:
    # grad_h[i] = sum_k grad_conv[k] * (h[k-i] * I(0<=k-i<n) + h[i-k] * I(0<=i-k<n))
    
    # This can be written as:
    # grad_h = convolve(grad_conv, h) + convolve(grad_conv_reversed, h) ... not quite.
    
    # Let's just do it with numpy operations.
    # For efficiency, we can use the fact that this is a cross-correlation.
    
    # Let's compute it as:
    # A[i] = sum_j grad_conv[j] * h[j+i] (where j+i < n) -> this is like a shifted dot product
    # B[i] = sum_j grad_conv[j] * h[i+j] (where i+j < n)
    # Wait, d(conv[k])/d(h[i]) involves h[k-i] and h[i-k].
    
    # Let me just compute it directly for clarity and correctness.
    grad_h = np.zeros(n)
    
    # For each i, grad_h[i] = sum_{k=0}^{2n-2} grad_conv[k] * (h[k-i] if 0<=k-i<n else 0 + h[i-k] if 0<=i-k<n else 0)
    
    # This is O(N * 2N) = O(2N^2) = 720k operations, which is fine.
    
    # Vectorized version:
    # Create indices
    i_idx = np.arange(n)[:, None]  # shape (n, 1)
    k_idx = np.arange(2*n-1)[None, :]  # shape (1, 2n-1)
    
    # k - i
    diff1 = k_idx - i_idx  # shape (n, 2n-1)
    valid1 = (diff1 >= 0) & (diff1 < n)
    
    # i - k
    diff2 = i_idx - k_idx  # shape (n, 2n-1)
    valid2 = (diff2 >= 0) & (diff2 < n)
    
    # h[diff1] where valid
    # We need to gather h[diff1] for valid entries
    # This is tricky with numpy. Let's use a different approach.
    
    # Alternative: use FFT-based convolution
    # d(surrogate)/d(h) = cross-correlation of grad_conv with h
    # Specifically:
    # term1[i] = sum_k grad_conv[k] * h[k-i] = (h * grad_conv_flipped)[i] where * is convolution?
    # Let's think: (a * b)[i] = sum_j a[j] * b[i-j]
    # So term1[i] = sum_k grad_conv[k] * h[k-i] = sum_k grad_conv[k] * h[-(i-k)] = ...
    
    # Let's just use the direct method with advanced indexing.
    
    # For term1: grad_h[i] += sum_{k: 0<=k-i<n} grad_conv[k] * h[k-i]
    # For each i, the valid k are from i to i+n-1 (clamped to [0, 2n-2])
    
    # This is getting complex. Let's use a simpler but slightly slower method.
    # Since N=600, we can afford O(N^2) computation.
    
    # Let's compute grad_h using matrix operations:
    # Create a matrix M where M[i, k] = h[k-i] if 0<=k-i<n else 0
    # Then grad_h = M @ grad_conv
    
    # But M is sparse. Let's just loop over i for clarity.
    # Actually, for N=600, a Python loop would be too slow. Let's use numpy.
    
    # Here's a vectorized approach:
    # For each i in range(n):
    #   valid_k1 = k where 0 <= k-i < n -> k in [i, i+n-1] intersect [0, 2n-2]
    #   valid_k2 = k where 0 <= i-k < n -> k in [i-n+1, i] intersect [0, 2n-2]
    
    # Let's compute term1 and term2 separately.
    
    # term1[i] = sum_{k=max(0,i)}^{min(2n-2, i+n-1)} grad_conv[k] * h[k-i]
    # term2[i] = sum_{k=max(0,i-n+1)}^{min(2n-2, i)} grad_conv[k] * h[i-k]
    
    # Let's compute these using numpy slicing.
    
    # For term1:
    # For i=0: k from 0 to n-1, h[k]
    # For i=1: k from 1 to n, h[k-1]
    # ...
    # For i=n-1: k from n-1 to 2n-2, h[k-(n-1)]
    
    # This is like a Toeplitz matrix multiplication.
    
    # Let's just use the fact that we can compute this as a convolution.
    # term1[i] = sum_k grad_conv[k] * h[k-i]
    # Let j = k-i, then k = j+i
    # term1[i] = sum_j grad_conv[j+i] * h[j] (where j+i is valid)
    # = sum_j h[j] * grad_conv[j+i]
    
    # This is the cross-correlation of h and grad_conv, evaluated at lag i.
    # Specifically, term1 = correlate(h, grad_conv) but with specific indexing.
    
    # Let's use scipy.signal.correlate or just compute it with FFT.
    
    # Actually, let's just compute the gradient numerically for simplicity, or use a 
    # different approach: compute the gradient of the smooth surrogate w.r.t. h 
    # using the chain rule with FFT.
    
    # Alternative approach: 
    # d(surrogate)/d(h) = 2 * (h * (grad_conv as a filter))
    # where * denotes convolution, but we need to be careful with the indexing.
    
    # Let's just compute it directly using the definition.
    # For N=600, we can create the full Jacobian matrix and multiply.
    
    # Actually, let's use a simpler approach: compute the gradient using the 
    # relationship between the convolution and the input.
    
    # conv = ifft(fft(h)^2)
    # d(conv)/dh is a linear operator. The adjoint of this operator applied to 
    # grad_conv gives us d(surrogate)/dh.
    
    # The adjoint of the convolution operator C(h) = h*h is:
    # C^*(g) = 2 * (h * g) where * is convolution, but with proper handling.
    
    # Specifically, if conv = h * h (linear convolution), then:
    # <conv, g> = sum_k (h*h)[k] * g[k] = sum_i h[i] * sum_k h[k-i] * g[k]
    # = sum_i h[i] * (h * g)[i] (with proper indexing)
    
    # So C^*(g) = 2 * (h * g) where the convolution is appropriately defined.
    
    # Let's compute h * g using FFT.
    # g = grad_conv, which has length 2n-1.
    # h has length n.
    # We need the linear convolution of h and g.
    
    # But wait, the adjoint relationship is:
    # <C(h), g> = <h, C^*(g)>
    # where C(h) = h * h (autocorrelation/convolution with itself)
    
    # C^*(g) = 2 * (h * g) where h * g is the cross-correlation or convolution?
    
    # Let's derive it:
    # <h*h, g> = sum_k (sum_i h[i] h[k-i]) g[k]
    # = sum_i h[i] (sum_k h[k-i] g[k])
    # Let j = k-i, then k = j+i
    # = sum_i h[i] (sum_j h[j] g[j+i])
    # = sum_i h[i] (h * g)[i] where (h*g)[i] = sum_j h[j] g[j+i]
    
    # So C^*(g) = 2 * h * g where (h*g)[i] = sum_j h[j] g[j+i]
    
    # This is a cross-correlation. Let's compute it using FFT.
    
    # (h*g)[i] = sum_j h[j] g[j+i]
    # This is like a shifted dot product.
    
    # Let's compute it using FFT:
    # We can compute the cross-correlation using FFT.
    
    # Actually, let's just compute grad_h using the direct formula with numpy.
    
    # For efficiency, let's compute term1 and term2 using vectorized operations.
    
    # term1[i] = sum_{k} grad_conv[k] * h[k-i] for valid k-i
    # term2[i] = sum_{k} grad_conv[k] * h[i-k] for valid i-k
    
    # Let's compute term1:
    # For each i, term1[i] = sum_{j=0}^{n-1} h[j] * grad_conv[j+i] (where j+i < 2n-1)
    
    # We can create a matrix of grad_conv values for each shift.
    
    # Actually, let's just use the following approach:
    # Create a padded version of grad_conv and use matrix multiplication.
    
    # For now, let's compute the gradient numerically if the analytical one is too complex.
    # But that would be too slow.
    
    # Let's try a different approach: use the fact that the gradient of the 
    # smooth max surrogate w.r.t. h can be computed as:
    # grad_h = 2 * (h * p) where p = grad_conv and * is the appropriate convolution.
    
    # Let's compute h * p using FFT.
    # We need the cross-correlation: (h ⋆ p)[i] = sum_j h[j] p[j+i]
    
    # Using FFT:
    # We can compute this as the real part of ifft(fft(h) * conj(fft(p))) with appropriate shifts.
    
    # Actually, let's just compute it directly using numpy for now, since N=600 is small.
    
    # term1[i] = sum_j h[j] * grad_conv[j+i] for j=0..n-1 where j+i < 2n-1
    # term2[i] = sum_j h[j] * grad_conv[i-j] for j=0..n-1 where i-j >= 0 (i.e., j <= i)
    
    # Let's compute these using matrix operations.
    
    # Create a matrix P where P[i, j] = grad_conv[j+i] for term1
    # and Q where Q[i, j] = grad_conv[i-j] for term2
    
    # This is O(N^2) memory, which is 600*600 = 360k floats = 2.8 MB, fine.
    
    # For term1:
    # P[i, j] = grad_conv[j+i] if j+i < 2n-1 else 0
    i_vals = np.arange(n)
    j_vals = np.arange(n)
    
    # Create index matrix for term1: idx1 = j + i
    idx1_matrix = j_vals[None, :] + i_vals[:, None]  # shape (n, n)
    valid1_matrix = idx1_matrix < (2*n - 1)
    
    # Gather grad_conv values
    grad_conv_padded = np.zeros(2*n)  # pad to avoid index issues
    grad_conv_padded[:2*n-1] = grad_conv
    
    P = grad_conv_padded[idx1_matrix]  # shape (n, n)
    P[~valid1_matrix] = 0
    
    term1 = P @ h  # shape (n,)
    
    # For term2:
    # Q[i, j] = grad_conv[i-j] if i-j >= 0 else 0
    idx2_matrix = i_vals[:, None] - j_vals[None, :]  # shape (n, n)
    valid2_matrix = idx2_matrix >= 0
    
    Q = grad_conv_padded[idx2_matrix + (2*n-1)]  # shift to avoid negative indices
    # Actually, let's just use a different approach.
    
    # grad_conv[i-j] for i-j >= 0
    # Let's create a matrix where Q[i, j] = grad_conv[i-j] if i>=j else 0
    
    # Use advanced indexing
    idx2 = i_vals[:, None] - j_vals[None, :]
    valid2 = idx2 >= 0
    
    # Create a padded grad_conv with zeros at negative indices
    grad_conv_full = np.zeros(2*n - 1)
    grad_conv_full[:] = grad_conv
    
    # For valid2, we want grad_conv[idx2]
    # But idx2 can be negative, so we need to handle that.
    
    # Let's create a full grad_conv array that handles negative indices by returning 0.
    # We can do this by creating an array of size 4n and placing grad_conv in the middle.
    
    grad_conv_ext = np.zeros(4*n)
    offset = 2*n
    grad_conv_ext[offset:offset+2*n-1] = grad_conv
    
    idx2_ext = idx2 + offset
    Q = grad_conv_ext[idx2_ext]  # shape (n, n)
    Q[~valid2] = 0
    
    term2 = Q @ h  # shape (n,)
    
    # d(surrogate)/d(h) = term1 + term2
    d_surr_dh = term1 + term2
    
    # Now, d(score)/d(h) = n/(total^2 * w) * d_surr_dh - 2*score*h/total
    # score = surrogate * n / (total^2 * w)
    # d(score)/d(h[i]) = n/(total^2 * w) * d_surr_dh[i] - 2*surrogate*n/(total^3 * w) * h[i]
    # = n/(total^2 * w) * (d_surr_dh[i] - 2*surrogate*h[i]/total)
    
    d_score_dh = (n / (total * total * w)) * (d_surr_dh - 2 * surrogate * h / total)
    
    # Chain rule: h = exp(x), so dh/dx[i] = h[i]
    # d(score)/d(x[i]) = d(score)/d(h[i]) * h[i]
    grad_x = d_score_dh * h
    
    return score, grad_x

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0

    rng = np.random.default_rng(42)
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N, dtype=np.float64))
    
    # 2. Gaussian centered with various widths
    x = np.linspace(0, 1, N)
    for width in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
    
    # 3. Two blocks with various configurations
    for left_frac in [0.1, 0.2, 0.3, 0.4]:
        for gap_frac in [0.0, 0.05, 0.1, 0.15, 0.2]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                candidates.append(hb)
                
    # 4. Random sparse
    for _ in range(10):
        hr = np.zeros(N)
        mask = rng.random(N) < 0.5
        hr[mask] = 1.0
        candidates.append(hr)
        
    # 5. Cosine tapers
    for tf in [0.1, 0.2, 0.3, 0.5]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        candidates.append(ht)

    # 6. Smooth bumps
    for center in [0.2, 0.3, 0.5, 0.7, 0.8]:
        for width in [0.05, 0.1, 0.2]:
            hb = np.exp(-((x - center) / width) ** 2)
            candidates.append(hb)

    # Evaluate initial candidates
    for hc in candidates:
        if deadline_reached():
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()

    # Log-domain BFGS optimization from multiple starting points
    # We'll run BFGS from several different initial points
    
    num_starts = 6
    
    for start_idx in range(num_starts):
        if deadline_reached():
            break
        
        # Choose starting point
        if start_idx == 0:
            # Start from best found so far
            h_start = best_h.copy()
        elif start_idx < len(candidates):
            h_start = candidates[start_idx].copy()
        else:
            # Random start
            h_start = np.maximum(np.ones(N) + rng.normal(0, 0.5, N), 0.0)
        
        h_start = np.maximum(h_start, 1e-10)  # Ensure positive for log
        
        # Convert to log domain
        x_start = np.log(h_start)
        
        # Run BFGS with smooth max surrogate
        try:
            result = minimize(
                _log_domain_obj_grad,
                x_start,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            if result.success or result.fun < best_score:
                x_opt = result.x
                h_opt = np.exp(x_opt)
                h_opt = np.maximum(h_opt, 0.0)
                
                sc = compute_score(h_opt)
                if sc < best_score:
                    best_score = sc
                    best_h = h_opt.copy()
                    
        except Exception as e:
            pass
    
    # Final polish: Gradient-based local optimization around best_h using the original gradient
    if time_left() > 10:
        current_h = best_h.copy()
        current_score = best_score
        
        # Gradient descent with line search
        for _ in range(2000):
            if deadline_reached():
                break
            
            score, grad = _score_and_grad(current_h)
            if score < best_score:
                best_score = score
                best_h = current_h.copy()
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                break
            
            # Normalize gradient
            grad = grad / grad_norm
            
            # Backtracking line search
            step_size = 0.1
            for _ in range(20):
                new_h = current_h - step_size * grad
                new_h = np.maximum(new_h, 0.0)
                new_score = compute_score(new_h)
                
                if new_score < current_score:
                    current_h = new_h
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
                    break
                else:
                    step_size *= 0.5
            
            # If we couldn't find a descent step, try random perturbation
            if step_size < 1e-6:
                noise = rng.normal(0, 0.01, N)
                new_h = np.maximum(current_h + noise, 0.0)
                new_score = compute_score(new_h)
                if new_score < current_score:
                    current_h = new_h
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
