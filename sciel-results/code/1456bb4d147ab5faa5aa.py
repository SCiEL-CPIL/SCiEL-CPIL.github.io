"""Optimized program for the first autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
# Interval width: total range is 1/2, so each interval is 1/(2*N)
dx = 1.0 / (2.0 * N)


def autoconvolve(h):
    """Compute autoconvolution of step function heights on equal intervals.
    
    For a step function f(t) = h[i] on interval i, the autoconvolution
    (f*f)(s) = integral f(t) f(s-t) dt.
    
    For discrete steps on [-1/4, 1/4] with N intervals of width dx:
    (f*f)[k] = sum_i h[i] * h[j] * (overlap_length) where the overlap
    depends on the shift.
    
    More precisely, if we index intervals 0..N-1 corresponding to
    [-1/4 + i*dx, -1/4 + (i+1)*dx], then the autoconvolution at shift
    corresponding to index k (shift = (k - N/2)*dx approximately) is:
    
    (f*f)[k] = sum_i h[i] * h[i + k - N//2] * dx * (some overlap factor)
    
    Actually, for a step function, the autoconvolution is:
    (f*f)(s) = sum_{i,j} h[i]*h[j] * length of {t: t in interval_i, s-t in interval_j}
    
    This equals sum_{i,j} h[i]*h[j] * max(0, min(right_i, right_j + s) - max(left_i, left_j + s))
    
    For equal-width intervals, this simplifies. Let's use FFT-based convolution.
    """
    # Pad to avoid circular convolution effects
    n_fft = 2 * N + 1
    F = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)
    # The autoconvolution of a function supported on [-1/4, 1/4] 
    # is supported on [-1/2, 1/2].
    # Our grid for the autoconvolution: shifts from -1/2 to 1/2
    # The FFT result gives circular convolution; we need to unwrap.
    # For linear autoconvolution of two sequences of length N:
    # result has length 2*N - 1
    # Let's just do direct linear convolution via FFT properly
    F = np.fft.rfft(h, n=2 * N - 1)
    conv = np.fft.irfft(F * F, n=2 * N - 1)
    return conv


def normalized_max_autoconv(h):
    """Compute the normalized maximum of the autoconvolution.
    
    The autoconvolution (f*f)(s) = integral f(t)f(s-t) dt.
    For a step function with heights h[i] on intervals of width dx:
    (f*f) is a piecewise linear function. Its maximum occurs at some point.
    
    The normalization: we want max_s (f*f)(s) / (integral f)^2 or similar.
    Actually, the task says "normalized maximum of its autoconvolution".
    
    A common normalization: max (f*f) / (integral f * integral f) = max (f*f) / (||f||_1^2)
    
    For the constant function f=1 on [-1/4,1/4]:
    integral f = 1/2
    (f*f)(s) = length of [-1/4,1/4] ∩ [s-1/4, s+1/4] = 1/2 - |s| for |s| <= 1/2
    max (f*f) = 1/2 at s=0
    normalized max = (1/2) / (1/2)^2 = 2
    
    But the seed gives score 1.0 for constant heights, so the normalization must be different.
    
    Let me reconsider. Perhaps it's max (f*f) / max possible, or perhaps the 
    "normalized" refers to something specific to the C1 constant.
    
    Looking at the seed: constant heights give score 1.0. Let's compute what 
    normalization gives 1.0 for constant.
    
    For constant h = 1:
    autoconv[k] via linear convolution of [1]*N with itself:
    conv[k] = min(k+1, N, 2*N-1-k) for k=0..2*N-2
    max conv = N (at k = N-1)
    
    If normalized max = max(conv) / N = N/N = 1. That works!
    
    So normalized max = max(autoconv) / N, where autoconv is the discrete 
    autoconvolution (sum of products, not integrated).
    
    Actually wait - for a step function, the actual integral autoconvolution at 
    the center is: sum_i h[i]^2 * dx = N * 1 * dx = N * (1/(2N)) = 1/2.
    
    The discrete convolution max is N. So normalized max = N/N = 1.
    
    Let me verify: the "normalized maximum" is likely max(discrete_autoconv) / N.
    """
    conv = autoconvolve(h)
    # The discrete autoconvolution: conv[k] = sum_i h[i]*h[i+k] (with appropriate indexing)
    # For constant h=1, conv has max = N
    # Normalized max = max(conv) / N
    return np.max(conv) / N


def construct():
    """Construct a nonnegative step function with N intervals to minimize 
    the normalized maximum of its autoconvolution."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with a reasonable initial guess
    # A cosine bump might be better than constant
    x = np.linspace(0, 1, N)
    h0 = np.maximum(0.0, np.cos(np.pi * (x - 0.5)))
    # Normalize so that it's nonnegative and has reasonable scale
    # The objective is scale-invariant in a sense? Let's check.
    # If we scale h by c, autoconv scales by c^2, so normalized max scales by c^2.
    # So we should fix the scale. Let's fix sum(h) = 1 or similar.
    
    # Actually, since the objective is max(conv)/N and conv scales as h^2,
    # we need to fix the scale. Let's fix the L1 norm: sum(h)*dx = 1, so sum(h) = 2N.
    # Or simpler: fix sum(h) = N (so average height is 1, like the seed).
    
    # Let's fix sum(h) = N to keep comparable to seed
    h0 = h0 / np.sum(h0) * N
    
    # Use a log-sum-exp based smooth surrogate for the max
    def objective(params):
        """Smooth surrogate for max(conv)/N using log-sum-exp."""
        h = np.maximum(0.0, params)  # Enforce nonnegativity
        # Fix scale: normalize so sum(h) = N
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h = h / s * N
        
        conv = autoconvolve(h)
        # Smooth max via log-sum-exp
        beta = 50.0
        m = np.max(conv)
        # For numerical stability
        shifted = conv / m
        log_sum_exp = np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta
        smooth_max = m * (1.0 + log_sum_exp)  # This is not quite right
        # Actually: log-sum-exp of conv / N
        # log(sum(exp(beta * conv / N))) / beta * (1/N) ... let me redo
        # smooth_max = (1/N) * log(sum(exp(beta * conv))) / beta
        # But this can overflow. Use the standard trick:
        max_conv = np.max(conv)
        if max_conv == 0:
            return 0.0
        exp_terms = np.exp(beta * (conv - max_conv) / N)
        smooth_max = (max_conv + np.log(np.sum(exp_terms)) / beta) / N
        return smooth_max
    
    # Gradient of the objective (numerical or analytical)
    def objective_and_grad(params):
        h = np.maximum(0.0, params)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10, np.zeros_like(params)
        h = h / s * N
        
        conv = autoconvolve(h)
        beta = 50.0
        max_conv = np.max(conv)
        if max_conv == 0:
            return 0.0, np.zeros_like(params)
        
        exp_terms = np.exp(beta * (conv - max_conv) / N)
        smooth_max = (max_conv + np.log(np.sum(exp_terms)) / beta) / N
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(params)
        for i in range(N):
            params_plus = params.copy()
            params_plus[i] += eps
            grad[i] = (objective(params_plus) - smooth_max) / eps
        
        return smooth_max, grad
    
    # Try multiple initializations and local optimizations
    best_h = h0.copy()
    best_score = normalized_max_autoconv(best_h)
    
    # Several initial guesses
    initializations = [
        h0.copy(),
        np.ones(N),
        np.maximum(0.0, np.cos(np.pi * (x - 0.5))) / np.sum(np.maximum(0.0, np.cos(np.pi * (x - 0.5)))) * N,
        np.maximum(0.0, 1.0 - 4 * (x - 0.5)**2) / np.sum(np.maximum(0.0, 1.0 - 4 * (x - 0.5)**2)) * N,
        # Triangular
        np.abs(1.0 - 2.0 * x) / np.sum(np.abs(1.0 - 2.0 * x)) * N,
        # Two bumps
        np.maximum(0.0, np.cos(2 * np.pi * (x - 0.25))) + np.maximum(0.0, np.cos(2 * np.pi * (x - 0.75))),
    ]
    
    for h_init in initializations:
        if time.time() > deadline - 20:
            break
        
        h_init = np.maximum(0.0, h_init)
        s = np.sum(h_init)
        if s < 1e-10:
            continue
        h_init = h_init / s * N
        
        # Local optimization using L-BFGS-B
        try:
            result = minimize(
                objective,
                h_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
                bounds=[(0, None)] * N
            )
            
            h_opt = np.maximum(0.0, result.x)
            s = np.sum(h_opt)
            if s > 1e-10:
                h_opt = h_opt / s * N
                score = normalized_max_autoconv(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Additional refinement: local peak reduction
    # Find where the autoconvolution peaks and try to reduce those peaks
    for iteration in range(50):
        if time.time() > deadline - 10:
            break
        
        h = best_h.copy()
        conv = autoconvolve(h)
        # Find peak locations in the autoconvolution
        peak_idx = np.argmax(conv)
        
        # The peak at index k in the autoconvolution corresponds to
        # contributions from pairs (i, j) where i + j = k (in the convolution sense)
        # For linear convolution: conv[k] = sum_i h[i] * h[k-i] (with valid range)
        
        # Try to reduce the peak by lowering heights at indices that contribute
        # most to the peak
        k = peak_idx
        # Indices contributing to conv[k]: i from max(0, k-N+1) to min(N-1, k)
        i_min = max(0, k - N + 1)
        i_max = min(N - 1, k)
        
        if i_max < i_min:
            break
        
        # Contribution of each pair
        contributions = np.array([h[i] * h[k - i] for i in range(i_min, i_max + 1)])
        if np.sum(contributions) < 1e-15:
            break
        
        # Find the index with the largest contribution
        max_contrib_idx = np.argmax(contributions)
        i_star = i_min + max_contrib_idx
        j_star = k - i_star
        
        # Try reducing h[i_star] or h[j_star] slightly
        # Use a small step
        step = 0.01 * np.mean(h)
        
        # Try both options
        h_test1 = h.copy()
        h_test1[i_star] = max(0, h_test1[i_star] - step)
        s1 = np.sum(h_test1)
        if s1 > 1e-10:
            h_test1 = h_test1 / s1 * N
            score1 = normalized_max_autoconv(h_test1)
        
        h_test2 = h.copy()
        h_test2[j_star] = max(0, h_test2[j_star] - step)
        s2 = np.sum(h_test2)
        if s2 > 1e-10:
            h_test2 = h_test2 / s2 * N
            score2 = normalized_max_autoconv(h_test2)
        
        # Also try increasing the minimum to redistribute
        h_test3 = h.copy()
        min_idx = np.argmin(h)
        h_test3[min_idx] += step
        s3 = np.sum(h_test3)
        h_test3 = h_test3 / s3 * N
        score3 = normalized_max_autoconv(h_test3)
        
        best_test_score = min(score1, score2, score3)
        if best_test_score < best_score:
            if best_test_score == score1:
                best_h = h_test1
            elif best_test_score == score2:
                best_h = h_test2
            else:
                best_h = h_test3
            best_score = best_test_score
        else:
            break
    
    # Final: ensure nonnegativity and return
    best_h = np.maximum(0.0, best_h)
    return list(best_h)


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"Min: {min(h)}, Max: {max(h)}")
    print(f"Score: {normalized_max_autoconv(np.array(h))}")
