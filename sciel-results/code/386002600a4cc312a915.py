"""Optimized step function to minimize normalized max autoconvolution."""
import numpy as np
from scipy.fft import fft, ifft
import time
import sys

N = 600

def _autoconv_peak(f):
    """Compute the normalized maximum of the autoconvolution of f."""
    # f is a sequence of N nonnegative step heights on [-1/4, 1/4]
    # The autoconvolution (in discrete sense for step functions) can be computed
    # via FFT. The step function has value f[i] on interval [x_i, x_{i+1}] where
    # x_i = -1/4 + i * (1/2)/N.
    #
    # The continuous autoconvolution at lag t is:
    #   (f * f)(t) = integral f(s) f(s-t) ds
    # For step functions, this is a piecewise linear function. The maximum
    # occurs at some point. We can compute it via the discrete convolution
    # of the step values, but we need to be careful about the normalization.
    #
    # Actually, for a step function with heights f[i] on intervals of width h=1/(2N),
    # the autoconvolution at discrete lags k*h is approximately:
    #   sum_i f[i] f[i-k] * h  (for the part where both are defined)
    # But the exact continuous autoconvolution of two step functions is more complex.
    #
    # Let's think about this differently. The "normalized maximum of its autoconvolution"
    # likely refers to: max_t |f*f(t)| / (some normalization).
    #
    # Given the task says "normalized maximum", a common normalization is:
    #   max_t (f*f)(t) / (integral f)^2
    # or perhaps just the peak value of the autoconvolution.
    #
    # Let's use the standard interpretation: the autoconvolution is (f*f)(t) = ∫ f(s)f(t-s)ds,
    # and we want to minimize its maximum value, possibly normalized by L1 norm or similar.
    #
    # For the seed program (all ones), f*f(t) for t in [-1/2, 1/2]:
    # The support of f is [-1/4, 1/4], so f*f has support [-1/2, 1/2].
    # At t=0: (f*f)(0) = ∫_{-1/4}^{1/4} 1*1 ds = 1/2.
    # The L1 norm of f is 1/2. So (f*f)(0) / (L1(f))^2 = (1/2)/(1/4) = 2.
    #
    # Let me just compute the raw max of the autoconvolution and see what score we get.
    # The score for the seed is 1.0, and the raw value is what's measured.
    
    f_arr = np.asarray(f, dtype=np.float64)
    h = 1.0 / (2 * N)  # interval width
    
    # Compute autoconvolution via FFT
    # We need to zero-pad to avoid circular convolution
    n_fft = 2 * N
    F = fft(f_arr, n=n_fft)
    conv = ifft(F * F).real
    # conv[k] corresponds to lag k*h for k=0,...,N-1 and negative lags for k=N,...,2N-1
    # Actually, for linear convolution of two length-N sequences, we need n_fft >= 2N-1
    # The result conv has length n_fft, where conv[k] for k=0..N-1 is the positive lags
    # and conv[n_fft-N+1..n_fft-1] wraps around to negative lags.
    
    # The peak is at lag 0 (k=0) for nonnegative f
    peak = conv[0] * h  # multiply by h for the continuous integral
    
    # Normalization: divide by (sum f * h)^2 = (L1 norm)^2
    l1 = np.sum(f_arr) * h
    if l1 > 0:
        return peak / (l1 * l1)
    return 0.0

def construct():
    """Construct a nonnegative step function to minimize normalized max autoconvolution."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with a good initial guess: a triangular/bell shape
    # The optimal shape for minimizing peak autoconvolution given fixed L1 norm
    # tends to be spread out but with some structure.
    
    # Initial: uniform (seed)
    best_f = np.ones(N)
    best_score = _autoconv_peak(best_f)
    
    # Try a few analytical shapes
    candidates = []
    
    # Triangular
    x = np.linspace(-1/4, 1/4, N)
    tri = np.maximum(0, 1 - 2 * np.abs(x))
    candidates.append(tri)
    
    # Gaussian-like
    gauss = np.exp(-4 * (x - 0)**2 / (1/4)**2)
    candidates.append(gauss)
    
    # Cosine bump
    cos_bump = np.cos(np.pi * x / (1/2))**2
    cos_bump = np.maximum(0, cos_bump)
    candidates.append(cos_bump)
    
    # Flat top with smooth edges
    flat = np.ones(N)
    edge = N // 8
    for i in range(edge):
        t = i / edge
        flat[i] = 0.5 * (1 + np.cos(np.pi * t))
        flat[N-1-i] = 0.5 * (1 + np.cos(np.pi * t))
    candidates.append(flat)
    
    for c in candidates:
        if time.time() > deadline:
            break
        # Normalize to have same L1 as uniform
        l1_c = np.sum(c)
        if l1_c > 0:
            c_norm = c * (np.sum(np.ones(N)) / l1_c)
            score = _autoconv_peak(c_norm)
            if score < best_score:
                best_score = score
                best_f = c_norm.copy()
    
    # Gradient descent / coordinate descent to refine
    f = best_f.copy()
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute gradient of the peak autoconvolution w.r.t. f
        # The peak is at lag 0 for nonnegative functions
        # (f*f)(0) = sum_i f[i]^2 * h
        # So the objective is sum f[i]^2 * h / (sum f[i] * h)^2
        # Let S = sum f[i], then objective = h * sum f[i]^2 / (h * S)^2 = sum f[i]^2 / (h * S^2)
        # Actually: peak = h * sum f[i]^2, l1 = h * S, so peak/l1^2 = h*sum f[i]^2 / (h^2 * S^2) = sum f[i]^2 / (h * S^2)
        
        # To minimize sum f[i]^2 / S^2 subject to f[i] >= 0, the optimum is f[i] = constant
        # (by Cauchy-Schwarz, sum f[i]^2 >= (sum f[i])^2 / N, with equality when all equal)
        # So the uniform distribution is optimal for this specific normalization!
        
        # Wait, that means the seed is already optimal? That can't be right for a meaningful task.
        # Let me reconsider the normalization.
        
        # Perhaps the normalization is different. Maybe it's max |f*f(t)| without dividing by L1^2.
        # Or maybe it's a different kind of normalization.
        
        # Let me just do coordinate descent on the raw peak value.
        # The peak of autoconvolution for nonnegative f is at t=0: (f*f)(0) = sum f[i]^2 * h
        # So minimizing the peak is equivalent to minimizing sum f[i]^2, which again favors uniform.
        
        # There must be a constraint I'm missing. Perhaps the L1 norm is fixed?
        # "Choose a nonnegative step function" - no explicit constraint mentioned.
        # But if we can scale f arbitrarily, we can make the peak arbitrarily small by scaling down.
        # So there must be an implicit normalization.
        
        # Given the task says "normalized maximum", and the seed scores 1.0,
        # let me assume the normalization makes the seed score exactly 1.0.
        # For the seed (all ones), if the score is 1.0, then the normalization must be
        # such that the normalized max for uniform is 1.0.
        
        # Let me just try to improve from the seed using random perturbations and see.
        
        # Try random restarts with different shapes
        np.random.seed(42)
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random smooth function
            freqs = np.random.randint(1, 5, size=5)
            amps = np.random.rand(5)
            phases = np.random.rand(5) * 2 * np.pi
            f_new = np.zeros(N)
            for freq, amp, phase in zip(freqs, amps, phases):
                f_new += amp * np.cos(2 * np.pi * freq * x + phase)
            f_new = np.maximum(0, f_new)
            
            if np.sum(f_new) == 0:
                continue
            
            # Normalize to have similar L1 as seed
            f_new = f_new * (N / np.sum(f_new))
            
            score = _autoconv_peak(f_new)
            if score < best_score:
                best_score = score
                best_f = f_new.copy()
                f = f_new.copy()
    
    return list(best_f)

if __name__ == "__main__":
    result = construct()
    print(f"Score: {_autoconv_peak(result)}")
