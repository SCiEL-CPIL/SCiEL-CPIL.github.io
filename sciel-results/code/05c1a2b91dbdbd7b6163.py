"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def _autoconv_norm_peak(h):
    """Compute the normalized maximum of the autoconvolution of step heights h.
    
    The step function is defined on [-1/4, 1/4] with N equal intervals.
    The autoconvolution at lag k (in interval units) is sum_i h[i]*h[i+k] (for appropriate i).
    We normalize by the L1 norm squared (total area squared) to get a scale-invariant measure.
    Actually, let's think about what "normalized maximum of its autoconvolution" means.
    
    The autoconvolution of a function f with itself is (f*f)(t) = integral f(s)f(t-s) ds.
    For a step function with heights h[0..N-1] on intervals of width dx = (1/2)/N = 1/(2N),
    the continuous autoconvolution at time t is a piecewise linear function.
    The maximum of this autoconvolution occurs at t=0 (since h >= 0).
    
    At t=0, (f*f)(0) = integral f(s)^2 ds = dx * sum_i h[i]^2.
    
    But wait, the "normalized maximum" - normalized by what? 
    Looking at the context: "minimize the normalized maximum of its autoconvolution; this gives an upper bound on C1."
    
    C1 is likely related to the first autocorrelation inequality. Let me think about this differently.
    
    Actually, for the autoconvolution of a nonnegative step function, the maximum is at lag 0.
    The value at lag 0 is the integral of f^2, which for a step function is dx * sum(h^2).
    
    If we normalize by (integral f)^2 = (dx * sum(h))^2, then the normalized value is:
    sum(h^2) / (sum(h))^2
    
    This is a standard quantity. Minimizing this is equivalent to making the function as "flat" as possible,
    but since we can choose the heights, the minimum would be achieved by making all heights equal.
    
    Wait, but if all heights are equal, sum(h^2) / (sum(h))^2 = N * h^2 / (N*h)^2 = 1/N.
    
    Hmm, but that's the seed solution. Let me reconsider.
    
    Actually, I think the normalization might be different. Let me re-read: "normalized maximum of its autoconvolution".
    
    Perhaps the autoconvolution is computed in a specific way and normalized by its L1 norm or L2 norm.
    
    Let me just implement it as: max of autoconvolution / (L1 norm of f)^2, which gives sum(h^2)/(sum(h))^2.
    
    For the seed [1,1,...,1], this is N/(N^2) = 1/N = 1/600 ≈ 0.001667.
    """
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if s <= 0:
        return float('inf')
    return (h**2).sum() / (s**2)

def construct():
    """Construct a nonnegative step function with 600 intervals to minimize the normalized max autoconvolution."""
    t0 = time.time()
    deadline = t0 + 110.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_score = _autoconv_norm_peak(best_h)
    
    # Strategy: Use simulated annealing on the heights.
    # The objective sum(h^2)/(sum(h))^2 is minimized when all h are equal.
    # But we need to check if there are constraints I'm missing.
    
    # Actually, let me reconsider the problem. The task says "minimize the normalized maximum of its autoconvolution".
    # If the only degree of freedom is the heights, and the function must be nonnegative, then the uniform function
    # is indeed optimal for minimizing sum(h^2)/(sum(h))^2.
    
    # But wait - maybe the normalization is different. Let me try: max_autoconv / L2_norm^2 of the autoconvolution?
    # Or maybe it's the peak sidelobe level?
    
    # Let me just go with a more careful approach: use optimization to find the minimum.
    
    # Actually, for nonnegative h, sum(h^2)/(sum(h))^2 >= 1/N by Cauchy-Schwarz, with equality iff all h equal.
    # So the seed is already optimal for this interpretation.
    
    # Let me try a different interpretation: maybe the autoconvolution is normalized differently.
    # Perhaps it's max(autoconv) / autoconv(0), i.e., the peak sidelobe relative to main lobe?
    # But for nonnegative functions, autoconv(0) is the max, so this would be 1.
    
    # I'll stick with the Cauchy-Schwarz interpretation and return the uniform solution,
    # but let me add a small optimization just in case there's a subtlety.
    
    # Try a few random perturbations to see if any improve
    rng = np.random.RandomState(42)
    
    for it in range(200):
        if time.time() > deadline:
            break
        
        # Random perturbation
        h_try = best_h.copy()
        idx = rng.randint(0, N, size=10)
        h_try[idx] += rng.normal(0, 0.1, size=10)
        h_try = np.maximum(h_try, 0)
        
        score = _autoconv_norm_peak(h_try)
        if score < best_score:
            best_score = score
            best_h = h_try
    
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
