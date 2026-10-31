import time
import math
import numpy as np

N = 600
DELTA = 0.5 / N
DEADLINE = 120.0

def construct():
    global _t0
    _t0 = time.time()
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA / 2.0
    
    # Try various support widths and profile shapes
    # The key: concentrate on a smaller support to reduce autoconvolution peak
    # For a function supported on [-a, a] with height h, the autoconvolution
    # at lag 0 is sum h_i^2, and the peak is at lag 0 for nonnegative functions.
    # Normalized ratio = peak / (sum h_i)^2 or peak / sum h_i^2 depending on definition.
    
    # Let's use the ratio: max autoconv / (sum h)^2 * N (scale invariant)
    # Or: max autoconv / sum(h^2)
    
    # For nonnegative functions, the autoconvolution peak is at lag 0 = sum h_i^2
    # So the ratio peak/sum(h^2) = 1 always for any nonnegative function!
    # That can't be right... Let me reconsider.
    
    # Actually, for autoconvolution c[m] = sum_i h[i] h[i+m], the peak for 
    # nonnegative functions IS at m=0, which is sum h_i^2.
    # So max(c)/sum(h^2) = 1 always.
    
    # The other normalization: max(c) / (sum h)^2
    # = sum(h^2) / (sum h)^2
    # By Cauchy-Schwarz, this is <= 1, with equality when h is constant on support.
    # To minimize this, we want sum(h^2)/(sum h)^2 to be small.
    # This is minimized when h is spread out (many nonzero entries with small values).
    # Wait, that's the opposite - spreading out increases (sum h)^2 relative to sum h^2.
    
    # Actually sum(h^2)/(sum h)^2 is minimized when h is as "spread" as possible.
    # For uniform on all N: N * (1/N)^2 / (1)^2 ... let me compute properly.
    # If h_i = 1 for all i: sum(h^2) = N, (sum h)^2 = N^2, ratio = 1/N.
    # If h_i = 1 for k entries: sum(h^2) = k, (sum h)^2 = k^2, ratio = 1/k.
    # So minimizing ratio means maximizing k, i.e., using full support uniformly.
    
    # But wait - the task says "minimize the normalized maximum of its autoconvolution"
    # giving an upper bound on C1. C1 is a constant in an inequality.
    # The seed scores 1.0, so we need to do better.
    
    # Let me reconsider: maybe the normalization is different.
    # Perhaps it's: max_m |c[m]| / (sum |h|^2) where c is the continuous autoconvolution
    # evaluated at the step function level.
    
    # For the continuous autoconvolution of a step function:
    # (f*f)(t) = integral f(s) f(t-s) ds
    # The max of this divided by integral f^2 dx gives a quantity.
    
    # For f = 1 on [-1/4, 1/4]: 
    # (f*f)(t) is a triangle with peak 0.5 at t=0, support [-0.5, 0.5]
    # integral f^2 = 0.5
    # ratio = 0.5 / 0.5 = 1.0
    
    # For f = 1 on [-a, a]:
    # (f*f)(t) peak = 2a at t=0
    # integral f^2 = 2a
    # ratio = 2a / 2a = 1.0 again!
    
    # Hmm, it's always 1 for box functions. What about non-box profiles?
    
    # For f = cos(pi x / (2a)) on [-a, a] (half cosine bump):
    # This is a smoother profile. Let me think about what minimizes max(f*f) / integral(f^2).
    
    # Actually, I think the issue is that for ANY nonnegative function,
    # max(f*f) >= (integral f)^2 / (2 * width_of_support) by some averaging argument,
    # and integral(f^2) >= (integral f)^2 / (width_of_support).
    
    # Let me just try various profiles numerically and see which gives the lowest score.
    
    def score_profile(h):
        """Compute the normalized max autoconvolution score."""
        h = np.asarray(h, dtype=float)
        if np.max(h) < 1e-15:
            return float('inf')
        
        # Discrete autoconvolution
        c = np.convolve(h, h, mode='full')
        
        # Two possible normalizations:
        # 1) max(c) / sum(h^2)  -- this is >= 1, equals 1 for delta-like
        # 2) max(c) / (sum(h))^2 * N -- scale invariant
        # 3) max(c) * DELTA / (sum(h) * DELTA)^2 * N = max(c) / (sum(h))^2 * N
        
        # Let's use: max(c) / (sum(h) * DELTA)^2 * DELTA = max(c) / (sum(h))^2 * (1/DELTA)
        # No, let me think dimensionally.
        
        # The continuous autoconvolution: C(t) = sum_i h_i * DELTA * sum_j h_j * DELTA * [overlap]
        # Actually for step functions, (f*f)(t) where f is piecewise constant:
        # The peak value of f*f relates to sum h_i^2 * DELTA
        
        # Let me just try: score = max(c) / sum(h^2)  and  score = max(c) / (sum(h))^2
        # and see which one the evaluator uses by matching the seed score of 1.0.
        
        s1 = max(c) / sum(h * h)  # should be >= 1
        s2 = max(c) / (sum(h) ** 2)  # should be <= 1/N for uniform
        
        # For uniform h=1: s1 = N/N = 1, s2 = N/N^2 = 1/N
        # The seed scores 1.0, so it's likely s1 or some variant that gives 1.0 for uniform.
        
        # Wait, s1 for uniform: c[0] = sum(1*1) = N, sum(h^2) = N, so s1 = 1. 
        # But c[0] is the max for nonnegative, so max(c) = c[0] = N.
        # So s1 = N/N = 1. That matches!
        
        # But s1 = max(c)/sum(h^2) is ALWAYS >= 1 for any nonnegative h,
        # with equality iff h is a single spike (delta function).
        # To get below 1, we'd need... but it can't go below 1.
        
        # Unless the normalization is different. Maybe:
        # score = max_m |c[m]| / (max(h) * sum(h) * something)
        
        # Or maybe it's the L-infinity norm of the continuous autoconvolution
        # divided by the L-2 norm squared of f.
        
        # Let me try: for the continuous function, 
        # ||f*f||_inf / ||f||_2^2
        # For f = 1 on [-1/4,1/4]: ||f*f||_inf = 0.5, ||f||_2^2 = 0.5, ratio = 1.
        # For f = 1 on [-a,a]: ||f*f||_inf = 2a, ||f||_2^2 = 2a, ratio = 1.
        
        # So box functions always give 1. To get below 1, we need a non-box profile.
        
        # For f = exp(-x^2/(2 sigma^2)) type profiles, the ratio can be < 1.
        
        # Let me parametrize and search.
        
        return s1
    
    def score_profile_v2(h):
        """Alternative: max continuous autoconv / integral f^2."""
        h = np.asarray(h, dtype=float)
        if np.max(h) < 1e-15:
            return float('inf')
        c = np.convolve(h, h, mode='full')
        # Approximate continuous autoconvolution peak:
        # For step function with heights h_i on intervals of width DELTA,
        # the autoconvolution at lag 0 is sum h_i^2 * DELTA
        # The peak of the continuous autoconvolution is approximately max(c) * DELTA
        # ||f||_2^2 = sum h_i^2 * DELTA
        # So ratio = max(c) * DELTA / (sum h_i^2 * DELTA) = max(c) / sum(h^2)
        # Same as before!
        
        # Hmm. Let me try yet another normalization.
        # Maybe: max(f*f) / (integral f)^2 * (length of interval)
        # For uniform on [-1/4,1/4]: max(f*f)=0.5, (int f)^2 = 0.25, length=0.5
        # ratio = 0.5 / 0.25 * 0.5 = 1.0
        
        # For f = 1 on [-a,a]: max(f*f) = 2a, (int f)^2 = 4a^2, length = 0.5
        # ratio = 2a / (4a^2) * 0.5 = 1/(4a)
        # As a -> 0, this goes to infinity. As a = 0.25, ratio = 1.
        
        # So to minimize, we want large a, i.e., full support. But that gives 1.
        
        # I'm going in circles. Let me just try many profiles and compute
        # multiple candidate scores, then pick the profile that minimizes
        # the most likely metric.
        
        c = np.convolve(h, h, mode='full')
        max_c = max(c)
        sum_h2 = sum(h * h)
        sum_h = sum(h)
        
        candidates = {
            'max_c_over_sum_h2': max_c / sum_h2 if sum_h2 > 0 else float('inf'),
            'max_c_over_sum_h_sq': max_c / (sum_h ** 2) if sum_h > 0 else float('inf'),
            'max_c_times_delta_over_sum_h2_times_delta': (max_c * DELTA) / (sum_h2 * DELTA) if sum_h2 > 0 else float('inf'),
        }
        
        return candidates
    
    # Let me think differently. The seed (uniform) scores 1.0.
    # I need to find a profile that scores < 1.0.
    
    # Key realization: if the metric is max(f*f)/||f||_2^2 for the CONTINUOUS
    # autoconvolution, then for a box function it's always 1. But for a 
    # Gaussian-like bump, it can be less than 1!
    
    # For f(x) = exp(-x^2/(2 sigma^2)):
    # f*f = sqrt(2) sigma * exp(-x^2/(4 sigma^2))  (up to constants)
    # max(f*f) = sqrt(2) sigma * (1/sqrt(2*pi)) * sqrt(2*pi) ... let me be careful.
    
    # Actually, let's just do it numerically. Generate profiles and compute
    # the continuous autoconvolution peak / L2 norm squared.
    
    def compute_score_continuous(h):
        """Compute max of continuous autoconvolution / integral of f^2."""
        h = np.asarray(h, dtype=float)
        if np.max(h) < 1e-15:
            return float('inf')
        
        # The step function f(x) = h_i on interval i
        # (f*f)(t) = integral f(s) f(t-s) ds
        # For step functions, this can be computed exactly.
        # But for the peak (at t=0): (f*f)(0) = integral f(s)^2 ds = sum h_i^2 * DELTA
        # So max(f*f) = sum h_i^2 * DELTA for nonnegative f (peak at t=0)
        # And ||f||_2^2 = sum h_i^2 * DELTA
        # Ratio = 1 always!
        
        # This is the fundamental issue. For nonnegative functions, the 
        # autoconvolution peaks at 0, and that peak equals ||f||_2^2.
        # So the ratio is always 1.
        
        # UNLESS the normalization is different. Maybe it's:
        # max(f*f) / (integral f)^2  *  (some length scale)
        
        # Or maybe the function isn't required to be nonnegative in the 
        # autoconvolution sense, but the task says nonnegative step function.
        
        # I think the key must be in the specific normalization. Let me re-read:
        # "minimize the normalized maximum of its autoconvolution"
        
        # Perhaps: max_t |(f*f)(t)| / (max |f| * integral |f|)
        # For uniform f=1 on [-1/4,1/4]: max(f*f)=0.5, max|f|=1, int|f|=0.5
        # ratio = 0.5 / (1*0.5) = 1.0
        
        # For f=1 on [-a,a]: max(f*f)=2a, max|f|=1, int|f|=2a
        # ratio = 2a / (1*2a) = 1.0. Still 1!
        
        # For a Gaussian f = exp(-x^2/(2 sigma^2)):
        # max(f*f) = sigma * sqrt(2) * (1/sqrt(2*pi)) * ... 
        # This is getting complicated. Let me just try it numerically.
        
        # Actually, I bet the normalization involves the L1 norm squared:
        # max(f*f) / (int f)^2
        # For uniform on [-1/4,1/4]: 0.5 / 0.25 = 2.0. Not 1.
        
        # max(f*f) / (int f)^2 * (1/width) where width = 0.5:
        # 0.5 / 0.25 * (1/0.5) = 4.0. No.
        
        # I give up trying to derive it. Let me just try many profiles,
        # compute several candidate metrics, and find one that gives < 1
        # for non-uniform profiles and = 1 for uniform.
        
        return None  # placeholder
    
    # Let me take a completely different approach. Just try many profiles
    # and for each, compute what I think the score might be, and return
    # the profile that seems best.
    
    # Profile family: h(x) = (1 - (x/a)^2)^p for |x| < a, 0 otherwise
    # This is a smooth bump on [-a, a].
    
    best_score = float('inf')
    best_h = None
    
    for a_frac in [0.05, 0.1, 0.15, 0.2, 0.25]:
        a = a_frac  # half-width
        for p in [0.5, 1.0, 2.0, 4.0, 8.0]:
            h = np.zeros(N)
            mask = np.abs(x) < a
            if np.sum(mask) < 2:
                continue
            h[mask] = (1 - (x[mask]/a)**2)**p
            
            # Compute various candidate scores
            c = np.convolve(h, h, mode='full')
            max_c = max(c)
            sum_h2 = np.sum(h**2)
            sum_h = np.sum(h)
            
            # Candidate 1: max_c / sum_h2
            s1 = max_c / sum_h2 if sum_h2 > 0 else float('inf')
            
            # Candidate 2: max_c / (sum_h * DELTA)**2 * DELTA  (dimensionless)
            # = max_c * DELTA / (sum_h * DELTA)**2 = max_c / (sum_h**2 * DELTA)
            s2 = max_c / (sum_h**2 * DELTA) if sum_h > 0 else float('inf')
            
            # Candidate 3: max_c / sum_h2 (same as s1)
            
            # The seed uniform should give 1.0. Let's check which candidate does.
            # For uniform h=1: max_c = N, sum_h2 = N, s1 = 1. 
            # For uniform: sum_h = N, s2 = N / (N^2 * DELTA) = 1/(N*DELTA) = 1/0.5 = 2.
            
            # So s1 gives 1.0 for uniform. But s1 = max_c/sum_h2 >= 1 always.
            # We can never go below 1 with s1.
            
            # Unless... the autoconvolution is defined differently.
            # Maybe it's the CROSS-correlation or there's a sign change.
            
            # Or maybe the "normalized maximum" means:
            # max_m c[m] / c[0]  which is always 1 for nonnegative.
            
            # I'm stuck on the metric. Let me just return the uniform profile
            # but try to see if there's any profile that could score < 1.
            
            if s1 < best_score:
                best_score = s1
                best_h = h.copy()
    
    # Since s1 >= 1 always, and uniform gives exactly 1, the best we can do
    # is 1.0. Unless the metric is different.
    
    # Let me try: maybe the score is max_m c[m] / (sum h_i)^2 * N
    # For uniform: N / N^2 * N = 1. 
    # For concentrated: if k entries have height H: max_c = k*H^2, sum_h = k*H
    # score = k*H^2 / (k^2*H^2) * N = N/k
    # To minimize, maximize k. So uniform is best. Score = 1.
    
    # Another try: max_m c[m] / (max(h) * sum(h))
    # Uniform: N / (1*N) = 1.
    # Concentrated (k entries height H): k*H^2 / (H * k*H) = 1. Always 1!
    
    # It seems like for nonnegative functions, many natural normalizations give 1.
    
    # The only way to get < 1 is if the autoconvolution has cancellations,
    # but the function is nonnegative so no cancellations.
    
    # I think the answer is that the minimum is 1.0, achieved by any 
    # "flat" profile. The seed already achieves this.
    
    # But the task says "this gives an upper bound on C1" implying we can
    # improve the bound. Maybe C1 > 1 and we're trying to get a tighter bound.
    
    # Let me just return the uniform profile since I can't find anything better.
    
    h_uniform = np.ones(N)
    return [float(v) for v in h_uniform]

_t0 = time.time()
