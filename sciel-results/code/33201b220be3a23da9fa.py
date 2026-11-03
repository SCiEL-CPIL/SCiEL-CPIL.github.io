"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 15s safety margin before 130s kill (evaluator limit is 120s, but let's be safe)
    
    # The task: choose nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    # to minimize the normalized maximum of its autoconvolution.
    # This gives an upper bound on C1.
    
    # Let's think about this mathematically.
    # We have a step function f(x) with 600 equal intervals on [-1/4, 1/4].
    # Each interval has width h = (1/4 - (-1/4))/600 = (1/2)/600 = 1/1200.
    # Let the heights be a_0, a_1, ..., a_599 where a_i >= 0.
    
    # The autoconvolution of f is (f * f)(x) = integral f(y) f(x-y) dy.
    # For step functions, this can be computed discretely.
    
    # "Normalized maximum" - we need to figure out what normalization is used.
    # Typically for autocorrelation bounds, we might normalize by the L1 norm or L2 norm.
    
    # Let's assume the score is: max_x (f*f)(x) / (some normalization)
    # Common normalizations:
    # - L1 norm: integral |f| = sum a_i * h
    # - L2 norm: integral |f|^2 = sum a_i^2 * h
    # - L-infinity norm: max a_i
    
    # For the constant function f=1, let's compute:
    # (f*f)(x) for f=1 on [-1/4, 1/4]:
    # This is the length of the intersection of [-1/4, 1/4] and [-1/4+x, 1/4+x]
    # At x=0: intersection is [-1/4, 1/4], length = 1/2
    # So max (f*f) = 1/2
    
    # If normalized by L1^2 = (1/2)^2 = 1/4, then score = (1/2)/(1/4) = 2
    # If normalized by L1 = 1/2, then score = (1/2)/(1/2) = 1
    # If normalized by L2^2 = 1/2, then score = (1/2)/(1/2) = 1
    
    # The seed returns [1.0]*600 and has score 0.0000, which means it's invalid or the scoring
    # is different. Let me reconsider.
    
    # Actually, looking at the task again: "minimize the normalized maximum of its autoconvolution; 
    # this gives an upper bound on C1"
    
    # C1 in the context of autocorrelation inequalities... This might be related to the 
    # first autocorrelation coefficient or similar.
    
    # Let me think about what makes sense. For a function f on [-1/4, 1/4], the autocorrelation
    # at lag 0 is integral f(x)^2 dx. The "first autocorrelation" might refer to something specific.
    
    # Given the seed program returns all 1s and scores 0.0000 (invalid), perhaps the normalization
    # requires the function to be normalized in some way, or the score computation has specific 
    # requirements.
    
    # Let me try a different approach: use optimization to find a good step function.
    
    # Let's define the autoconvolution more carefully.
    # f(x) = a_i for x in [x_i, x_{i+1}) where x_i = -1/4 + i*h, h = 1/1200
    # (f*f)(z) = integral f(y) f(z-y) dy
    
    # For discrete step functions, we can compute this numerically.
    
    # Let me implement a simple gradient-based or coordinate descent approach.
    
    def compute_score(a):
        """Compute the normalized maximum of autoconvolution for step heights a."""
        h = 1.0 / 1200.0
        a = np.asarray(a, dtype=float)
        n = len(a)
        
        # Compute autoconvolution using FFT for efficiency
        # Pad to avoid circular convolution
        n_fft = 2 * n
        A = np.zeros(n_fft)
        A[:n] = a
        A_fft = np.fft.fft(A)
        conv = np.fft.ifft(A_fft * A_fft).real
        conv = conv * h  # Scale by interval width
        
        # The autoconvolution is defined on [-1/2, 1/2] effectively
        # We need the maximum over the valid range
        # The valid range for z is where both f(y) and f(z-y) are defined
        # f is defined on [-1/4, 1/4], so z-y in [-1/4, 1/4] and y in [-1/4, 1/4]
        # So z in [-1/2, 1/2]
        
        # The convolution result conv[i] corresponds to z = (i - n/2) * h approximately
        # Let's just take the max of the relevant portion
        # The autoconvolution at lag 0 (z=0) should be the largest typically
        
        max_conv = np.max(conv)
        
        # Normalization: let's try L1 norm squared
        l1 = np.sum(a) * h
        if l1 <= 0:
            return 1e10
        score = max_conv / (l1 * l1)
        
        return score
    
    # Try some simple patterns first
    best_a = np.ones(N)
    best_score = compute_score(best_a)
    
    # Try a triangular profile
    tri_a = np.linspace(1, 0, N) + np.linspace(0, 1, N)  # tent shape
    tri_a = np.maximum(tri_a, 0)
    s = compute_score(tri_a)
    if s < best_score:
        best_a = tri_a.copy()
        best_score = s
    
    # Try a Gaussian-like profile
    x = np.linspace(-1, 1, N)
    gauss_a = np.exp(-x**2 * 5)
    s = compute_score(gauss_a)
    if s < best_score:
        best_a = gauss_a.copy()
        best_score = s
    
    # Try a compact support function (nonzero only in middle)
    compact_a = np.zeros(N)
    compact_a[150:450] = 1.0
    s = compute_score(compact_a)
    if s < best_score:
        best_a = compact_a.copy()
        best_score = s
    
    # Try to optimize using coordinate descent
    a = best_a.copy()
    a = np.maximum(a, 1e-10)  # Ensure positive
    
    # Coordinate descent
    for iteration in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = a[i]
            # Try a few candidate values
            for candidate in [0.0, old_val * 0.5, old_val * 0.8, old_val, old_val * 1.2, old_val * 1.5, 1.0]:
                a[i] = max(0.0, candidate)
                new_score = compute_score(a)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    a[i] = old_val
                    break
        if not improved:
            break
    
    # Ensure nonnegativity
    a = np.maximum(a, 0.0)
    
    # Convert to list
    return [float(x) for x in a]
# EVOLVE-BLOCK-END
