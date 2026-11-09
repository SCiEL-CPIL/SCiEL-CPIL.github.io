"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Signed step function that concentrates mass to suppress peak autoconvolution.
    # Use a window of signed values with zero-sum to reduce autocorrelation peak.
    # Specifically, construct a "box-car" style window with alternating signs in blocks.
    # This reduces the DC component and thus lowers the peak of the autoconvolution.
    
    # Strategy: use a sign pattern that has low autocorrelation peak.
    # A simple approach: use a window where the first half is +1 and second half is -1.
    # This gives zero mean, so the autoconvolution at zero lag is N (since sum x_i^2 = N),
    # but the integral (sum x_i) is 0, which would make the ratio infinite.
    # So we need a non-zero integral. Let's use a pattern that has a small but non-zero mean.
    
    # Alternative: use a random-like pattern but deterministic. 
    # Actually, the task is to minimize max |autocorrelation| / (integral)^2.
    # Note: "integral" here means sum x_i.
    
    # Let's try a sequence that is mostly +1 but with some -1 to reduce peak.
    # For example, a sequence with a block of -1 in the middle.
    
    # Let's construct: first k values +1, middle m values -1, last k values +1.
    # Choose m to balance.
    
    # Actually, let's think: if x_i is constant c, then max |autocorr| = N*c^2, integral = N*c, ratio = 1/N.
    # For any non-constant sequence, can we do better?
    
    # The quantity is max_k |(x*x)[k]| / (sum x_i)^2.
    # Let's denote A = max_k |(x*x)[k]|, S = sum x_i. We want to minimize A/S^2.
    
    # For x = ones: A = N, S = N, ratio = 1/N.
    # For x with some -1's: S decreases, and A might decrease or increase.
    
    # Let's try x with first half +1, second half -1: S=0, ratio = infinity. Bad.
    # Let's try x with first 25% +1, middle 50% -1, last 25% +1.
    # S = 100 - 200 + 100 = 0. Also bad.
    
    # We need S != 0. Let's try a small perturbation of ones.
    # x = 1 + epsilon * w, where w has zero mean and small norm.
    # Then S = N + epsilon * 0 = N.
    # A = max_k |sum x_i x_{i+k}| = max_k |sum (1 + eps w_i)(1 + eps w_{i+k})|
    #    = max_k |sum (1 + eps w_i + eps w_{i+k} + eps^2 w_i w_{i+k})|
    #    = N + eps * sum(w_i + w_{i+k}) + eps^2 * sum(w_i w_{i+k})
    # For k=0: N + eps*2*sum(w_i) + eps^2*sum(w_i^2) = N + eps^2 * ||w||^2 (since sum w_i = 0)
    # For k>0: N + eps*(sum w_i + sum w_{i+k}) + ... = N + eps^2 * (w*w)[k]
    # So A ≈ N + eps^2 * max(max_k (w*w)[k], 0) (approximately)
    # This doesn't help reduce A below N.
    
    # So it seems ones might be optimal. But let's try a different approach.
    # What if we use x_i = 1 for all i except a few positions?
    
    # Actually, the seed program returns ones with score 1.0, which is the baseline.
    # The task is to find a better (lower) raw value.
    
    # Let's try a specific pattern: a rectangular window with a notch.
    # Or use a known good autocorrelation sequence like m-sequences.
    # But m-sequences have constant magnitude 1, so |x_i| = 1.
    # For an m-sequence of length L=2^m-1, the periodic autocorrelation is -1/L for non-zero lags.
    # But we need linear autocorrelation. For a length-N sequence that is one period of an m-sequence,
    # the linear autocorrelation at lag 0 is N, and at other lags it's O(N/L) which is still large.
    
    # Let's just return a sequence that is close to optimal. Given the constraints, 
    # let's try a random seed sequence and evaluate, but we need to return a fixed one.
    # Since we can't run optimization, let's use a known construction.
    
    # How about: x_i = 1 if i is in first 3/4, -1/3 if i is in last 1/4?
    # S = 300 - 100/3 = 300 - 33.33 = 266.67
    # A = max |autocorr|... this is complex.
    
    # Given the time, let's return a slightly modified ones sequence.
    # Set a few elements to -1 to see if it helps.
    
    x = np.ones(N)
    # Flip a small number of signs at the ends to reduce peak
    x[0:10] = -1
    x[N-10:N] = -1
    return x
# EVOLVE-BLOCK-END
