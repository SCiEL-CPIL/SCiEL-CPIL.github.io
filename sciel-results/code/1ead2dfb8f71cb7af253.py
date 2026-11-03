"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct step heights h[0..N-1] in [0,1] with sum exactly N/2.
    Minimize the maximum overlap of h with shifted 1-h.
    
    The overlap for shift k is: sum_i h[i] * (1 - h[(i+k) % N])
    = sum_i h[i] - sum_i h[i]h[(i+k)%N]
    = N/2 - autocorr(k)
    
    So minimizing max_k overlap is equivalent to maximizing min_k autocorr(k).
    We want the autocorrelation to be as flat and high as possible.
    
    Strategy: Use a pure tone (complex exponential) whose real part gives a
    smooth periodic signal, then project onto [0,1] with sum constraint.
    Search over frequency and phase to maximize the minimum autocorrelation.
    """
    deadline = time.time() + 140  # 10s safety margin
    
    best_min_ac = -np.inf
    best_h = np.full(N, 0.5)
    
    # Try a range of frequencies and phases
    # The autocorrelation of a pure cosine is also a cosine, so it has peaks
    # and troughs. We want to find parameters where the minimum is maximized.
    
    # Actually, let's think differently. We want h such that h * (1-h) shifted
    # has small max. 
    
    # Let's try: h[i] = 0.5 + A * cos(2*pi*f*i/N + phase)
    # Then clip to [0,1] and adjust to satisfy sum constraint.
    
    # For a pure cosine, the autocorrelation is A^2 * cos(2*pi*f*k/N)
    # So min autocorr = -A^2 (when cos = -1)
    # Overlap = N/2 - autocorr, so max overlap = N/2 + A^2
    
    # To minimize max overlap, we want small A. But A=0 gives h=0.5 everywhere,
    # which gives overlap = N/2 - N/4 = N/4 = 25 for N=100.
    
    # Wait, let me recompute. If h[i] = 0.5, then:
    # overlap(k) = sum_i 0.5 * (1-0.5) = sum_i 0.25 = N * 0.25 = 25
    
    # Can we do better? We need sum_i h[i]*(1-h[i+k]) to be smaller.
    # = N/2 - sum_i h[i]h[i+k]
    # So we need sum_i h[i]h[i+k] to be larger for all k.
    
    # The maximum possible sum_i h[i]h[i+k] is N/4 (when h is constant 0.5).
    # So 0.5 is actually optimal for constant! 
    # 
    # Wait, that means constant 0.5 gives min possible overlap of N/4 = 25.
    # But we can't do better than that? Let me check.
    #
    # sum_i h[i]h[i+k] <= (sum_i h[i]^2) by Cauchy-Schwarz? No, that's not right.
    # Actually, by the spectral representation, the autocorrelation at lag 0 is
    # sum h[i]^2, and at other lags it's bounded by sum h[i]^2.
    #
    # Hmm, let me just try to optimize numerically.
    
    # Let's try a direct approach: parameterize h as 0.5 + x where sum x = 0,
    # and x[i] in [-0.5, 0.5]. Minimize max_k sum x[i](x[i+k]) + N/4 - N/4
    # Wait, overlap(k) = N/2 - sum h[i]h[i+k]
    # = N/2 - sum (0.5+x[i])(0.5+x[i+k])
    # = N/2 - (N/4 + 0.5*sum x[i] + 0.5*sum x[i+k] + sum x[i]x[i+k])
    # = N/2 - N/4 - 0.5*sum x[i] - 0.5*sum x[i+k] - sum x[i]x[i+k]
    # Since sum x = 0, sum x[i] = 0 and sum x[i+k] = 0:
    # = N/4 - sum x[i]x[i+k]
    
    # So we want to maximize min_k sum x[i]x[i+k] where sum x = 0, x in [-0.5, 0.5].
    
    # Let's use a gradient-free search over a low-dimensional parameterization.
    
    # Try: x[i] = A * cos(2*pi*f*i/N + phase)
    # Then sum x[i]x[i+k] = A^2/2 * cos(2*pi*f*k/N) (for large N, ignoring edges)
    # Min over k is -A^2/2
    # So we want small A. A=0 gives 0, which is the best.
    
    # This suggests constant 0.5 is optimal! But let's verify with a numerical
    # search to be sure.
    
    # Let's do a random search to see if we can beat 25.
    
    np.random.seed(42)
    
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        # Random perturbation around 0.5
        x = np.random.randn(N) * 0.1
        x -= np.mean(x)  # ensure sum is 0
        x = np.clip(x, -0.5, 0.5)
        # Re-center after clipping
        x -= np.mean(x)
        
        h = 0.5 + x
        
        # Compute max overlap
        max_ov = 0
        for k in range(1, N):
            if time.time() > deadline:
                break
            ov = np.sum(h * (1 - np.roll(h, k)))
            if ov > max_ov:
                max_ov = ov
        
        if max_ov < best_min_ac or best_min_ac == -np.inf:
            best_min_ac = max_ov
            best_h = h.copy()
        
        # Early stop if we hit the theoretical lower bound
        if max_ov <= 25.0 + 1e-6:
            break
    
    # If we haven't beaten 25, return the constant solution
    if best_min_ac >= 25.0 - 1e-6:
        return np.full(N, 0.5)
    
    # Ensure sum is exactly N/2
    best_h = np.clip(best_h, 0, 1)
    best_h = best_h - np.mean(best_h) + 0.5
    best_h = np.clip(best_h, 0, 1)
    
    # Final adjustment for exact sum
    diff = N/2 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Distribute the difference
        idx = np.arange(N)
        best_h[idx] += diff / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
