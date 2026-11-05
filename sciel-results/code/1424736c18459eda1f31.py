"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_value(h):
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / s**2)

def _try_sequence(h):
    return _compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with a good base: use a sequence that has low peak autocorrelation
    # Strategy: use a "chirp" or structured sequence
    
    # Try a binary-like sequence with careful sign pattern
    # Key insight: for real sequences, the minimum of max autocorrelation / (sum)^2
    # is related to the Welch bound. For N real signals, but here we have one signal.
    
    # Let's try: h[i] = 1 for most, but with a few negative values to reduce peak
    # Or better: use an optimization approach
    
    # First, try a simple structured approach:
    # h[i] = sign(cos(2*pi*i/N)) gives alternating blocks
    
    # Actually, let's try a gradient-free search starting from a good initial point
    
    # Initial guess: a sequence that's mostly 1s with a small perturbation
    # to reduce the peak of the autocorrelation
    
    # Better idea: use the fact that if h is a "perfect" sequence, 
    # all non-zero lags of autocorrelation are 0.
    # For real sequences of length N, this is impossible for N > 4,
    # but we can get close.
    
    # Let's try a Zadoff-Chu-like sequence adapted for real values
    # Or use a Barker-code-inspired approach
    
    # Practical approach: start with ones, then do coordinate descent
    # to minimize the value
    
    h = np.ones(N)
    best_val = _try_sequence(h)
    best_h = h.copy()
    
    # Coordinate descent: flip or adjust individual elements
    rng = np.random.default_rng(42)
    
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try adjusting a random element
        idx = rng.integers(0, N)
        old_val = h[idx]
        
        # Try a few candidate new values
        candidates = [
            -old_val,  # flip sign
            old_val + 0.1,
            old_val - 0.1,
            0.0,
            2.0 * old_val,
        ]
        
        best_candidate = old_val
        best_candidate_val = best_val
        
        for cand in candidates:
            h[idx] = cand
            val = _try_sequence(h)
            if val < best_candidate_val:
                best_candidate_val = val
                best_candidate = cand
        
        h[idx] = best_candidate
        if best_candidate_val < best_val:
            best_val = best_candidate_val
            best_h = h.copy()
        else:
            h[idx] = old_val
    
    # Try a second phase: more aggressive random restarts
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation of best_h
            h = best_h + 0.5 * rng.standard_normal(N)
            # Ensure sum is not zero
            if abs(np.sum(h)) < 1e-10:
                h += 0.1
            val = _try_sequence(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                
            # Quick local improvement
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                idx = rng.integers(0, N)
                old_val = h[idx]
                candidates = [-old_val, old_val + 0.1, old_val - 0.1, 0.0]
                best_cand = old_val
                best_cand_val = _try_sequence(h)
                for cand in candidates:
                    h[idx] = cand
                    v = _try_sequence(h)
                    if v < best_cand_val:
                        best_cand_val = v
                        best_cand = cand
                h[idx] = best_cand
                if best_cand_val < best_val:
                    best_val = best_cand_val
                    best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
