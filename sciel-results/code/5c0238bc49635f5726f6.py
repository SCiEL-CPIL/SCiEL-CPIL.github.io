"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve


def _autoconv(s):
    """Compute full autoconvolution of sign sequence."""
    return fftconvolve(s.astype(np.int32), s.astype(np.int32), mode='full')


def _max_abs_ac(ac):
    """Return max absolute value of autoconvolution."""
    return np.max(np.abs(ac))


def _score(signs):
    """Compute the C3-prime value for a binary sign sequence."""
    s = signs.astype(np.int32)
    integral = np.sum(s)
    if integral == 0:
        return np.inf
    denom = integral ** 2
    ac = _autoconv(s)
    max_ac = _max_abs_ac(ac)
    return float(max_ac) / denom


def construct():
    # Start with all-ones sequence
    signs = np.ones(N, dtype=np.int32)
    current_score = _score(signs)
    
    # Precompute autoconvolution for efficient updates
    ac = _autoconv(signs)
    current_max = _max_abs_ac(ac)
    
    # Local search with efficient incremental updates
    # When flipping sign at position i, the autoconvolution changes in a predictable way
    # ac[k] = sum_j s[j] * s[j+k]
    # Flipping s[i] to -s[i] changes ac by:
    # For each offset d, ac[d] changes by -2*s[i]*s[i+d] (when i+d is valid)
    # and ac[-d] changes by -2*s[i]*s[i-d] (when i-d is valid)
    
    max_iterations = 50
    for iteration in range(max_iterations):
        improved = False
        # Try flipping each position
        for i in range(N):
            old_sign = signs[i]
            new_sign = -old_sign
            
            # Compute the change in autoconvolution
            # ac_new[k] = ac_old[k] - 2 * old_sign * (contribution from position i)
            # The contribution of position i to ac[k] is:
            #   if k >= 0: s[i] * s[i+k] (if i+k < N)
            #   if k < 0: s[i+k] * s[i] (if i+k >= 0)
            # So flipping s[i] to -s[i] changes ac[k] by:
            #   -2 * old_sign * s[i+k] for k >= 0, i+k < N
            #   -2 * old_sign * s[i+k] for k < 0, i+k >= 0
            #   Note: s[i+k] here is the OLD value
            
            # Build the delta vector
            # For k in range(-(i), N-i): ac[k] changes by -2 * old_sign * signs[i+k]
            # But we need to be careful about the indexing in full mode
            # In full mode, ac has length 2N-1, indexed 0 to 2N-2
            # ac[m] = sum_j s[j] * s[j + (m - (N-1))]
            # So ac[m] corresponds to offset d = m - (N-1)
            # ac[m] = sum_j s[j] * s[j + d] where d = m - (N-1)
            
            # When we flip s[i], for each valid j such that both s[j] and s[j+d] are involved:
            # The term involving s[i] is when j = i or j+d = i
            # If j = i: term is s[i] * s[i+d], changes by -2*s[i]*s[i+d]
            # If j+d = i, i.e., j = i-d: term is s[i-d] * s[i], changes by -2*s[i-d]*s[i]
            # But note: when d=0, both conditions give the same term, so we only count once
            
            # Let's compute delta for each m in 0..2N-2
            # d = m - (N-1)
            # delta[m] = -2 * old_sign * (s[i+d] if i+d in [0,N) else 0)
            #           -2 * old_sign * (s[i-d] if i-d in [0,N) else 0)
            # But if d=0, we double count, so subtract one copy
            # Actually, the autoconvolution is:
            # ac[m] = sum_{j} s[j] * s[j + (m-(N-1))]
            # The terms involving s[i] are:
            # 1. j = i: s[i] * s[i + d] where d = m-(N-1), valid if 0 <= i+d < N
            # 2. j + d = i => j = i - d: s[i-d] * s[i], valid if 0 <= i-d < N
            # If d = 0, these are the same term, so total change is -2 * s[i] * s[i] = -2
            # If d != 0, total change is -2 * s[i] * s[i+d] (if valid) - 2 * s[i-d] * s[i] (if valid)
            # But wait, when we flip s[i] to -s[i], the new term becomes:
            # (-s[i]) * s[i+d] = -s[i]*s[i+d]
            # So the change is -2 * s[i] * s[i+d]
            # Similarly for the other term.
            
            # Let's just compute the new autoconvolution directly for simplicity
            # but that's O(N^2) per flip. Instead, let's use FFT-based approach
            # or just accept O(N) per flip with the delta computation
            
            # For efficiency, let's compute the delta vector
            delta = np.zeros(2 * N - 1, dtype=np.int32)
            d_start = -(i)
            d_end = N - 1 - i
            # For each d in [d_start, d_end], the index m = d + (N-1)
            # delta[m] gets contribution from flipping s[i]
            # The term s[i] * s[i+d] contributes -2 * old_sign * signs[i+d] to ac[m]
            # But we also need to consider that ac is symmetric: ac[m] = ac[2N-2-m]
            # Actually, let's just compute it directly
            
            # For d from -i to N-1-i:
            # m = d + N - 1
            # The term involving s[i] in ac[m] is:
            # If we write ac[m] = sum_j s[j] s[j+d], the terms with s[i] are:
            # j=i: s[i] s[i+d] (if i+d valid)
            # j=i-d: s[i-d] s[i] (if i-d valid)
            # When d=0, these coincide
            # Change = -2*old_sign*(s[i+d] if i+d in [0,N) else 0) 
            #          -2*old_sign*(s[i-d] if i-d in [0,N) else 0)
            # + old_sign*(s[i+d] if i+d in [0,N) else 0)  [because old term was old_sign*s[i+d], new is -old_sign*s[i+d]]
            # Wait, let me redo this.
            # Old ac[m] includes term: old_sign * s[i+d] (from j=i) if i+d valid
            # New ac[m] includes term: -old_sign * s[i+d] (from j=i) if i+d valid
            # So change from this term: -2 * old_sign * s[i+d]
            # Similarly, old ac[m] includes term: s[i-d] * old_sign (from j=i-d) if i-d valid
            # New ac[m] includes term: s[i-d] * (-old_sign) = -s[i-d]*old_sign
            # Change: -2 * s[i-d] * old_sign
            # If d=0, i+d = i-d = i, so both terms are the same, and we should only count once
            # But the formula gives -2*old_sign*s[i] - 2*s[i]*old_sign = -4*old_sign*s[i] = -4
            # But the actual change should be: old term is old_sign*s[i] = old_sign^2 = 1
            # New term is (-old_sign)*s[i] = -old_sign^2 = -1
            # Change = -2
            # So we have a factor of 2 error when d=0.
            
            # Let's handle it correctly:
            # For each d, the change in ac[m] where m = d + N - 1 is:
            # Let term1 = 0, term2 = 0
            # if 0 <= i+d < N: term1 = old_sign * signs[i+d]
            # if 0 <= i-d < N: term2 = signs[i-d] * old_sign
            # if d == 0: change = -2 * term1  (since term1 == term2)
            # else: change = -2 * term1 - 2 * term2
            
            # This is getting complex. Let's just recompute ac from scratch for each candidate flip
            # but only if it looks promising. Or use a simpler heuristic.
            
            # Given the time constraints, let's use a simpler approach:
            # Just try random flips and keep the best ones
            
            signs[i] = new_sign
            new_ac = _autoconv(signs)
            new_max = _max_abs_ac(new_ac)
            new_score = float(new_max) / (np.sum(signs) ** 2) if np.sum(signs) != 0 else np.inf
            
            if new_score < current_score:
                current_score = new_score
                ac = new_ac
                current_max = new_max
                improved = True
            else:
                signs[i] = old_sign
                
        if not improved:
            break
            
    return signs
# EVOLVE-BLOCK-END
