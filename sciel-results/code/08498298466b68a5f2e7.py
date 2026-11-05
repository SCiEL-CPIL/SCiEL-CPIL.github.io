"""Low discrepancy on all arithmetic progressions - Bitmask Vectorized SA."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

# Precompute all Arithmetic Progressions (APs) for n=256
# An AP is defined by (start, step, length)
# We represent each AP as a bitmask of indices [0, 255]
# Since 256 > 64, we use Python integers (arbitrary precision) or multiple uint64s.
# Python integers are efficient for bitwise ops in loops, but for vectorized numpy 
# we might need to be careful. 
# However, for n=256, we can represent the signs as a vector of +1/-1.
# The sum of an AP is sum(signs[i] for i in AP).
# 
# Strategy:
# 1. Precompute all valid APs as lists of indices or bitmasks.
# 2. To speed up evaluation, we can group APs by step 'd'.
# 3. For a given step d, the APs are just shifted versions of the sequence s[0], s[d], s[2d]...
#    Actually, for a fixed step d, the APs starting at i=0..d-1 are disjoint in terms of 
#    "phase" but overlap in indices? No.
#    AP(start=i, step=d) uses indices i, i+d, i+2d...
#    These are exactly the elements of the subsequence s[i::d].
#    The discrepancy for step d is the max over all i in [0, d-1] of max_abs_prefix_sum(s[i::d]).
#    
#    Wait, the definition says "ANY arithmetic progression".
#    This includes length 1 (sum = +/-1, abs=1).
#    It includes full length.
#    It includes any prefix? "signed sum on ANY arithmetic progression... length including one".
#    Usually "sum on an AP" implies the sum of the terms in that specific AP segment.
#    The prompt says: "Minimize the maximum absolute signed sum on ANY arithmetic progression... any start, positive step and length".
#    This implies we consider every contiguous sub-sequence of the AP? 
#    "signed sum on ANY arithmetic progression" usually means the sum of the elements *of* that progression.
#    If the progression is defined by start, step, length, then the sum is fixed.
#    So we check all (start, step, length) triplets.
#    
#    Let's re-read carefully: "maximum absolute signed sum on ANY arithmetic progression contained in [1,256]".
#    An arithmetic progression is a set of numbers. The sum is the sum of signs at those indices.
#    So for each valid (start, step, length), we compute S = sum(signs[start + k*step] for k in 0..length-1).
#    We want min max |S|.
#    
#    Optimization:
#    For a fixed step d, the possible APs are determined by start i (0 <= i < d) and length L (1 <= L <= max_len).
#    The sum for start i and length L is the prefix sum of the sequence s[i::d] up to index L-1.
#    So for each d, we compute the prefix sums for each residue class i mod d.
#    Then we take the max absolute value across all these prefix sums.
#    This is exactly what the current program does, but we can vectorize the *search* part.
#
#    The bottleneck in the current program is the Python loop over d and i, and the numpy cumsum inside.
#    We can precompute the indices for each (d, i) pair.
#    There are sum(d for d in 1..256) = 256*257/2 = 32896 such sequences.
#    Storing indices for all of them is 32896 * avg_len. Avg len is roughly 256/(2d)? No.
#    Total number of (d, i, L) triples is sum_{d=1}^{256} sum_{i=0}^{d-1} floor((255-i)/d) + 1.
#    This is large.
#
#    Alternative Vectorization for Evaluation:
#    Instead of iterating d, we can iterate over all possible lengths? No.
#    
#    Let's stick to the structure:
#    Precompute a list of arrays `ap_indices[d][i]` containing the indices for start=i, step=d.
#    Actually, `np.arange(i, 256, d)` is fast.
#    
#    The real speedup for Simulated Annealing comes from being able to evaluate many candidates.
#    But each evaluation is O(N^2) roughly (sum of lengths).
#    N=256. Sum of lengths for all APs is approx N^2/2 * log N?
#    Let's estimate cost of one evaluation.
#    Current code:
#    for d in 1..256:
#       for i in 0..d-1:
#          cumsum(s[i::d])
#    Total operations ~ sum_{d=1}^{N} d * (N/d) = N * N = N^2 = 65536 operations (vectorized cumsum).
#    This is quite fast in NumPy.
#    
#    Why is the current program slow or stuck?
#    It's greedy. It gets stuck in local minima.
#    Simulated Annealing can jump out.
#    
#    To make SA fast, we need to evaluate quickly.
#    We can use a "delta" evaluation if we flip one bit?
#    Flipping sign at index k changes the sum of every AP containing k.
#    How many APs contain k?
#    For each step d, k is in APs starting at i = k % d? No.
#    k is in AP(start=i, step=d) if (k-i) % d == 0 and i <= k.
#    Also length must be sufficient.
#    
#    Actually, full re-evaluation is likely fast enough for N=256 in Python/NumPy if optimized.
#    Let's check: 65k cumsums of small arrays.
#    Maybe we can precompute the *structure* to avoid creating `s[i::d]` slices repeatedly?
#    No, slicing is a view, cumsum does the work.
#    
#    Let's implement a robust Simulated Annealing with a very fast evaluation function.
#    We will also use a "restart" strategy if SA stagnates.

def _precompute_ap_structures(n):
    """
    Precompute the indices for all (start, step) pairs.
    Returns a list of lists of arrays.
    structures[d][i] = np.array([i, i+d, i+2d, ...])
    """
    structures = []
    for d in range(1, n + 1):
        level = []
        for i in range(d):
            if i >= n:
                break
            idx = np.arange(i, n, d)
            level.append(idx)
        structures.append(level)
    return structures

# Global cache for structures
_STRUCTURES = None

def _get_structures(n):
    global _STRUCTURES
    if _STRUCTURES is None or _STRUCTURES[0] != n:
        _STRUCTURES = (n, _precompute_ap_structures(n))
    return _STRUCTURES[1]

def _fast_max_ap_discrepancy(signs, structures):
    """
    Compute max absolute sum over all APs.
    signs: numpy array of int16 (-1, 1)
    structures: precomputed list of lists of index arrays
    """
    max_disc = 0
    # signs is int16, cumsum will be int64 or similar, safe.
    for d_level in structures:
        for idx_arr in d_level:
            # Extract signs
            s_sub = signs[idx_arr]
            # Cumulative sum
            csum = np.cumsum(s_sub)
            # Max absolute
            m = np.max(np.abs(csum))
            if m > max_disc:
                max_disc = m
    return max_disc

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    n = 256
    
    structures = _get_structures(n)
    
    # Initial random signs
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    
    current_disc = _fast_max_ap_discrepancy(signs, structures)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Simulated Annealing Parameters
    # Start temperature high, cool down
    T = 5.0
    cooling = 0.995
    min_T = 0.01
    
    # We will also try a "greedy polish" at the end or periodically.
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with a slightly perturbed best solution or random?
            # Perturb best solution
            signs = best_signs.copy()
            # Flip 5 random bits
            flips = rng.choice(n, size=5, replace=False)
            signs[flips] *= -1
            current_disc = _fast_max_ap_discrepancy(signs, structures)
            T = 2.0 # Reset temperature
            
        # Generate candidate: flip one random bit
        idx = rng.integers(0, n)
        candidate = signs.copy()
        candidate[idx] *= -1
        
        # Evaluate candidate
        cand_disc = _fast_max_ap_discrepancy(candidate, structures)
        
        # Acceptance probability
        if cand_disc < current_disc:
            # Accept
            signs = candidate
            current_disc = cand_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
                # If we hit the theoretical minimum (1), we can stop?
                # Min possible is 1 (length 1 APs).
                if best_disc <= 1:
                    break
        else:
            # Metropolis criterion
            delta = cand_disc - current_disc
            # P = exp(-delta / T)
            # delta is integer >= 1
            # If T is small, exp is tiny.
            # Optimization: if delta > 10*T, prob is effectively 0.
            if delta < 20 * T:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    signs = candidate
                    current_disc = cand_disc
        
        # Cool down
        T *= cooling
        
        # Periodic Greedy Polish
        # If we haven't improved in a while, do a few greedy steps?
        # Or just rely on SA.
        
    # Final Greedy Polish
    # Try to fix the worst APs
    time_left = deadline - time.time()
    if time_left > 5.0:
        signs = best_signs.copy()
        current_disc = best_disc
        polish_deadline = time.time() + time_left - 2.0
        
        while time.time() < polish_deadline:
            # Find worst AP
            # We need a function that returns the worst AP details
            # For now, just re-evaluate and if we can't improve, break.
            # Since we don't have the "worst AP" detail in _fast_max_ap_discrepancy,
            # we'll do a simple random restart polish or just return.
            # Actually, let's implement a quick greedy repair for the best solution found.
            
            # To do greedy repair, we need to find the specific AP.
            # Let's modify _fast_max_ap_discrepancy to optionally return details?
            # Or just write a separate function for the polish phase.
            
            break # Skip complex polish for now, SA should be good enough with restarts.
            
    return best_signs.tolist()

# EVOLVE-BLOCK-END
