"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 256
_DEADLINE = 87.0  # safety margin before 90s kill

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +1/-1 signs."""
    signs = np.ones(n, dtype=np.int32)
    for i in range(1, n):
        # Count trailing ones in binary representation of i
        # Thue-Morse: t(i) = parity of number of 1s in binary i
        # If odd number of 1s, sign is -1
        t = 0
        x = i
        while x:
            t ^= 1
            x &= x - 1
        signs[i] = -1 if t else 1
    return signs

def _compute_ap_data():
    """
    Precompute all arithmetic progressions.
    Returns:
        ap_starts: list of arrays, each containing start indices (0-based) for one AP
        We'll store as: for each (start, step, length), the list of indices.
    For efficiency, we store:
        - For each step d, for each residue r in 0..d-1:
          the sequence of positions r, r+d, r+2d, ... < n
          Any contiguous subsequence of this is an AP with step d.
    
    We return a structure that allows fast sum computation.
    """
    n = _N
    # For each step d from 1 to n-1 (step 1 is just intervals, already covered by contiguous)
    # Actually step 1: APs are just contiguous sequences of length >= 1
    # We need all APs: start s (0-indexed), step d >= 1, length L >= 1, such that s + (L-1)*d < n
    
    # Strategy: for each d, for each residue r mod d, get the sequence seq = signs[r::d]
    # Then all APs with step d and residue r are contiguous subsequences of seq.
    # The max absolute sum over contiguous subsequences of seq can be found via prefix sums.
    
    # We'll precompute the index arrays for each (d, r) group.
    groups = []  # list of (d, r, indices_array)
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            indices = np.arange(r, n, d)
            if len(indices) >= 1:
                groups.append((d, r, indices))
    return groups

def _max_abs_ap_sum(signs, groups):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions.
    signs: numpy array of length n with values in {-1, +1}
    groups: precomputed list of (d, r, indices)
    Returns: max absolute sum (at least 1 due to singletons)
    """
    max_disc = 1
    for d, r, indices in groups:
        seq = signs[indices]
        if len(seq) <= 1:
            continue
        csum = np.cumsum(seq)
        # Max subarray sum and min subarray sum
        # Using Kadane's algorithm on csum: max(csum[j] - min(csum[i]) for i < j)
        # But we want max |sum| over all contiguous subarrays of seq.
        # sum(i..j) = csum[j] - csum[i-1] (with csum[-1] = 0)
        # So max sum = max_j csum[j] - min_{i<j} csum[i-1]
        # Let's prepend 0 to csum for convenience
        csum_with_zero = np.concatenate(([0], csum))
        # Running minimum of csum_with_zero
        running_min = np.minimum.accumulate(csum_with_zero)
        max_sum = np.max(csum_with_zero - running_min)
        
        # For min sum (most negative), we want min of (csum[j] - csum[i-1]) = min csum[j] - max csum[i-1]
        running_max = np.maximum.accumulate(csum_with_zero)
        min_sum = np.min(csum_with_zero - running_max)
        
        disc = max(abs(max_sum), abs(min_sum))
        if disc > max_disc:
            max_disc = disc
    return max_disc

def _local_search(signs, groups, deadline):
    """
    Local search: try flipping individual signs to reduce max AP sum.
    Returns improved signs array.
    """
    n = _N
    current_disc = _max_abs_ap_sum(signs, groups)
    best_signs = signs.copy()
    best_disc = current_disc
    start_time = time.time()
    
    # For faster evaluation, we can try flipping each position and see if it reduces the max.
    # But full re-evaluation is expensive. Let's do a simple hill-climbing with full eval.
    # To speed up, we can use the fact that flipping position p changes sums of all APs containing p.
    
    # Simpler approach: iterate through positions, try flipping, keep if it improves.
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for p in range(n):
            if (time.time() - start_time) > deadline:
                break
            # Try flipping position p
            old_val = signs[p]
            signs[p] = -old_val
            new_disc = _max_abs_ap_sum(signs, groups)
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
                improved = True
            else:
                signs[p] = old_val
        # After a full pass, if no improvement, try random restarts or stop
        if not improved:
            break
    
    return best_signs, best_disc

def _random_signs(seed):
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=_N).astype(np.int32)

def construct():
    start_time = time.time()
    groups = _compute_ap_data()
    
    # Start with Thue-Morse
    best_signs = _thue_morse(_N)
    best_disc = _max_abs_ap_sum(best_signs, groups)
    
    # Also try a few random starts
    for seed in [42, 123, 777]:
        if (time.time() - start_time) > _DEADLINE - 30:
            break
        rs = _random_signs(seed)
        rd = _max_abs_ap_sum(rs, groups)
        if rd < best_disc:
            best_disc = rd
            best_signs = rs
    
    # Local search from best
    if (time.time() - start_time) < _DEADLINE - 10:
        best_signs, best_disc = _local_search(best_signs.copy(), groups, _DEADLINE - 5)
    
    # Try another local search from Thue-Morse if we have time
    if (time.time() - start_time) < _DEADLINE - 10:
        tm = _thue_morse(_N)
        tm_disc = _max_abs_ap_sum(tm, groups)
        if tm_disc <= best_disc:
            tm_signs, tm_disc2 = _local_search(tm.copy(), groups, _DEADLINE - 2)
            if tm_disc2 < best_disc:
                best_signs = tm_signs
                best_disc = tm_disc2
    
    # Convert to list of Python ints
    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
