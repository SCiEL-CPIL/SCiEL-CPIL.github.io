"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 85  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _make_all_inputs():
    """Create array of all 2^N Boolean inputs as (NUM_INPUTS, N) uint8 array."""
    bits = np.arange(NUM_INPUTS, dtype=np.uint64)
    arr = np.empty((NUM_INPUTS, N), dtype=np.uint8)
    for k in range(N):
        arr[:, k] = (bits >> k) & 1
    return arr

def _is_sorting_network(comparators, input_matrix):
    """Check if network sorts all inputs using zero-one principle."""
    if not comparators:
        # Empty network only sorts if input is already sorted (it's not for all inputs)
        # But we only call this on valid candidates derived from a base
        pass
    wires = input_matrix.copy()
    for i, j in comparators:
        # min to i, max to j
        # Since values are 0 or 1:
        # min = AND, max = OR
        new_i = wires[:, i] & wires[:, j]
        new_j = wires[:, i] | wires[:, j]
        wires[:, i] = new_i
        wires[:, j] = new_j
    
    # Check if each row is sorted (non-decreasing)
    # A row is sorted if all adjacent pairs satisfy wires[:, k] <= wires[:, k+1]
    # Since values are 0/1, this means no 1 followed by 0.
    for k in range(N - 1):
        if np.any(wires[:, k] > wires[:, k + 1]):
            return False
    return True

def _try_remove(comparators, idx, input_matrix):
    """Try removing comparator at idx, return True if still valid."""
    # Optimization: Instead of copying the whole list every time, 
    # we can simulate the removal. But for 100-200 comparators, list slicing is fast enough.
    # The bottleneck is _is_sorting_network.
    new_comps = comparators[:idx] + comparators[idx+1:]
    return _is_sorting_network(new_comps, input_matrix)

def construct():
    start_time = time.time()
    
    # Base: odd-even transposition
    # For N=17, this generates 17 phases.
    # Phase 0: (0,1), (2,3), ..., (14,15) -> 8 comparators
    # Phase 1: (1,2), (3,4), ..., (15,16) -> 8 comparators
    # ...
    # Total comparators = N * (N-1)/2? No.
    # Let's just generate it.
    comps = [(i, i + 1) for phase in range(N)
             for i in range(phase % 2, N - 1, 2)]
    
    input_matrix = _make_all_inputs()
    
    # Verify base is correct
    if not _is_sorting_network(comps, input_matrix):
        # Fallback: should not happen, but return the base if it's somehow broken (it won't be)
        return comps
    
    # Iteratively remove redundant comparators
    # We use a greedy approach: try to remove each comparator from 0 to end.
    # If removal keeps the network valid, remove it and restart the scan from the beginning.
    # This is O(N^2 * T_verify) worst case, but T_verify is the main cost.
    # With N~100 and T_verify ~ 1ms (numpy), this is ~100ms per full pass.
    # We can do many passes in 85s.
    
    # To speed up verification, we can cache results? 
    # Not easily, because the state of the network changes.
    # However, we can optimize the verification itself.
    # The current _is_sorting_network is already vectorized.
    
    # Another optimization: 
    # If a comparator (i, j) is such that i and j are not "adjacent" in terms of 
    # potential dependency, it might be easier to remove? 
    # Actually, the order matters.
    
    # Let's add a simple heuristic: try removing comparators that are "far apart" first?
    # Or just stick with sequential. Sequential is fine.
    
    # To improve speed, we can limit the number of checks per second? 
    # No, we want to maximize removals.
    
    # One significant speedup: 
    # Instead of checking ALL 131072 inputs, can we use a smaller set? 
    # The zero-one principle says we need all. 
    # But maybe we can use a random subset for a quick filter, then verify with full set?
    # If it fails on subset, it's definitely invalid.
    # If it passes on subset, it might still be invalid, so we must verify with full set.
    # This doesn't help much if most removals are invalid.
    
    # Let's stick to the full verification but optimize the loop.
    
    changed = True
    while changed and (time.time() - start_time) < DEADLINE - 2:
        changed = False
        for idx in range(len(comps)):
            if (time.time() - start_time) > DEADLINE - 2:
                break
            if _try_remove(comps, idx, input_matrix):
                comps = comps[:idx] + comps[idx+1:]
                changed = True
                break  # Restart loop since indices changed
    
    return comps

# EVOLVE-BLOCK-END
