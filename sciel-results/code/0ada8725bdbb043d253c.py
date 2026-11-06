"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def _verify_network_fast(comps, n=N):
    """
    Faster verification using numpy vectorization.
    """
    num_inputs = 1 << n
    # Create all input vectors as a 2D array: shape (num_inputs, n)
    # Each row is a binary vector
    all_vecs = np.zeros((num_inputs, n), dtype=np.uint8)
    for i in range(n):
        all_vecs[:, i] = (np.arange(num_inputs) >> i) & 1
    
    # Apply each comparator
    for (i, j) in comps:
        # Where vec[i] > vec[j], swap
        mask = all_vecs[:, i] > all_vecs[:, j]
        # Swap values where mask is True
        tmp = all_vecs[mask, i].copy()
        all_vecs[mask, i] = all_vecs[mask, j]
        all_vecs[mask, j] = tmp
    
    # Check if all rows are sorted
    # A row is sorted if all_vecs[:, i] <= all_vecs[:, i+1] for all i
    for i in range(n - 1):
        if np.any(all_vecs[:, i] > all_vecs[:, i + 1]):
            return False
    
    return True

def _initial_network():
    """
    Generate an initial correct sorting network for 18 wires.
    Uses a simple but correct approach: Odd-even transposition sort.
    """
    comps = []
    # Use a series of odd-even transposition phases
    # This gives a correct but suboptimal network (153 comparators for N=18)
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _remove_redundant_comparators(comps, n=N, time_limit=50):
    """
    Iteratively remove redundant comparators from the network.
    A comparator is redundant if removing it still leaves the network correct.
    """
    start_time = time.time()
    current = list(comps)
    
    # Try removing each comparator, starting from the end
    # Use a greedy approach: try to remove each one
    changed = True
    while changed:
        changed = False
        # Iterate from end to start to maintain index stability during removal
        for idx in range(len(current) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            
            # Try removing comparator at idx
            test = current[:idx] + current[idx+1:]
            
            # Verify
            if _verify_network_fast(test, n):
                current = test
                changed = True
        
        if time.time() - start_time > time_limit:
            break
    
    return current

def construct():
    """
    Construct a sorting network for 18 wires.
    Returns a list of (i, j) comparators.
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for the 90s limit (evaluator kills at 90s)
    
    # Generate initial network
    comps = _initial_network()
    
    # Remove redundant comparators
    # We give it most of the time budget.
    # The verification is O(2^N * N) per check, which is fast enough for ~100-200 comparators.
    # With N=18, 2^18 = 262144. 262144 * 18 ops is ~5M ops. Numpy handles this in <1ms.
    # So we can do many checks.
    comps = _remove_redundant_comparators(comps, N, time_limit=80)
    
    # Verify the final network
    if not _verify_network_fast(comps, N):
        # Fallback to the initial network
        comps = _initial_network()
    
    return comps
# EVOLVE-BLOCK-END
