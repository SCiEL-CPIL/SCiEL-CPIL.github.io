"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        # Generate all 2^N binary inputs as a numpy array of shape (2^N, N)
        # Using integer arithmetic for speed
        idx = np.arange(1 << N)
        _ALL_INPUTS = (idx[:, None] >> np.arange(N)) & 1
    return _ALL_INPUTS

def _verify_network(net):
    """
    Verify if the network sorts all 2^N Boolean inputs using NumPy vectorization.
    Returns True if valid, False otherwise.
    """
    vals = _get_all_inputs().copy()
    
    for i, j in net:
        # Extract columns
        vi = vals[:, i]
        vj = vals[:, j]
        
        # Compute min and max
        min_val = np.minimum(vi, vj)
        max_val = np.maximum(vi, vj)
        
        # Update columns
        vals[:, i] = min_val
        vals[:, j] = max_val
        
    # Check if all rows are sorted
    # We can check if any adjacent pair is inverted
    # for k in range(N - 1):
    #     if np.any(vals[:, k] > vals[:, k+1]):
    #         return False
    # return True
    
    # Vectorized check for sortedness
    # diffs = np.diff(vals, axis=1)
    # return np.all(diffs >= 0)
    
    # Manual loop is often faster for small N in Python/Numpy boundary overhead
    for k in range(N - 1):
        if np.any(vals[:, k] > vals[:, k+1]):
            return False
    return True

def _bitonic_merge(n):
    """
    Generate bitonic merge network for n wires.
    Assumes input is a bitonic sequence (first half ascending, second half descending).
    Returns list of (i, j) comparators.
    """
    if n <= 1:
        return []
    
    half = n // 2
    # Compare (i, i + half) for i in range(half)
    # This splits the bitonic sequence into two bitonic sequences of size half
    # One will be all mins (ascending), one all maxes (descending) ?
    # Actually, after comparing (i, i+half), the first half contains the smaller elements
    # and the second half contains the larger elements.
    # The first half is bitonic (ascending then descending? No, it's just sorted ascending? No.)
    # Standard bitonic merge:
    # 1. Compare (i, i+n/2)
    # 2. Recursively merge the first half and the second half.
    
    comparators = [(i, i + half) for i in range(half)]
    
    # Recursively merge the two halves
    # The first half (0 to half-1) is now a bitonic sequence?
    # Yes, the standard bitonic merge algorithm works recursively on the halves.
    left_merge = _bitonic_merge(half)
    right_merge = _bitonic_merge(n - half)
    
    # Shift right merge indices
    right_merge_shifted = [(i + half, j + half) for i, j in right_merge]
    
    return comparators + left_merge + right_merge_shifted

def _bitonic_sort(n):
    """
    Generate bitonic sort network for n wires.
    """
    if n <= 1:
        return []
    
    half = n // 2
    # Sort first half ascending
    left_sort = _bitonic_sort(half)
    
    # Sort second half descending
    # To sort descending, we can sort ascending and then reverse the comparators?
    # Or generate a descending sort network.
    # A simpler way: Generate ascending sort for the second half, but swap the inputs?
    # Actually, bitonic sort requires the second half to be sorted in descending order.
    # We can achieve this by generating a standard bitonic sort for the second half,
    # but the definition of "sorted" in the merge step assumes the direction.
    # Standard algorithm:
    # 1. Sort first half ascending.
    # 2. Sort second half descending.
    # 3. Merge.
    
    # To sort descending, we can just take the ascending sort network and swap i, j in all comparators?
    # No, compare-exchange is symmetric in logic (min to i, max to j).
    # If we want the result to be descending, we want max at i and min at j?
    # The task says: "compare-exchange writes min to wire i and max to j".
    # So (i, j) always puts min at i.
    # To sort a list in descending order using min-to-i comparators,
    # we effectively need to sort it ascending and then reverse the wires?
    # Or we can just generate the network for ascending sort, and then 
    # the "descending" requirement for the merge is handled by the structure.
    
    # Let's look at the standard Bitonic Sort algorithm for N=4:
    # (0,1), (2,3) -> Sort pairs
    # (0,2), (1,3) -> Merge? No.
    
    # Correct Bitonic Sort for N=4:
    # Stage 1: (0,1), (2,3)
    # Stage 2: (0,2), (1,3)
    # Stage 3: (1,2)
    
    # Let's stick to the recursive definition:
    # bitonic_sort(n):
    #   if n==1: []
    #   left = bitonic_sort(n/2)
    #   right = bitonic_sort(n-n/2) but reversed?
    
    # Actually, for the second half to be descending, we can generate a standard
    # ascending sort network for that half, but then the values will be ascending.
    # The merge step expects the second half to be descending.
    # So we must generate a network that sorts the second half in *descending* order.
    # How to generate a descending sort network with min-to-i comparators?
    # If we want wire 0 to have the max, and wire n-1 to have the min,
    # we can just take the ascending sort network and reverse the indices?
    # Let's just generate the ascending network for the right half, and then
    # we will have an ascending sequence. The merge algorithm `_bitonic_merge`
    # expects a bitonic sequence (asc then desc).
    # If we feed it (asc, asc), it is not bitonic.
    
    # So we MUST sort the right half in descending order.
    # To sort in descending order using (i, j) where min->i:
    # We want the largest element at the smallest index.
    # If we run a standard ascending sort on wires [half, n), the smallest element
    # will be at `half`. We want the largest element at `half`.
    # So we can generate the ascending sort network for the right half,
    # and then swap the indices in the comparators?
    # If we swap (i, j) to (j, i), min goes to j (larger index) and max to i (smaller index).
    # This effectively sorts the sub-array in descending order!
    
    right_sort_asc = _bitonic_sort(n - half)
    # Shift and reverse to make it descending
    right_sort_desc = [(j + half, i + half) for i, j in right_sort_asc]
    
    # Merge
    merge = _bitonic_merge(n)
    
    return left_sort + right_sort_desc + merge

def construct():
    # Generate initial network
    net = _bitonic_sort(N)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    # Local Search: Greedy Removal
    # We try to remove comparators one by one if the network remains valid.
    # We repeat this process until no more can be removed or time runs out.
    
    while time.time() < deadline:
        improved = False
        # Iterate through the network in reverse to handle index shifting if we remove
        # Or just iterate forward and break on first improvement, then restart.
        # Restarting is simpler and safer.
        for idx in range(len(net)):
            if time.time() > deadline:
                break
            
            # Try removing net[idx]
            candidate = net[:idx] + net[idx+1:]
            
            # Verify
            if _verify_network(candidate):
                net = candidate
                improved = True
                break
        
        if not improved:
            # No single comparator can be removed.
            # We could try swapping, but removal is the primary driver for improvement
            # from a non-optimal seed.
            # If we are stuck, we break to save time.
            break
            
    return net
# EVOLVE-BLOCK-END
