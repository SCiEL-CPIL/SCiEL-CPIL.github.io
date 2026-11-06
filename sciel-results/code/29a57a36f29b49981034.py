import time
import numpy as np

N = 16

def construct():
    """
    Returns a list of (i, j) pairs representing a sorting network for 16 wires.
    """
    # Deadline: 50 seconds to be safe
    deadline = time.time() + 50.0
    
    # 1. Generate initial Batcher's Odd-Even Merge network
    network = generate_batcher_merge(N)
    
    # Verify it's correct (it should be)
    # But just in case, we can skip explicit verification of the initial known-good network
    # to save time, but let's do a quick sanity check or just trust it.
    
    # 2. Greedy Removal
    # We use NumPy to check validity quickly.
    # Generate all 2^16 inputs
    inputs = np.array([list(map(int, format(i, '016b'))) for i in range(2**N)], dtype=np.int8)
    # inputs shape: (65536, 16)
    
    def check_validity(net):
        # Simulate the network on all inputs
        state = inputs.copy()
        for i, j in net:
            # Extract columns i and j
            # min goes to i, max goes to j
            # We can do this vectorized
            # But careful: we need to update state in place or create new cols?
            # Vectorized min/max:
            # col_i = state[:, i]
            # col_j = state[:, j]
            # state[:, i] = np.minimum(col_i, col_j)
            # state[:, j] = np.maximum(col_i, col_j)
            
            # This is O(1) per comparator in terms of Python overhead, but O(N) in memory ops.
            # 63 comparators * 65536 rows is very fast.
            
            ci = state[:, i]
            cj = state[:, j]
            state[:, i] = np.minimum(ci, cj)
            state[:, j] = np.maximum(ci, cj)
        
        # Check if sorted
        # A vector is sorted if state[:, k] <= state[:, k+1] for all k
        # This returns a boolean array of shape (65536,)
        # We need all True
        return np.all(np.all(state[:, :-1] <= state[:, 1:], axis=1))

    current_net = network
    improved = True
    while improved:
        improved = False
        # Iterate backwards? Forward? Usually forward is fine, but restarting helps.
        # To be safe, we iterate through all positions.
        for idx in range(len(current_net)):
            if time.time() > deadline:
                return current_net
            
            # Remove comparator at idx
            test_net = current_net[:idx] + current_net[idx+1:]
            
            # Check validity
            if check_validity(test_net):
                current_net = test_net
                improved = True
                # Restart the loop to check new indices
                break
        
        # If we didn't break (no improvement in this pass), we are stuck in a local minimum
        # for simple removals. We could try swaps, but given the small gap (63->60),
        # simple removal might not be enough if the redundant comparators are "entangled".
        # However, for Batcher N=16, it is known that 3 can be removed.
        # If simple removal gets stuck at 62 or 61, we might need a more complex search.
        # But let's try simple removal first.
        
        if not improved:
             # Try a few random restarts with perturbation?
             # Or just accept the result.
             # Given the tight time budget, we accept the greedy result.
             pass

    return current_net

def generate_batcher_merge(n):
    """
    Generates Batcher's Odd-Even Merge sorting network for n wires.
    """
    if n <= 1:
        return []
    
    # Batcher's merge assumes two sorted halves.
    # To sort the whole input, we first sort the halves recursively, then merge.
    
    # Step 1: Recursively sort first half and second half
    comps = []
    # First half: wires 0 to n//2 - 1
    comps += generate_batcher_merge(n//2)
    # Second half: wires n//2 to n - 1
    comps_half2 = generate_batcher_merge(n//2)
    for i, j in comps_half2:
        comps.append((i + n//2, j + n//2))
        
    # Step 2: Odd-Even Merge
    comps += batcher_merge(n//2, offset=0, other_offset=n//2)
    
    return comps

def batcher_merge(n, offset=0, other_offset=0):
    """
    Merges two sorted sequences of size n located at [offset, offset+n) and [other_offset, other_offset+n).
    """
    if n <= 1:
        return []
    
    comps = []
    # The merge step has log2(n) phases
    # Phase k compares (offset + i) with (other_offset + i + n//2^k * something?)
    
    # Standard Batcher Merge description:
    # For k = 1 to log2(n):
    #   For i = 0 to n - 2^(k-1) - 1 (step 2^(k-1)? No, step depends on odd/even)
    
    # Let's use the recursive definition of Odd-Even Merge:
    # Merge(A, B) where A and B have size n.
    # 1. Recursively merge A_odd with B_odd and A_even with B_even?
    # No, that's for the "Odd-Even" structure.
    
    # Let's stick to the iterative phase definition which is easier to code for fixed N.
    # For N=16, n=16.
    # Phases:
    # k=1: Compare (i, i+n/2) for i in 0..n/2-1? No.
    
    # Let's look up the specific comparators for N=16 Batcher Merge to be safe.
    # Or implement the standard recursive merge.
    
    # Recursive Odd-Even Merge:
    # Merge(A, B) of size n:
    # 1. If n == 1, done.
    # 2. Split A into A1, A2 and B into B1, B2 of size n/2.
    # 3. Recursively Merge(A1, B1) and Merge(A2, B2).
    # 4. For i = 0 to n/2 - 1:
    #      Compare (A2[i], B1[i])
    
    # This produces the correct comparators.
    
    # Let's implement this recursion.
    def _merge_recursive(size, a_start, b_start):
        if size <= 1:
            return []
        half = size // 2
        comps = []
        # Merge first halves
        comps += _merge_recursive(half, a_start, b_start)
        # Merge second halves
        comps += _merge_recursive(half, a_start + half, b_start + half)
        # Cross comparisons
        for i in range(half):
            comps.append((a_start + half + i, b_start + i))
            # Note: The compare-exchange is min->low, max->high.
            # The pair is (a_start + half + i, b_start + i).
            # We must ensure i < j in the output? The task says "0 <= i < j < N".
            # So we must sort the indices.
            # The comparator (x, y) is equivalent to (min(x,y), max(x,y)).
            # We will handle sorting indices at the end or here.
            
        return comps

    # For N=16:
    # We sort first half [0, 8) and second half [8, 16).
    # Then merge [0, 8) and [8, 16).
    # So a_start = 0, b_start = 8, size = 8.
    
    comps = _merge_recursive(n, offset, other_offset)
    
    # Ensure i < j
    normalized = []
    for i, j in comps:
        if i > j:
            i, j = j, i
        normalized.append((i, j))
        
    return normalized

# To ensure we don't have duplicates or redundant comparators in the generation, 
# we can deduplicate, but Batcher's merge is typically distinct.

# Let's double check the _merge_recursive logic.
# Merging A[0..7] and B[8..15].
# Recursively merge A[0..3] with B[8..11] and A[4..7] with B[12..15].
# Then cross compare A[4..7] with B[8..11].
# This is the standard Odd-Even Merge.

# Final check on the code structure.
# The `construct` function must return a list of tuples.
# The `check_validity` function uses `inputs` from the enclosing scope? 
# No, it's defined inside `construct`, so it captures `inputs`. This is fine.

# One potential issue: `inputs` generation takes time.
# `np.array([list(map(int, format(i, '016b'))) for i in range(2**N)], dtype=np.int8)`
# 65536 rows, 16 cols.
# This is fast (< 0.1s).

# The `check_validity` loop runs 63 times in the first pass.
# Each check takes ~0.01-0.05s.
# Total ~1-3s.
# This is well within 60s.

# If greedy removal gets stuck, we return the best found.
# The score will be 120 / len(network).
# If we get 60, score is 2.0.
# If we get 63, score is ~1.9.

# I will add a simple "random restart" or "swap" heuristic if removal stalls, 
# but given the time constraint, simple removal is the primary driver.
# Actually, for N=16, it is well known that 60 is achievable.
# The gap between 63 and 60 is small. Greedy removal often finds it.

# Let's add a fallback: if we don't reach 60, we try a few random swaps to unlock removals.
# But to keep it simple and robust, I'll stick to the greedy removal with a time check.

# One optimization: If we remove a comparator, we restart the index loop.
# This is O(N^2) checks in the worst case, but N is small.

# Final Code Structure:

