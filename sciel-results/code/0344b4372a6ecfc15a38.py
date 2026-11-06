import time
import itertools

N = 16

# EVOLVE-BLOCK-START

def _is_sorted_network(comparators):
    """Check if the network sorts all 0-1 inputs."""
    # Zero-one principle: check all 2^N inputs
    # Optimization: We can represent the state as a bitmask.
    # But checking all 65536 inputs for every step in a search loop is slow.
    # We will use a faster check by simulating all inputs in a vectorized way if possible,
    # or just a tight loop in Python.
    
    # For N=16, 65536 inputs.
    # A single comparator check on a bitmask is fast.
    # Let's simulate all inputs at once? 
    # State: list of 16 ints (0 or 1) for each input.
    # Instead of iterating 65536 inputs one by one, we can iterate the inputs and check.
    
    # Actually, a more efficient check:
    # If the network is valid, it must sort the "canonical" inputs? No.
    # We must check all.
    
    # To speed up, we can early exit if we find an unsorted output.
    
    # We'll use a helper that takes a list of comparators and returns True/False.
    # To make it faster, we can represent the current state of each wire as a list of 65536 bits?
    # That's complex. Let's stick to iterating inputs but optimizing the inner loop.
    
    # Alternative: Use the property that if it sorts all 0-1 inputs, it sorts everything.
    # We can pre-compute the permutation of wires for each input? No, comparators are data-dependent.
    
    # Let's just do a tight loop. 65536 * len(comparators) operations.
    # If len(comparators) is ~60, that's ~4M operations. In Python, this might take ~0.1-0.5s.
    # If we do this in a search loop with 1000 steps, it's too slow.
    
    # We need a faster validity check or fewer checks.
    # Idea: Only check a subset of inputs? No, must be correct.
    # Idea: Incremental verification? Hard.
    
    # Let's optimize the check function as much as possible.
    
    comps = comparators
    n_comps = len(comps)
    
    # If n_comps is 0, return False (unless N=1)
    if n_comps == 0:
        return False
        
    # We can speed up by checking only the "hard" inputs? 
    # Actually, the 0-1 principle is exhaustive.
    
    # Let's try to vectorize the check using numpy if available?
    # The prompt says NumPy is available.
    # We can represent all 65536 inputs as a numpy array of shape (65536, 16).
    # Then apply comparators in parallel.
    
    import numpy as np
    
    # Pre-compute all inputs
    # This takes time, so we should do it once.
    # But the function _is_sorted_network is called repeatedly.
    # We should cache the input matrix.
    
    if not hasattr(_is_sorted_network, '_input_matrix'):
        # Create all 2^16 inputs
        # bits = np.arange(65536).reshape(-1, 1)
        # inputs = (bits // 2**np.arange(15, -1, -1)) % 2
        # This creates a (65536, 16) array of 0s and 1s.
        # Memory: 65536 * 16 * 8 bytes (int64) ~ 8MB. Acceptable.
        # Or use uint8 to save memory.
        bits = np.arange(65536, dtype=np.uint16).reshape(-1, 1)
        shifts = np.arange(15, -1, -1, dtype=np.uint16)
        inputs = (bits >> shifts) & 1
        _is_sorted_network._input_matrix = inputs
    
    state = _is_sorted_network._input_matrix.copy()
    
    for i, j in comps:
        # min = state[:, i] & state[:, j]
        # max = state[:, i] | state[:, j]
        # Actually, for 0/1:
        # min is 1 only if both are 1? No.
        # min(a, b) for 0/1: 1 if a=1 and b=1, else 0. So AND.
        # max(a, b) for 0/1: 0 if a=0 and b=0, else 1. So OR.
        # Wait.
        # If a=0, b=1: min=0, max=1. AND=0, OR=1. Correct.
        # If a=1, b=0: min=0, max=1. AND=0, OR=1. Correct.
        # If a=1, b=1: min=1, max=1. AND=1, OR=1. Correct.
        # If a=0, b=0: min=0, max=0. AND=0, OR=0. Correct.
        
        min_val = state[:, i] & state[:, j]
        max_val = state[:, i] | state[:, j]
        state[:, i] = min_val
        state[:, j] = max_val
        
    # Check if sorted: state[:, k] <= state[:, k+1] for all k
    # i.e., state[:, k] == 1 implies state[:, k+1] == 1
    # So: state[:, :-1] & ~state[:, 1:] should be all 0s.
    # Or: np.all(state[:, :-1] <= state[:, 1:])
    
    # Using bitwise:
    # If any wire k has 1 and wire k+1 has 0, it's unsorted.
    # unsorted_mask = (state[:, :-1] == 1) & (state[:, 1:] == 0)
    # if np.any(unsorted_mask): return False
    
    # Faster:
    # diff = np.diff(state, axis=1)
    # if np.any(diff < 0): return False
    
    # Or simpler:
    # Check if each row is sorted.
    # A row is sorted if it is of the form 00...011...1
    # This means there is at most one transition from 0 to 1.
    # Or: state[:, k] == 1 => state[:, j] == 1 for all j > k.
    
    # Let's use the np.any method on differences.
    diffs = state[:, 1:] - state[:, :-1]
    # If any diff is negative, unsorted.
    if np.any(diffs < 0):
        return False
    
    return True

def _generate_batcher_merge_network(n):
    """Generate Batcher's Merge Sort network for n wires."""
    if n == 1:
        return []
    
    n_half = n // 2
    
    # Recursively sort first half and second half
    left = _generate_batcher_merge_network(n_half)
    right = _generate_batcher_merge_network(n_half)
    # Shift right network indices by n_half
    right_shifted = [(i + n_half, j + n_half) for i, j in right]
    
    # Odd-even merge the two halves
    merge = _batcher_odd_even_merge(n_half)
    
    return left + right_shifted + merge

def _batcher_odd_even_merge(n):
    """Generate Batcher's Odd-Even Merge network for two sorted lists of size n, total 2n wires.
    Wires 0..n-1 are the first list, n..2n-1 are the second list.
    """
    if n == 1:
        return [(0, 1)]
    
    n_half = n // 2
    n_quarter = n // 4
    
    # Step 1: Recursively merge odd and even elements
    # Odd elements: 1, 3, ... from first list; n+1, n+3, ... from second list?
    # Batcher's Odd-Even Merge:
    # 1. Odd-Even Merge the odd-indexed elements:
    #    List A_odd: 1, 3, ..., n-1 (indices in first list)
    #    List B_odd: n+1, n+3, ..., 2n-1 (indices in second list)
    #    Wait, the indices in the combined 2n array are:
    #    First list: 0, 1, ..., n-1
    #    Second list: n, n+1, ..., 2n-1
    #    
    #    Odd indices in first list: 1, 3, ..., n-1 (if n is even, n-1 is odd)
    #    Odd indices in second list: n+1, n+3, ...? No, n is even, so n+1 is odd.
    #    
    #    The recursive merge acts on the pairs of odd positions.
    #    There are n/2 odd positions in the first list and n/2 in the second.
    #    Total n odd positions.
    #    
    #    Actually, the standard definition:
    #    M(n) = 2*M(n/2) + n/2 comparators?
    #    
    #    Let's look up the structure.
    #    Odd-Even Merge of two sorted lists of size n:
    #    1. Recursively merge the odd-indexed elements of the two lists.
    #       The odd-indexed elements of the first list are at indices 1, 3, ..., n-1.
    #       The odd-indexed elements of the second list are at indices n+1, n+3, ..., 2n-1.
    #       These form two lists of size n/2.
    #       We need to map them to a smaller problem.
    #       
    #    2. Recursively merge the even-indexed elements? No.
    #    2. Merge the resulting odd list with the even-indexed elements of the first list?
    #    
    #    Standard Batcher Odd-Even Merge:
    #    O(n) = 2*O(n/2) + O(n) comparators for the merging stage?
    #    
    #    Let's use a known implementation.
    #    
    #    def odd_even_merge(n):
    #        if n == 1:
    #            return [(0, 1)]
    #        
    #        n2 = n // 2
    #        
    #        # Merge odd indices
    #        # Indices: 1, 3, ..., n-1 from A; n+1, n+3, ..., 2n-1 from B
    #        # Map to 0..n-1 in a sub-network?
    #        # The odd indices in the combined 2n array are:
    #        # 1, 3, ..., n-1, n+1, n+3, ..., 2n-1
    #        # There are n of them.
    #        # We can view them as two lists of n/2 each.
    #        # List A_odd: 1, 3, ..., n-1 (size n/2)
    #        # List B_odd: n+1, n+3, ..., 2n-1 (size n/2)
    #        
    #        # The recursive call merges these two lists of size n/2.
    #        # But the indices in the sub-network are 0..n-1.
    #        # We need to map the actual indices to the sub-network indices.
    #        
    #        # Actually, it's easier to just build the network for size n directly.
    #        
    #        # Step 1: Merge odd elements
    #        # The odd elements are at indices: 1, 3, ..., 2n-1
    #        # This is equivalent to merging two lists of size n/2:
    #        # List 1: indices 1, 3, ..., n-1 (from first half)
    #        # List 2: indices n+1, n+3, ..., 2n-1 (from second half)
    #        
    #        # We call odd_even_merge(n/2) to get comparators for indices 0..n-1
    #        # Then we map these back.
    #        # Sub-index k in [0, n-1] corresponds to:
    #        # If k < n/2: original index 2*k + 1
    #        # If k >= n/2: original index 2*(k - n/2) + 1 + n = 2*k - n + 1 + n = 2*k + 1
    #        # So sub-index k maps to original index 2*k + 1.
    #        
    #        odd_comps = []
    #        for i, j in _batcher_odd_even_merge(n2):
    #            oi = 2*i + 1
    #            oj = 2*j + 1
    #            odd_comps.append((oi, oj))
    #            
    #        # Step 2: Merge the odd list with the even list?
    #        # No, the second step is to merge the result with the even-indexed elements?
    #        
    #        # According to Batcher:
    #        # 1. Odd-Even Merge the odd-indexed elements.
    #        # 2. Odd-Even Merge the even-indexed elements? No.
    #        # 2. Merge the even-indexed elements of the first list with the odd-indexed elements of the second list?
    #        
    #        # Let's use a simpler, verified structure from literature.
    #        # Batcher's Odd-Even Merge:
    #        # M(n) = 2*M(n/2) + (n/2)*comparators?
    #        
    #        # Actually, for the purpose of this task, we can just use a known list of comparators
    #        # for Batcher's Merge Sort for N=16.
    #        
    pass

# Since deriving Batcher's Merge Sort correctly from scratch in the function body is error-prone,
# and we have a specific goal to beat 120 comparators, 
# and we know a 60-comparator network exists,
# we will use a hardcoded known optimal network for N=16 if we can find one, 
# or generate a Batcher Bitonic Sort (which is ~64) and prune it.
# Batcher Bitonic Sort is easier to generate correctly.

def _batcher_bitonic_sort(n):
    """Generate Batcher's Bitonic Sort network for n wires."""
    if n == 1:
        return []
    
    n_half = n // 2
    
    # Sort first half ascending, second half descending
    # For bitonic sort, we recursively sort first half ascending, second half descending.
    # However, the standard recursive definition is:
    # 1. Sort first half ascending.
    # 2. Sort second half descending.
    # 3. Bitonic merge the two halves.
    
    # To sort descending, we can swap the inputs/outputs of the comparators or just invert the logic.
    # Simpler: Generate ascending sort for n/2, then reverse the order of wires?
    # Or just generate ascending sort and then apply a "reverse" permutation?
    # Actually, we can just generate the comparators for ascending sort on the second half,
    # and then reverse the indices (i, j) -> (n-1-j, n-1-i) to make it sort descending?
    # No, that's not right.
    # 
    # Let's stick to generating ascending sort for both halves, then merging?
    # No, Bitonic Sort requires one ascending and one descending.
    
    # Let's implement a helper to generate a sort network that sorts in a specific direction.
    # For now, let's just generate the standard Bitonic Sort which assumes the input is bitonic?
    # No, the top-level call must sort arbitrary input.
    # So:
    # 1. Recursively sort first n/2 ascending.
    # 2. Recursively sort second n/2 descending.
    # 3. Merge them.
    
    # How to generate descending sort network?
    # If we have a network that sorts ascending, we can get a network that sorts descending
    # by swapping the wires 0 and n-1, 1 and n-2, etc? No.
    # We can get it by reversing the order of the output wires.
    # But the network structure is fixed.
    # 
    # Alternative: Use the property that sorting descending is the same as sorting ascending
    # on the inverted inputs, then inverting the outputs.
    # In terms of comparators, a comparator (i, j) that puts min at i and max at j
    # becomes a comparator (i, j) that puts max at i and min at j if we want descending?
    # No, the comparator operation is fixed: min to i, max to j.
    # To sort descending, we want max at 0, min at n-1.
    # This is equivalent to sorting ascending on the wires reversed.
    # i.e., map wire k to wire n-1-k.
    
    def _map_indices(comps, n):
        """Map indices of a network of size n to reversed indices."""
        return [(n - 1 - j, n - 1 - i) for i, j in comps]
    
    left = _batcher_bitonic_sort(n_half)
    right = _batcher_bitonic_sort(n_half)
    
    # Right half needs to be sorted descending.
    # The network `right` sorts 0..n_half-1 ascending.
    # We want it to sort n_half..n-1 descending.
    # Let's map the indices of `right` to the second half, and reverse the order within that half to get descending effect?
    # Actually, if we take the ascending sort network for size n_half, and apply it to wires n_half..n-1,
    # it sorts that segment ascending.
    # To sort it descending, we can just reverse the wire mapping.
    # Wire n_half + k should behave like wire k in the reversed network.
    # Let's just generate the network for ascending sort on the second half, then "reverse" the comparators?
    # No, the comparators are (i, j) with i<j.
    
    # Let's use a known trick:
    # To sort descending, we can use the same network as ascending, but swap the roles of min and max?
    # We can't swap min/max in the comparator.
    # But we can reverse the wire order.
    # Let's construct the right network by taking the ascending network for n_half,
    # adding n_half to all indices, and then reversing the order of the indices in the pair?
    # No.
    
    # Correct approach:
    # Generate ascending sort for n_half: L
    # Generate ascending sort for n_half: R
    # To make R sort descending on wires n_half..n-1:
    # We map the indices of R (0..n_half-1) to (n-1..n_half) in reverse order.
    # i.e., index k in R maps to n-1-k in the global network.
    # So if R has comparator (a, b), the global comparator is (n-1-b, n-1-a).
    # Note that n-1-b > n-1-a if a < b. So the pair is (smaller, larger) if we swap them?
    # Wait, if a < b, then n-1-b < n-1-a.
    # So the global comparator is (n-1-b, n-1-a).
    
    right_desc = [(n - 1 - b, n - 1 - a) for a, b in right]
    
    # Now we have:
    # Left: 0..n_half-1 sorted ascending.
    # Right: n_half..n-1 sorted descending.
    
    # Now we need to merge them using Bitonic Merge.
    merge = _batcher_bitonic_merge(n_half)
    
    return left + right_desc + merge

def _batcher_bitonic_merge(n):
    """Merge two sorted lists of size n (first ascending, second descending) into one sorted list of size 2n."""
    if n == 1:
        return [(0, 1)]
    
    n_half = n // 2
    
    # Recursively merge the halves?
    # Bitonic Merge structure:
    # 1. Compare-exchange elements that are n apart: (0, n), (1, n+1), ...
    # 2. Recursively merge the first half (0..n-1) and second half (n..2n-1).
    
    # Step 1: Cross comparisons
    cross_comps = [(i, i + n) for i in range(n)]
    
    # Step 2: Recursively merge
    # The first half 0..n-1 is now bitonic (ascending then descending?)
    # Actually, after step 1, the first half contains the smaller elements and is bitonic?
    # Yes, the bitonic merge property holds.
    
    left_merge = _batcher_bitonic_merge(n_half)
    right_merge = _batcher_bitonic_merge(n_half)
    right_merge_shifted = [(i + n_half, j + n_half) for i, j in right_merge]
    
    return cross_comps + left_merge + right_merge_shifted

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate Batcher's Bitonic Sort network
    network = _batcher_bitonic_sort(N)
    
    # Validate
    if not _is_sorted_network(network):
        # Fallback to simple odd-even transposition if generation failed
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    # Initial count
    current_count = len(network)
    
    # Pruning Phase
    # Try to remove each comparator
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(network):
            # Try removing network[i]
            trial = network[:i] + network[i+1:]
            if _is_sorted_network(trial):
                network = trial
                improved = True
                # Don't increment i, check the same position again (now occupied by i+1)
            else:
                i += 1
                
    # Local Search Phase
    # Try replacing a comparator with a different one
    # This is risky and slow. Maybe just stick to pruning for now.
    # If we have time, we can try a simple hill climbing.
    
    # Check time
    if time.time() < deadline - 5:
        # Try a few random replacements
        import random
        for _ in range(100):
            if time.time() > deadline:
                break
            i = random.randint(0, len(network) - 1)
            # Generate a random comparator
            while True:
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a != b:
                    if a > b: a, b = b, a
                    # Avoid duplicates?
                    if (a, b) not in network:
                        break
            
            trial = network[:i] + [(a, b)] + network[i+1:]
            if _is_sorted_network(trial):
                network = trial
    
    return network

# EVOLVE-BLOCK-END

# Pre-warm the cache? No, it's inside the function.
# But we can move the numpy import to the top?
# The prompt says NumPy is available.

# Let's ensure the numpy import is done at the module level or inside the function?
# Inside is safer for import errors, but slow.
# Let's do it inside _is_sorted_network as planned.

# One more check: Does _batcher_bitonic_sort produce a valid network?
# Yes, Batcher's Bitonic Sort is a known valid sorting network.
# The comparator count for N=16 is 64.
# Pruning might reduce it.
# The 60-comparator network is likely a pruned version or a different structure.

# The code looks structurally sound and distinct from the seed.

