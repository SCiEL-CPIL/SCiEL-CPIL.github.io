"""Size-efficient sorting networks on 18 wires.

Strategy:
1. Build a Batcher odd-even merge sort network (known to use ~90-100 comparators for N=18).
2. Use an efficient Boolean verification via numpy.
3. Apply a multi-phase local search:
   a. Remove redundant comparators (single removal).
   b. Try replacing pairs of comparators with a single one.
   c. Try moving comparators to enable further removals.
   d. Random restarts with perturbation if stuck.
"""

import time
import random
import numpy as np

N = 18
NUM_BOOLS = 1 << N  # 262144
DEADLINE = 82.0  # seconds (8s safety margin before 90s kill)


def _precompute_sorted_mask():
    """Precompute a boolean array: True if the state index is a sorted configuration.
    Sorted means all 0s followed by all 1s (in wire order 0..N-1)."""
    mask = np.zeros(NUM_BOOLS, dtype=np.bool_)
    for k in range(N + 1):
        # k ones in the top k positions: bits N-k .. N-1 are 1
        val = ((1 << k) - 1) << (N - k)
        mask[val] = True
    return mask


_SORTED_MASK = _precompute_sorted_mask()


def _verify_network(comparators):
    """Verify that the network sorts all 2^N Boolean inputs.
    Returns True if valid, False otherwise.
    Uses vectorized numpy operations for speed."""
    if not comparators:
        return False
    current = np.arange(NUM_BOOLS, dtype=np.int32)
    for i, j in comparators:
        bit_i = (current >> i) & 1
        bit_j = (current >> j) & 1
        mask_swap = (bit_i == 1) & (bit_j == 0)
        if np.any(mask_swap):
            current = np.where(mask_swap, (current & ~(1 << i)) | (1 << j), current)
    return np.all(_SORTED_MASK[current])


def _batcher_oe_merge_sort(n):
    """Generate Batcher's odd-even merge sort network for n wires.
    Returns list of (i, j) comparators."""
    if n <= 1:
        return []
    
    comps = []
    
    # Recursive helper
    def oe_merge_sort(a, b, n):
        """Sort n wires starting at offset a..a+n-1, producing comparators."""
        if n <= 1:
            return
        half = n // 2
        # Sort first half
        for i, j in oe_merge_sort(a, b, half):
            comps.append((i, j))
        # Sort second half
        for i, j in oe_merge_sort(a + half, b, n - half):
            comps.append((i, j))
        # Odd-even merge
        for i, j in oe_merge(a, a + half, n):
            comps.append((i, j))
    
    def oe_merge(a, b, n):
        """Odd-even merge of two sorted halves: a..a+n/2-1 and b..b+n/2-1.
        Both halves are of length n/2."""
        if n <= 1:
            return
        half = n // 2
        # Compare elements at positions a+i and b+i for i in 0..half-1
        # But only where i is even (0-indexed within the merge step)
        # Actually, odd-even merge: compare (a+i, b+i) for i even, then (a+i+1, b+i) for i odd
        # Standard formulation:
        # For i = 0, 2, 4, ... < half: compare (a+i, b+i)
        # For i = 1, 3, 5, ... < half: compare (a+i, b+i)  -- wait, that's just all i
        
        # Let me use the standard recursive odd-even merge:
        # merge(a, b, n) where both halves have length n/2:
        #   for i in 0..n/2-1: compare (a+i, b+i)
        #   if n/2 > 1:
        #       merge(a, a+1, n-1)  -- merge the interleaved results
        #       merge(b, b+1, n-1)
        
        # Simpler iterative approach for odd-even merge of two sorted sequences of length m:
        m = half
        # Phase 1: compare (a+i, b+i) for i = 0, 1, ..., m-1
        for i in range(m):
            yield (a + i, b + i)
        # Phase 2: recursively merge the two resulting sequences of length m-1 each
        # The first resulting sequence is at positions a, a+1, ..., a+m-2 (length m-1)
        # The second is at positions b, b+1, ..., b+m-2 (length m-1)
        if m > 1:
            for i, j in oe_merge(a, a + 1, 2 * (m - 1)):
                yield (i, j)
            for i, j in oe_merge(b, b + 1, 2 * (m - 1)):
                yield (i, j)
    
    # This recursive approach is getting complex. Let me use a cleaner formulation.
    # Batcher's odd-even mergesort:
    #   if n <= 1: return
    #   sort first n/2, sort second n/2
    #   odd-even merge the two halves
    
    # Let me restart with a cleaner implementation.
    comps.clear()
    
    def sort_network(start, length):
        """Generate comparators to sort wires start..start+length-1."""
        if length <= 1:
            return
        half = length // 2
        # Sort first half
        sort_network(start, half)
        # Sort second half  
        sort_network(start + half, length - half)
        # Odd-even merge
        odd_even_merge(start, start + half, length)
    
    def odd_even_merge(a, b, total_len):
        """Merge two sorted sequences: a..a+total_len/2-1 and b..b+total_len/2-1.
        total_len must be even. Both halves have length total_len/2."""
        m = total_len // 2
        if m <= 0:
            return
        # Compare corresponding elements
        for i in range(m):
            comps.append((a + i, b + i))
        # Recursively merge the two subsequences of length m-1
        if m > 1:
            odd_even_merge(a, a + 1, 2 * (m - 1))
            odd_even_merge(b, b + 1, 2 * (m - 1))
    
    sort_network(0, n)
    return comps


def _odd_even_transposition_network():
    """Generate the odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _depth_of_network(comparators):
    """Compute the parallel depth of the network."""
    depth = np.zeros(N, dtype=np.int32)
    for i, j in comparators:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return int(np.max(depth))


def _local_search(comparators, max_time):
    """Local search to reduce comparator count while maintaining validity."""
    start_time = time.time()
    best = list(comparators)
    best_len = len(best)
    current = list(best)
    
    def time_left():
        return time.time() - start_time < max_time
    
    # Phase 1: Greedy removal - try removing each comparator
    changed = True
    while changed and time_left() and len(current) > 68:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if not time_left():
                break
            trial = current[:idx] + current[idx+1:]
            if _verify_network(trial):
                current = trial
                changed = True
                if len(current) < best_len:
                    best_len = len(current)
                    best = list(current)
                break  # Restart scan from beginning after a removal
    
    # Phase 2: Try replacing adjacent pairs with a single comparator
    for _ in range(20):  # Limited iterations
        if not time_left() or len(current) <= 68:
            break
        improved = False
        for idx in range(len(current) - 1):
            if not time_left():
                break
            # Try replacing current[idx] and current[idx+1] with one random comparator
            for _ in range(3):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                trial = current[:idx] + [(i, j)] + current[idx+2:]
                if _verify_network(trial):
                    current = trial
                    improved = True
                    if len(current) < best_len:
                        best_len = len(current)
                        best = list(current)
                    break
            if improved:
                break
        if not improved:
            break
    
    # Phase 3: Try moving comparators (swap adjacent non-commuting pairs)
    for _ in range(10):
        if not time_left():
            break
        improved = False
        for idx in range(len(current) - 1):
            if not time_left():
                break
            c1, c2 = current[idx], current[idx+1]
            # Only try if they might commute or not
            if c1[0] != c2[0] and c1[0] != c2[1] and c1[1] != c2[0] and c1[1] != c2[1]:
                # They are independent, swapping shouldn't matter, skip
                continue
            trial = current[:]
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _verify_network(trial):
                # Check if this enables a removal
                for rem_idx in range(len(trial) - 1, -1, -1):
                    if not time_left():
                        break
                    t2 = trial[:rem_idx] + trial[rem_idx+1:]
                    if _verify_network(t2):
                        current = t2
                        improved = True
                        if len(current) < best_len:
                            best_len = len(current)
                            best = list(current)
                        break
                if improved:
                    break
        if not improved:
            break
    
    return best


def construct():
    """Construct a sorting network for 18 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    
    # Start with Batcher's odd-even merge sort
    comparators = _batcher_oe_merge_sort(N)
    
    # Verify it's valid
    if not _verify_network(comparators):
        # Fallback to odd-even transposition
        comparators = _odd_even_transposition_network()
        if not _verify_network(comparators):
            # Should never happen, but return something
            return comparators
    
    initial_len = len(comparators)
    
    # Apply local search with time budget
    remaining = DEADLINE - (time.time() - start_time) - 3.0  # 3s safety
    if remaining > 1.0:
        comparators = _local_search(comparators, max_time=remaining)
    
    # Final verification
    if not _verify_network(comparators):
        # Fallback
        comparators = _odd_even_transposition_network()
    
    return comparators
