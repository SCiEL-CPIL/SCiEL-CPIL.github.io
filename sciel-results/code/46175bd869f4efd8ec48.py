N = 18

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_DEADLINE = 80.0  # seconds; leave margin under the 90s kill
_start_time = None

def _check_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_valid_network(comparators, n=N):
    """Check if a comparator list sorts all 2^n Boolean inputs."""
    if len(comparators) == 0:
        return n <= 1
    # Quick structural checks
    for i, j in comparators:
        if not (0 <= i < j < n):
            return False
    # Verify on all 2^n Boolean inputs
    for mask in range(1 << n):
        wires = [(mask >> k) & 1 for k in range(n)]
        for i, j in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        # Check sorted
        for k in range(n - 1):
            if wires[k] > wires[k + 1]:
                return False
    return True

def _bitonic_sort_network(n, offset=0, direction=1):
    """Generate bitonic sort comparators for n wires (n must be power of 2)."""
    comps = []
    if n <= 1:
        return comps
    
    def bitonic_merge(k, off, d):
        if k <= 1:
            return
        h = k // 2
        for i in range(h):
            a = off + i
            b = off + i + h
            if d == 1:
                comps.append((a, b))
            else:
                comps.append((b, a))
        bitonic_merge(h, off, d)
        bitonic_merge(h, off + h, d)
    
    def bitonic_sort(k, off, d):
        if k <= 1:
            return
        h = k // 2
        bitonic_sort(h, off, 1)
        bitonic_sort(h, off + h, -1)
        bitonic_merge(k, off, d)
    
    bitonic_sort(n, offset, direction)
    return comps

def _odd_even_transposition(n=N):
    """Odd-even transposition sort network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _try_remove_comparator(comparators, index):
    """Try removing comparator at index and check validity."""
    new_comps = comparators[:index] + comparators[index+1:]
    return _is_valid_network(new_comps)

def _greedy_prune(comparators, time_limit=75.0):
    """Greedily remove comparators while maintaining validity."""
    comps = list(comparators)
    deadline = _start_time + time_limit if _start_time else time.time() + time_limit
    
    # Try removing from the end first (often less critical)
    i = len(comps) - 1
    while i >= 0:
        if time.time() > deadline:
            break
        if _try_remove_comparator(comps, i):
            comps = comps[:i] + comps[i+1:]
        i -= 1
    
    # Multiple passes
    pass_count = 0
    while time.time() < deadline and pass_count < 5:
        changed = False
        for i in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                break
            if _try_remove_comparator(comps, i):
                comps = comps[:i] + comps[i+1:]
                changed = True
        pass_count += 1
        if not changed:
            break
    
    return comps

def _swap_comparators(comparators, i, j):
    """Try swapping two comparators."""
    new_comps = list(comparators)
    new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
    return _is_valid_network(new_comps)

def _local_search(comparators, time_limit=70.0):
    """Local search: try removing and swapping comparators."""
    comps = list(comparators)
    deadline = _start_time + time_limit if _start_time else time.time() + time_limit
    
    best = comps
    best_len = len(comps)
    
    # First, greedy removal
    i = len(comps) - 1
    while i >= 0 and time.time() < deadline:
        if _try_remove_comparator(comps, i):
            comps = comps[:i] + comps[i+1:]
            best = comps
            best_len = len(comps)
        i -= 1
    
    # Then, try more removals in forward direction
    for pass_num in range(3):
        if time.time() >= deadline:
            break
        i = 0
        while i < len(comps) and time.time() < deadline:
            if _try_remove_comparator(comps, i):
                comps = comps[:i] + comps[i+1:]
                best = comps
                best_len = len(comps)
            else:
                i += 1
    
    return best

def _construct_bitonic_18():
    """Construct a sorting network for 18 wires using bitonic sort on 16 + merge 2."""
    # Bitonic sort on wires 0-15
    comps = _bitonic_sort_network(16, 0, 1)
    
    # Sort wires 16-17
    comps.append((16, 17))
    
    # Merge the 2-element sorted sequence into the 16-element sorted sequence
    # Use a simple but correct merge: compare each of the 2 elements with
    # appropriate elements from the 16-element sequence
    
    # For a correct merge of sorted A[0..15] and sorted B[16,17]:
    # We need to interleave them properly.
    # Simple approach: use a cascade of comparisons
    
    # Insert B[16] into A: compare with A[0..15] in a specific pattern
    # Then insert B[17] similarly
    
    # A correct (but not optimal) merge:
    # Compare B[16] with A[7], then depending on result, compare with appropriate half
    # But this is data-dependent, so for a fixed network we need a different approach.
    
    # Use the "merge" network: for each element in B, do a binary insertion
    # into A. For a fixed network, we can use a "shuffle" approach.
    
    # Simplest correct merge: use odd-even transposition on the boundary
    # Actually, let's use a known correct construction:
    
    # After sorting A[0..15] and B[16,17], the merged result can be obtained
    # by a network that ensures all A[i] < B[j] comparisons are resolved.
    
    # A simple correct (if not optimal) approach:
    # For i in range(8):
    #   Compare A[i] with B[16]
    # For i in range(8):
    #   Compare A[8+i] with B[16]
    # Similarly for B[17]
    
    # But this may not be correct. Let's use a safer approach:
    # Run a few rounds of odd-even transposition on the last 9 wires (9-17)
    # to merge them.
    
    # Actually, the safest approach: after bitonic sort on 16 and sort on 2,
    # run a small number of global comparisons to ensure correctness.
    
    # Let's use: compare B[16] and B[17] with all of A in a structured way.
    # The "merge" network for m=16, n=2 has a known structure.
    
    # For now, use a simple correct but possibly suboptimal merge:
    # Compare each B element with each A element in a specific order.
    # This gives 16*2 = 32 extra comparators, total 80+1+32 = 113.
    
    # A better approach: use the fact that we can do a "bitonic merge"
    # of the two sorted sequences.
    
    # Let's use a different strategy: instead of bitonic 16 + merge 2,
    # use a different decomposition.
    
    # Alternative: use the "shell sort" network or "knuth" network.
    
    # For now, return the simple construction and let pruning optimize it.
    # Add the merge comparators:
    for a_idx in range(16):
        comps.append((a_idx, 16))
    for a_idx in range(16):
        comps.append((a_idx, 17))
    
    return comps

def _construct_hybrid():
    """Construct a hybrid network: bitonic on subsets + targeted comparisons."""
    comps = []
    
    # Phase 1: Bitonic sort on wires 0-7 (8 wires, 28 comparators)
    comps.extend(_bitonic_sort_network(8, 0, 1))
    
    # Phase 2: Bitonic sort on wires 8-15 (8 wires, 28 comparators)
    comps.extend(_bitonic_sort_network(8, 8, 1))
    
    # Phase 3: Sort wires 16-17
    comps.append((16, 17))
    
    # Phase 4: Merge the three sorted sequences
    # Merge 8+8: bitonic merge of two 8-element sorted sequences
    # This requires 28 comparators for the merge
    def bitonic_merge_8(off, d=1):
        m = []
        for i in range(4):
            a = off + i
            b = off + 4 + i
            if d == 1:
                m.append((a, b))
            else:
                m.append((b, a))
        # Recursively merge halves
        for i in range(2):
            a = off + i
            b = off + 2 + i
            if d == 1:
                m.append((a, b))
            else:
                m.append((b, a))
        for i in range(1):
            a = off + i
            b = off + 1 + i
            if d == 1:
                m.append((a, b))
            else:
                m.append((b, a))
        return m
    
    # Merge wires 0-7 and 8-15
    comps.extend(bitonic_merge_8(0, 1))
    
    # Now wires 0-15 are sorted. Merge with 16-17.
    # Simple merge: compare 16 and 17 with appropriate elements
    # Use a cascade approach
    for i in range(8):
        comps.append((i, 16))
    for i in range(8, 16):
        comps.append((i, 16))
    for i in range(8):
        comps.append((i, 17))
    for i in range(8, 16):
        comps.append((i, 17))
    
    return comps

def construct():
    """
    Construct a sorting network for 18 wires.
    Returns an ordered list of integer pairs (i, j) where 0 <= i < j < 18.
    """
    global _start_time
    _start_time = time.time()
    
    # Generate initial candidate networks
    candidates = []
    
    # Candidate 1: Odd-even transposition (153 comparators)
    candidates.append(_odd_even_transposition(N))
    
    # Candidate 2: Bitonic 16 + merge 2
    if _check_time() < _DEADLINE - 10:
        try:
            c2 = _construct_bitonic_18()
            if _is_valid_network(c2):
                candidates.append(c2)
        except Exception:
            pass
    
    # Candidate 3: Hybrid
    if _check_time() < _DEADLINE - 10:
        try:
            c3 = _construct_hybrid()
            if _is_valid_network(c3):
                candidates.append(c3)
        except Exception:
            pass
    
    # Pick the shortest valid candidate
    if not candidates:
        # Fallback: return odd-even transposition
        return _odd_even_transposition(N)
    
    candidates.sort(key=len)
    best = candidates[0]
    
    # Local search to prune further
    if _check_time() < _DEADLINE - 5:
        best = _local_search(best, time_limit=_DEADLINE - _check_time() - 5)
    
    # Final validity check
    if not _is_valid_network(best):
        return _odd_even_transposition(N)
    
    return best
# EVOLVE-BLOCK-END
