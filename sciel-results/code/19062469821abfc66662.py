import time
import numpy as np
import random

N = 18
ALL_COUNT = 1 << N
DEADLINE = 75.0  # Safety margin (90s kill, leave buffer)

def _init_test_vectors():
    """Create the array of all 2^N boolean inputs, shape (2^N, N)."""
    idx = np.arange(ALL_COUNT, dtype=np.uint32)
    bits = (idx[:, None] >> np.arange(N, dtype=np.uint32)) & 1
    return bits.astype(np.uint8)

def _check_sorted_fast(vals):
    """Fast check if all rows are sorted (non-decreasing)."""
    return np.all(vals[:, :-1] <= vals[:, 1:])

def _simulate_net(net, test_vectors):
    """Simulate a network on test vectors, return the output array."""
    vals = test_vectors.copy()
    for a, b in net:
        va = vals[:, a]
        vb = vals[:, b]
        lo = np.minimum(va, vb)
        hi = np.maximum(va, vb)
        vals[:, a] = lo
        vals[:, b] = hi
    return vals

def _is_sorted(net, test_vectors):
    """
    Check if the network sorts all boolean inputs.
    net: list of (i, j)
    test_vectors: np.array of shape (ALL_COUNT, N)
    """
    if not net:
        return N <= 1
    vals = _simulate_net(net, test_vectors)
    return _check_sorted_fast(vals)

def _try_remove_batch(net, test_vectors, candidates):
    """
    Try to remove multiple comparators at once.
    candidates: list of indices to try removing.
    Returns the maximum number of candidates that can be removed simultaneously.
    Uses a greedy approach: try to remove all, then back off.
    """
    n_cand = len(candidates)
    if n_cand == 0:
        return 0
    
    # First, try removing all candidates at once
    if n_cand <= len(net):
        remaining = [i for i in range(len(net)) if i not in set(candidates)]
        if _is_sorted([net[i] for i in remaining], test_vectors):
            return n_cand
    
    # Try subsets - use greedy: try each individually, keep those that work
    removable = []
    for idx in candidates:
        if idx >= len(net):
            continue
        candidate_net = net[:idx] + net[idx+1:]
        if _is_sorted(candidate_net, test_vectors):
            removable.append(idx)
    
    if not removable:
        return 0
    
    # Try removing all removable ones at once
    rem_set = set(removable)
    remaining = [i for i in range(len(net)) if i not in rem_set]
    if _is_sorted([net[i] for i in remaining], test_vectors):
        return len(removable)
    
    # Fallback: remove them one by one
    count = 0
    for idx in removable:
        if idx >= len(net):
            continue
        candidate_net = net[:idx] + net[idx+1:]
        if _is_sorted(candidate_net, test_vectors):
            net = candidate_net
            count += 1
            # Adjust indices
            removable = [i-1 if i > idx else i for i in removable if i != idx]
    
    return count

def generate_bitonic(n):
    """
    Generate Batcher's Bitonic Sort network for n wires.
    Returns a list of (i, j) pairs.
    """
    def bitonic_merge(n, direction):
        # direction: 0 for ascending, 1 for descending
        if n <= 1:
            return []
        
        half = n // 2
        # Comparators between i and i + half for i in range(half)
        # Note: In bitonic merge, we compare elements at distance 'half'
        # The standard bitonic merge step:
        # Compare (i, i + half) for i in 0..half-1
        # Then recursively sort the two halves in opposite directions.
        
        comparators = []
        for i in range(half):
            comparators.append((i, i + half))
            
        # Recursively sort the two halves
        # If direction is ascending (0):
        #   First half sorted ascending, second half sorted descending?
        #   Actually, the bitonic merge produces a bitonic sequence.
        #   To sort a bitonic sequence of length n:
        #   1. Split into two halves.
        #   2. Compare corresponding elements (i, i+n/2).
        #   3. Recursively sort first half ascending, second half descending?
        #      No, the standard algorithm is:
        #      merge(n, dir):
        #        if n == 1: return
        #        merge(n/2, dir)
        #        merge(n/2, 1-dir)
        #        for i in 0..n/2-1: compare(i, i+n/2)
        #        
        #   Wait, the order of operations in Batcher's merge is:
        #   1. Recursively merge the two halves? No.
        #   The standard recursive definition for Bitonic Sort:
        #   sort(n, dir):
        #     if n == 1: return
        #     sort(n/2, dir)
        #     sort(n/2, 1-dir)
        #     merge(n, dir)
        #     
        #   merge(n, dir):
        #     if n == 1: return
        #     merge(n/2, dir)
        #     merge(n/2, 1-dir)
        #     for i in 0..n/2-1: compare(i, i+n/2)
        #
        #   This is getting confusing. Let's use the standard iterative or correct recursive form.
        #   
        #   Correct Bitonic Merge for length n (assuming input is bitonic):
        #   1. Split into two halves of length n/2.
        #   2. Compare (i, i + n/2) for i in 0..n/2-1.
        #   3. Recursively sort the first half in direction 'dir' and the second half in direction '1-dir'?
        #      No, after the comparators, the first half is the min of pairs, second is max.
        #      If the original was bitonic (up then down), the mins form a sorted sequence in dir, and maxes in 1-dir.
        #      So we recursively sort the first half in 'dir' and the second half in '1-dir'?
        #      Actually, the standard algorithm is:
        #      merge(n, dir):
        #         if n == 1: return
        #         merge(n/2, dir)
        #         merge(n/2, 1-dir)
        #         for i in 0..n/2-1: compare(i, i+n/2)
        #
        #   Let's trace n=4, dir=0 (ascending).
        #   merge(4, 0):
        #     merge(2, 0) -> compare(0,1)
        #     merge(2, 1) -> compare(2,3)
        #     compare(0,2), compare(1,3)
        #   
        #   This seems correct.
        
        # So, for bitonic_merge(n, direction):
        # 1. Recursively merge(n//2, direction)
        # 2. Recursively merge(n//2, 1-direction)
        # 3. Add comparators (i, i + n//2) for i in range(n//2)
        
        left = bitonic_merge(half, direction)
        right = bitonic_merge(half, 1 - direction)
        
        # Add the cross comparators
        for i in range(half):
            comparators.append((i, i + half))
            
        return left + right + comparators

    def bitonic_sort(n, direction=0):
        if n <= 1:
            return []
        half = n // 2
        # Sort first half in direction
        # Sort second half in opposite direction
        # Then merge
        left = bitonic_sort(half, direction)
        right = bitonic_sort(n - half, 1 - direction)
        
        # Now we need to merge the two sorted halves into a bitonic sequence?
        # No, bitonic_sort creates a sorted sequence.
        # The standard bitonic sort algorithm:
        # sort(n, dir):
        #   if n == 1: return
        #   sort(n/2, dir)
        #   sort(n/2, 1-dir)
        #   merge(n, dir)
        
        # My previous merge function was for merging two bitonic sequences?
        # No, the recursive structure is:
        # To sort n elements:
        # 1. Sort first n/2 in dir.
        # 2. Sort second n/2 in 1-dir.
        # 3. Merge the two sorted sequences into a single sorted sequence of length n.
        
        # The merge step for two sorted sequences (one asc, one desc) to form a bitonic sequence?
        # Actually, the output of step 1 and 2 is a bitonic sequence (asc then desc).
        # Then we apply the bitonic merge to sort it.
        
        # So:
        # left = sort(n//2, dir)
        # right = sort(n - n//2, 1-dir)
        # merged = merge(n, dir) # But merge assumes the input is bitonic?
        # The concatenation of left (asc) and right (desc) is bitonic.
        # So we call merge(n, dir) on the combined wires.
        
        # But my merge function above is recursive and generates comparators for the whole n.
        # It expects the input to be bitonic?
        # The recursive merge(n, dir) I wrote above:
        # merge(n, dir):
        #   merge(n/2, dir)
        #   merge(n/2, 1-dir)
        #   compare(i, i+n/2)
        #
        # This is the standard bitonic merge. It takes a bitonic sequence and sorts it.
        
        # So the total network is:
        # sort(n, dir):
        #   if n == 1: return []
        #   half = n // 2
        #   net_left = sort(half, dir)
        #   net_right = sort(n - half, 1 - dir)
        #   net_merge = merge(n, dir)
        #   return net_left + net_right + net_merge
        
        # Wait, the merge step operates on the wires 0..n-1.
        # The left sort operates on 0..half-1.
        # The right sort operates on half..n-1.
        # The merge operates on 0..n-1.
        # This is correct.
        
        return left + right + bitonic_merge(n, direction)

    return bitonic_sort(n)

def construct():
    start_time = time.time()
    test_vectors = _init_test_vectors()
    
    # Generate initial network: Batcher's Bitonic Sort
    net = generate_bitonic(N)
    
    # Verify
    if not _is_sorted(net, test_vectors):
        # Fallback to OET if Bitonic generation is flawed
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        
    best_net = net
    best_len = len(net)
    
    def _greedy_prune(net_in, test_vectors, deadline):
        """Aggressively prune a network using multiple strategies."""
        current = net_in.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each comparator, in random order
            indices = list(range(len(current)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                if idx >= len(current):
                    continue
                candidate = current[:idx] + current[idx+1:]
                if _is_sorted(candidate, test_vectors):
                    current = candidate
                    improved = True
                    # After removal, restart shuffle for better mixing
                    break
        return current
    
    def _multi_start_optimize(start_net, test_vectors, deadline, num_starts=5):
        """Try multiple random perturbations and prune each."""
        best = start_net.copy()
        best_len = len(start_net)
        
        for s in range(num_starts):
            if time.time() > deadline:
                break
            perturbed = start_net.copy()
            # Insert 1-3 random comparators
            n_ins = random.randint(1, 3)
            for _ in range(n_ins):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                perturbed.append((i, j))
            
            # Prune the perturbed network
            pruned = _greedy_prune(perturbed, test_vectors, deadline)
            if len(pruned) < best_len:
                best = pruned
                best_len = len(pruned)
        
        return best
    
    # Phase 1: Initial aggressive pruning
    deadline1 = start_time + 15
    pruned = _greedy_prune(net, test_vectors, deadline1)
    if len(pruned) < best_len:
        best_net = pruned
        best_len = len(pruned)
    
    # Phase 2: Multiple restarts with perturbations
    deadline2 = start_time + 45
    while time.time() < deadline2:
        improved = _multi_start_optimize(best_net, test_vectors, deadline2, num_starts=3)
        if len(improved) < best_len:
            best_net = improved
            best_len = len(improved)
        else:
            # No improvement from last round, try a different strategy
            # Try swapping pairs of comparators
            if time.time() > deadline2 - 5:
                break
    
    # Phase 3: Final polish - very careful single-comparator removal
    deadline3 = start_time + 70
    current = best_net.copy()
    changed = True
    while changed and time.time() < deadline3:
        changed = False
        # Try each position in order
        for idx in range(len(current)):
            if time.time() > deadline3:
                break
            candidate = current[:idx] + current[idx+1:]
            if _is_sorted(candidate, test_vectors):
                current = candidate
                changed = True
                break
    
    if len(current) < best_len:
        best_net = current
        best_len = len(current)
    
    return best_net
