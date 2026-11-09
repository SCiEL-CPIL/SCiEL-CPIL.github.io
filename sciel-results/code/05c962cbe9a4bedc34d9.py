import time
import numpy as np
import random

N = 18
ALL_COUNT = 1 << N
DEADLINE = 50.0  # Safety margin

def _init_test_vectors():
    """Create the array of all 2^N boolean inputs, shape (2^N, N)."""
    idx = np.arange(ALL_COUNT, dtype=np.uint32)
    bits = (idx[:, None] >> np.arange(N, dtype=np.uint32)) & 1
    return bits.astype(np.uint8)

def _is_sorted(net, test_vectors):
    """
    Check if the network sorts all boolean inputs.
    net: list of (i, j)
    test_vectors: np.array of shape (ALL_COUNT, N)
    """
    if not net:
        return N <= 1
        
    vals = test_vectors.copy()
    for a, b in net:
        va = vals[:, a]
        vb = vals[:, b]
        lo = np.minimum(va, vb)
        hi = np.maximum(va, vb)
        vals[:, a] = lo
        vals[:, b] = hi
        
    return np.all(vals[:, :-1] <= vals[:, 1:])

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
    
    # Optimization Loop
    while time.time() - start_time < DEADLINE:
        current_net = best_net.copy()
        
        # 1. Greedy Removal (Backward)
        changed = True
        while changed:
            changed = False
            for idx in range(len(current_net) - 1, -1, -1):
                if time.time() - start_time > DEADLINE:
                    break
                candidate = current_net[:idx] + current_net[idx+1:]
                if _is_sorted(candidate, test_vectors):
                    current_net = candidate
                    changed = True
            if time.time() - start_time > DEADLINE:
                break
        
        # 2. Greedy Removal (Forward)
        changed = True
        while changed:
            changed = False
            for idx in range(len(current_net)):
                if time.time() - start_time > DEADLINE:
                    break
                candidate = current_net[:idx] + current_net[idx+1:]
                if _is_sorted(candidate, test_vectors):
                    current_net = candidate
                    changed = True
                    # Note: after removal, indices shift, so we restart the loop
                    # by breaking out and restarting 'while changed'
                    break 
            if time.time() - start_time > DEADLINE:
                break

        # Check if we improved
        if len(current_net) < best_len:
            best_net = current_net
            best_len = len(current_net)
            
        # If we hit a local minimum, try perturbation
        # Insert a random comparator
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Randomly insert 1 comparator
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        perturbed_net = current_net.copy()
        perturbed_net.append((i, j))
        
        # Optimize the perturbed network
        # Just run a few passes of greedy removal
        for _ in range(2): # Limit passes to save time
            changed = True
            while changed and time.time() - start_time < DEADLINE:
                changed = False
                # Random order removal to escape symmetry
                indices = list(range(len(perturbed_net)))
                random.shuffle(indices)
                for idx in indices:
                    if time.time() - start_time > DEADLINE:
                        break
                    if idx >= len(perturbed_net): continue
                    candidate = perturbed_net[:idx] + perturbed_net[idx+1:]
                    if _is_sorted(candidate, test_vectors):
                        perturbed_net = candidate
                        changed = True
                        break # Restart shuffle loop
                        
        if len(perturbed_net) < best_len:
            best_net = perturbed_net
            best_len = len(perturbed_net)
            
    return best_net
