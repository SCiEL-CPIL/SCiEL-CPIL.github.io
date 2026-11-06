"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Pre-compute all binary inputs for zero-one principle check
# Shape: (65536, 16)
# Using efficient bit unpacking
_INPUTS = None

def _get_inputs():
    global _INPUTS
    if _INPUTS is None:
        # Create array of indices 0 to 2^N - 1
        idx = np.arange(1 << N, dtype=np.uint32)
        # Unpack bits
        # Shift right by 0 to N-1 and mask with 1
        # Reshape to (2^N, N)
        shifts = np.arange(N, dtype=np.uint32)
        # idx[:, None] has shape (2^N, 1), shifts[None, :] has shape (1, N)
        # Broadcasting works
        bits = (idx[:, None] >> shifts[None, :]) & 1
        _INPUTS = bits.astype(np.uint8)
    return _INPUTS

def _is_sorted_network(net):
    """Verify network sorts all 2^16 boolean inputs using zero-one principle with NumPy."""
    if not net:
        return N <= 1
    inputs = _get_inputs().copy() # Must copy to avoid modifying global state
    for i, j in net:
        a = inputs[:, i]
        b = inputs[:, j]
        # For binary inputs, min is AND, max is OR
        # This is faster than np.minimum/maximum potentially, but let's use min/max for clarity/safety
        # Actually, bitwise ops on uint8 are very fast.
        # inputs[:, i] = a & b
        # inputs[:, j] = a | b
        # Let's use this optimization
        inputs[:, i] = a & b
        inputs[:, j] = a | b
        
    # Check if sorted: no '1' followed by '0'
    # diff > 0 is fine (0->1), diff == 0 is fine, diff < 0 is bad (1->0)
    # We can check if the array is non-decreasing.
    # A faster check for binary sorted: 
    # If we count the number of 1s, they must all be at the end.
    # Or simply: np.any(np.diff(inputs, axis=1) < 0)
    # Let's use the diff method, it's robust.
    # Optimization: If we find a 1, all subsequent must be 1.
    # But diff is O(N*2^N) which is 16 * 65536 = 1M ops. Very fast in C.
    if np.any(np.diff(inputs, axis=1) < 0):
        return False
    return True

def _greedy_prune(net, time_limit=2.0):
    """Greedily remove redundant comparators from the network."""
    net = list(net)
    start_time = time.time()
    deadline = start_time + time_limit
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(net)):
            if time.time() > deadline:
                break
            # Try removing net[idx]
            test_net = [net[i] for i in range(len(net)) if i != idx]
            if _is_sorted_network(test_net):
                net = test_net
                changed = True
                break # Restart scan from beginning as indices shifted
    return net

def _random_network():
    """Generate a random valid network (not necessarily sorting)."""
    # Random length between 40 and 120
    length = random.randint(40, 120)
    net = []
    for _ in range(length):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        net.append((i, j))
    return net

def _odd_even_transposition():
    """Standard Odd-Even Transposition network."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _batcher_odd_merge():
    """Batcher's Odd-Merge sort network."""
    # Recursively build
    def odd_merge(n):
        if n == 1:
            return []
        half = n // 2
        left = odd_merge(half)
        right = odd_merge(n - half)
        # Shift right network by half
        right_shifted = [(i + half, j + half) for i, j in right]
        
        # Odd-even merge comparators
        # Phase 1: Compare (0, 1), (2, 3), ... up to n-1
        # Actually, the standard odd-even merge phase is:
        # For k in 0 to n-1 step 2: compare (k, k+1) -- wait, indices need to be valid.
        # The comparators are (i, i+1) for i in 0, 2, 4... and (i, i+1) for i in 1, 3, 5...
        # Specifically:
        # Compare (0, 1), (2, 3), ..., (n-2, n-1)
        # Compare (1, 2), (3, 4), ..., (n-3, n-2)
        # Note: The second set stops at n-2 because we need pairs (i, i+1) where i+1 < n.
        # If n is even, last index is n-1. Pairs (n-3, n-2) is valid.
        # If n is odd, last index is n-1. Pairs (n-3, n-2) is valid.
        
        merge_comps = []
        # First pass: even indices
        for i in range(0, n - 1, 2):
            merge_comps.append((i, i + 1))
        # Second pass: odd indices
        for i in range(1, n - 1, 2):
            merge_comps.append((i, i + 1))
            
        return left + right_shifted + merge_comps

    return odd_merge(N)

def _batcher_bitonic():
    """Batcher's Bitonic sort network."""
    def bitonic_sort(n):
        if n == 1:
            return []
        half = n // 2
        left = bitonic_sort(half)
        right = bitonic_sort(n - half)
        right_shifted = [(i + half, j + half) for i, j in right]
        
        # Bitonic merge
        # Merge two sorted sequences (one ascending, one descending)
        # Standard bitonic merge comparators:
        # For k in range(log2(n)):
        #   for i in range(n - 2**k):
        #     compare (i, i + 2**k)
        #   for i in range(n - 2**k, n):
        #     compare (i, i - 2**k) -- wait, direction matters.
        
        # Simpler recursive definition for comparators:
        # Merge phase for size n:
        # Compare (i, i + n/2) for i in 0..n/2-1
        # Then recursively merge first half and second half.
        
        merge_comps = []
        step = n // 2
        for i in range(step):
            merge_comps.append((i, i + step))
        
        # Recursively merge the two halves
        # The first half (0..step-1) is now a bitonic sequence?
        # Actually, after the cross comparisons, the top half contains the larger elements?
        # No, bitonic sort logic:
        # 1. Sort first half ascending, second half descending.
        # 2. Merge them.
        # My recursive call above sorts both ascending. 
        # Standard Bitonic Sort:
        # 1. Sort first half ascending.
        # 2. Sort second half descending.
        # 3. Merge.
        
        # Let's stick to a known good implementation or just use the transposition/odd-merge as seeds.
        # Odd-merge is better (119 vs 120).
        pass
    # Since implementing Bitonic correctly from scratch is error-prone, 
    # and Odd-Merge is already good, we will rely on Odd-Merge and Randoms.
    return _odd_even_transposition() # Fallback

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    # Initial Population
    pop = []
    # Add known good structures
    pop.append(_odd_even_transposition())
    pop.append(_batcher_odd_merge())
    
    # Add randoms
    for _ in range(10):
        pop.append(_random_network())
    
    # Prune all initial individuals
    pruned_pop = []
    for net in pop:
        if time.time() > deadline:
            break
        p_net = _greedy_prune(net, time_limit=1.0)
        if _is_sorted_network(p_net):
            pruned_pop.append(p_net)
        else:
            # If pruning broke it (shouldn't happen), keep original if valid
            if _is_sorted_network(net):
                pruned_pop.append(net)
            # else discard
    
    if not pruned_pop:
        # Fallback to basic valid network
        base = _odd_even_transposition()
        return base

    # Sort by length
    pruned_pop.sort(key=len)
    best_net = pruned_pop[0]
    best_len = len(best_net)
    
    # Genetic Algorithm
    pop_size = 10
    # Fill population with variations of best
    while len(pruned_pop) < pop_size:
        # Mutate best
        net = list(best_net)
        # Random mutation: swap a comparator
        if net:
            idx = random.randint(0, len(net) - 1)
            i, j = net[idx]
            # Change to random valid pair
            new_i = random.randint(0, N - 2)
            new_j = random.randint(new_i + 1, N - 1)
            net[idx] = (new_i, new_j)
        # Prune
        p_net = _greedy_prune(net, time_limit=0.5)
        if _is_sorted_network(p_net):
            pruned_pop.append(p_net)
        else:
            pruned_pop.append(best_net) # Keep best if mutation failed

    generations = 0
    while time.time() < deadline:
        generations += 1
        # Selection: Keep top half
        pruned_pop.sort(key=len)
        survivors = pruned_pop[:pop_size // 2]
        
        # Crossover and Mutation
        new_pop = list(survivors) # Elitism
        
        while len(new_pop) < pop_size and time.time() < deadline:
            # Pick two parents
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            
            # Crossover: Take random prefix of p1 and suffix of p2
            # Or just concatenate and prune
            # Simple concat
            child = list(p1) + list(p2)
            
            # Prune child aggressively
            child = _greedy_prune(child, time_limit=1.0)
            
            # Mutation: Try to remove random comparator
            if random.random() < 0.5 and child:
                idx = random.randint(0, len(child) - 1)
                test_child = [c for i, c in enumerate(child) if i != idx]
                if _is_sorted_network(test_child):
                    child = test_child
            
            # Mutation: Add random comparator (low prob, to explore)
            if random.random() < 0.1:
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                child.append((i, j))
                child = _greedy_prune(child, time_limit=0.5)
            
            if _is_sorted_network(child):
                new_pop.append(child)
            else:
                # If invalid, add a random valid one from survivors
                new_pop.append(random.choice(survivors))
        
        # Update best
        new_pop.sort(key=len)
        if len(new_pop[0]) < best_len:
            best_len = len(new_pop[0])
            best_net = new_pop[0]
            
        pruned_pop = new_pop
        
        # Check time
        if time.time() > deadline - 5.0:
            break

    return best_net
# EVOLVE-BLOCK-END
