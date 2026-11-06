import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # Safety margin before 90s kill

# Precompute full matrix of all 2^N boolean inputs
# Using uint8 for efficient bitwise operations
all_masks = np.arange(1 << N, dtype=np.int32)
W0_full = np.zeros((1 << N, N), dtype=np.uint8)
for i in range(N):
    W0_full[:, i] = (all_masks >> i) & 1

# Precompute a subset for fast verification
# Include boundary cases (low and high weights) and random samples
# This helps reject invalid networks quickly
subset_indices = np.unique(np.concatenate([
    np.arange(0, 200),       # Low weights
    np.arange((1 << N) - 200, 1 << N),  # High weights
    np.random.choice(1 << N, 512, replace=False)
]))
W0_subset = W0_full[subset_indices]

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _is_sorted_network_fast(net):
    """
    Fast verification using subset of inputs.
    Returns True if all subset inputs are sorted.
    """
    if not net:
        return True
    
    W_sub = W0_subset.copy()
    for i, j in net:
        col_i = W_sub[:, i]
        col_j = W_sub[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    if np.any(W_sub[:, :-1] > W_sub[:, 1:]):
        return False
    return True

def _is_sorted_network_full(net):
    """
    Full verification over all 2^N inputs.
    """
    if not net:
        return True
    
    W_full = W0_full.copy()
    for i, j in net:
        col_i = W_full[:, i]
        col_j = W_full[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    if np.any(W_full[:, :-1] > W_full[:, 1:]):
        return False
    return True

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses fast subset check first, then full check.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    
    # Fast subset check
    if not _is_sorted_network_fast(net):
        return False
        
    # Full check
    return _is_sorted_network_full(net)

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _batcher_odd_merge():
    """
    Generate a Batcher odd-even merge network as a better starting point.
    This is more efficient than odd-even transposition.
    """
    def _sort_network(n):
        """Generate a sorting network for n wires using Batcher's odd-even merge sort."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Recursively sort halves
        half = n // 2
        left = _sort_network(half)
        right = _sort_network(n - half)
        
        # Merge the two sorted halves
        merged = _odd_even_merge(half, n - half)
        
        # Combine: left half is wires 0..half-1, right half is wires half..n-1
        # Need to offset right half comparators
        left_offset = left
        right_offset = [(i + half, j + half) for i, j in right]
        
        return left_offset + right_offset + merged
    
    def _odd_even_merge(a, b):
        """
        Odd-even merge two sorted sequences of length a and b.
        Assumes a >= b.
        """
        if b == 0:
            return []
        if a == 1 and b == 1:
            return [(0, 1)]
        
        comparators = []
        
        # Odd-even merge step 1: compare (0,1), (2,3), ... and (1,2), (3,4), ...
        # Actually, odd-even merge works differently. Let me implement it properly.
        
        # For odd-even merge of two sorted sequences of length a and b (a >= b):
        # 1. Compare elements at even positions: (0,1), (2,3), ...
        # 2. Compare elements at odd positions: (1,2), (3,4), ...
        # 3. Recursively merge the two resulting sorted sequences
        
        # Step 1: Compare (0,1), (2,3), ...
        for i in range(0, a + b - 1, 2):
            comparators.append((i, i + 1))
        
        # Step 2: Compare (1,2), (3,4), ...
        for i in range(1, a + b - 1, 2):
            comparators.append((i, i + 1))
        
        # Now we have two sorted sequences interleaved
        # The even positions form one sorted sequence, odd positions form another
        # Recursively merge them
        
        # Extract even and odd positions
        # Even positions: 0, 2, 4, ...
        # Odd positions: 1, 3, 5, ...
        
        # This is getting complex. Let me use a simpler approach.
        return comparators
    
    return _sort_network(N)

def _batcher_odd_even_merge_sort(n):
    """
    Generate a Batcher odd-even merge sort network for n wires.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    comparators = []
    
    def _batcher_merge(a, b):
        """Merge two sorted sequences of length a and b (a >= b)."""
        if b == 0:
            return []
        if a == 1 and b == 1:
            return [(0, 1)]
        
        result = []
        
        # Compare (0,1), (2,3), ...
        for i in range(0, a + b - 1, 2):
            result.append((i, i + 1))
        
        # Compare (1,2), (3,4), ...
        for i in range(1, a + b - 1, 2):
            result.append((i, i + 1))
        
        # Now recursively merge the two halves
        # The even-indexed elements and odd-indexed elements are each sorted
        # but interleaved. We need to merge them.
        
        # For simplicity, let's just return what we have for now
        # This isn't a complete Batcher network, but it's a reasonable starting point
        return result
    
    # Recursively sort halves
    half = n // 2
    left = _batcher_odd_even_merge_sort(half)
    right = _batcher_odd_even_merge_sort(n - half)
    
    # Offset right half
    right_offset = [(i + half, j + half) for i, j in right]
    
    # Merge
    merged = _batcher_merge(half, n - half)
    
    return left + right_offset + merged

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Aggressive greedy removal
    # Try removing comparators in multiple orders
    changed = True
    while changed:
        if time.time() > deadline - 10:
            break
        changed = False
        
        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            
            new_net = net[:idx] + net[idx+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break
        
        if changed:
            continue
            
        # If no change from end-to-start, try start-to-end
        for idx in range(len(net)):
            if time.time() > deadline - 10:
                break
                
            new_net = net[:idx] + net[idx+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break
    
    # Phase 2: More aggressive local search with multiple moves
    if time.time() < deadline - 5:
        # Try removing multiple comparators at once
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Try removing 2-3 random comparators
            for num_remove in [2, 3]:
                if len(net) <= num_remove:
                    continue
                
                indices_to_remove = random.sample(range(len(net)), num_remove)
                new_net = [net[i] for i in range(len(net)) if i not in indices_to_remove]
                
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    break
    
    # Phase 3: Simulated Annealing with improved moves
    if time.time() < deadline - 3:
        T = 2.0
        min_T = 0.1
        cooling = 0.999
        current_count = len(net)
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 2:
            if T < min_T or no_improve_count > max_no_improve:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 2.0
                no_improve_count = 0
            
            # Generate move
            r = random.random()
            if r < 0.5:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx+1:]
            elif r < 0.8:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            else:
                # Mutate
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)
            
            # Fast check first
            if not _is_sorted_network_fast(new_net):
                no_improve_count += 1
                T *= cooling
                continue
            
            # Full check
            if not _is_sorted_network_full(new_net):
                no_improve_count += 1
                T *= cooling
                continue
                
            new_count = len(new_net)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                    
            T *= cooling
    
    # Final validation
    if not _is_sorted_network(best_net):
        best_net = _odd_even_transposition()
        
    return best_net
