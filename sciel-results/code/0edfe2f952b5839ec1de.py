"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe (evaluator kills at 90s, but we want margin)
    
    # Compute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    
    # Weights: w[i] = L // (i+1) for i in 0..21
    # Corresponds to x = 1..22
    weights = [L // (i + 1) for i in range(22)]
    
    best_set = []
    best_size = 0
    
    def update_best_from_mask(mask, order_indices, size):
        nonlocal best_set, best_size
        if size > best_size:
            best_size = size
            s = set()
            for i in range(22):
                if (mask >> i) & 1:
                    s.add(order_indices[i] + 1)
            best_set = list(s)
    
    def is_valid_addition(current_sums, w):
        """Check if adding weight w to current_sums keeps all sums distinct."""
        new_sums = {s + w for s in current_sums}
        if current_sums & new_sums:
            return False, None
        return True, current_sums | new_sums
    
    def greedy_build(order_indices, order_weights):
        """Greedy: add elements in order if they don't cause collision."""
        current_sums = {0}
        mask = 0
        size = 0
        for i in range(22):
            w = order_weights[i]
            valid, new_sums = is_valid_addition(current_sums, w)
            if valid:
                current_sums = new_sums
                mask |= (1 << i)
                size += 1
        update_best_from_mask(mask, order_indices, size)
        return size
    
    def backtrack_search(k, start, current_sums, current_mask, current_size, order_indices, order_weights, remaining_weights):
        """
        Try to find a subset of size k from order_indices[start:].
        remaining_weights: suffix sums for pruning
        """
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return False
            
        if current_size == k:
            if k > best_size:
                update_best_from_mask(current_mask, order_indices, k)
            return True
            
        # Pruning: not enough elements left
        if 22 - start < k - current_size:
            return False
        
        # Pruning: even taking all remaining, can't reach k
        needed = k - current_size
        if needed > 22 - start:
            return False
        
        # Try taking the current element first (prefer larger weights for dissociation)
        w = order_weights[start]
        valid, new_sums = is_valid_addition(current_sums, w)
        if valid:
            new_mask = current_mask | (1 << start)
            if backtrack_search(k, start + 1, new_sums, new_mask, current_size + 1, order_indices, order_weights, remaining_weights):
                return True
        
        # Try skipping
        if backtrack_search(k, start + 1, current_sums, current_mask, current_size, order_indices, order_weights, remaining_weights):
            return True
            
        return False

    # Create multiple orderings for different search strategies
    # Descending weight (smallest x first)
    indices_desc = sorted(range(22), key=lambda i: weights[i], reverse=True)
    sorted_weights_desc = [weights[i] for i in indices_desc]
    
    # Ascending weight (largest x first)
    indices_asc = sorted(range(22), key=lambda i: weights[i])
    sorted_weights_asc = [weights[i] for i in indices_asc]
    
    # Try greedy first to get a baseline
    greedy_build(indices_desc, sorted_weights_desc)
    greedy_build(indices_asc, sorted_weights_asc)
    
    # Now try backtracking from best_size+1 upward, and also from high k downward
    # First, try to find a set larger than what greedy found
    for k in range(best_size + 1, 23):
        if time.time() > deadline:
            break
        
        # Try descending order
        if backtrack_search(k, 0, {0}, 0, 0, indices_desc, sorted_weights_desc, []):
            if time.time() > deadline:
                break
            # Continue trying to find even larger sets
            continue
        
        # Try ascending order
        if time.time() < deadline - 3:
            if backtrack_search(k, 0, {0}, 0, 0, indices_asc, sorted_weights_asc, []):
                continue
    
    # If no set found, return a default
    if not best_set:
        best_set = [1]
        
    return best_set
# EVOLVE-BLOCK-END
