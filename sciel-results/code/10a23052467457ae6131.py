import time
from math import lcm
from functools import reduce
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 58s limit to be safe against 60s kill
    
    # Compute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    
    # Weights: w[i] = L // (i+1) for i in 0..21
    # Corresponds to x = 1..22
    weights = [L // (i + 1) for i in range(22)]
    
    # Indices 0..21 correspond to values 1..22
    indices = list(range(22))
    
    # Heuristic: Sort by descending weight (ascending x)
    # This is often a good ordering for packing/dissociation
    sorted_indices = sorted(indices, key=lambda i: weights[i], reverse=True)
    sorted_weights = [weights[i] for i in sorted_indices]

    best_set = []
    best_k = 0
    
    def is_valid_addition(current_sums, w):
        """Check if adding weight w to current_sums keeps all sums distinct."""
        new_sums = {s + w for s in current_sums}
        if current_sums & new_sums:
            return False, None
        return True, current_sums | new_sums

    def greedy_search(order_indices, order_weights):
        """
        Greedily add elements in the given order.
        Returns a list of original indices.
        """
        current_sums = {0}
        chosen = []
        for idx, w in zip(order_indices, order_weights):
            valid, new_sums = is_valid_addition(current_sums, w)
            if valid:
                current_sums = new_sums
                chosen.append(idx)
        return chosen

    # 1. Initialize with a good greedy solution
    # Try the standard sorted order
    cand = greedy_search(sorted_indices, sorted_weights)
    if len(cand) > best_k:
        best_k = len(cand)
        best_set = cand[:]
        
    # Try a few random orders to see if we can get a larger greedy set
    for _ in range(10):
        if time.time() > deadline - 20: # Keep 20s for exact search
            break
        random.shuffle(sorted_indices)
        # Re-sort weights to match? No, weights are tied to indices.
        # We need to iterate in the shuffled index order.
        shuffled_indices = sorted_indices[:]
        shuffled_weights = [weights[i] for i in shuffled_indices]
        cand = greedy_search(shuffled_indices, shuffled_weights)
        if len(cand) > best_k:
            best_k = len(cand)
            best_set = cand[:]

    # 2. Exact search to see if we can improve best_k
    # We try to find a set of size best_k + 1
    # If found, we try best_k + 2, etc.
    
    def search(k, start, current_sums, current_indices):
        """
        Backtracking search for a set of size k.
        Uses the sorted_indices/sorted_weights for a consistent ordering.
        """
        nonlocal best_set, best_k
        
        if time.time() > deadline:
            return False
            
        if len(current_indices) == k:
            # Found a valid set of size k
            best_set = current_indices[:]
            best_k = k
            return True
            
        # Pruning: not enough elements left
        if 22 - start < k - len(current_indices):
            return False
            
        # Option 1: Take the next element
        idx = sorted_indices[start]
        w = sorted_weights[start]
        valid, new_sums = is_valid_addition(current_sums, w)
        if valid:
            if search(k, start + 1, new_sums, current_indices + [idx]):
                return True
                
        # Option 2: Skip the next element
        if search(k, start + 1, current_sums, current_indices):
            return True
            
        return False

    # Try to increase the size
    target_k = best_k + 1
    while target_k <= 22:
        if time.time() > deadline - 5: # Stop if we're close to deadline
            break
            
        # Reset for search
        found = search(target_k, 0, {0}, [])
        
        if found:
            target_k += 1
        else:
            # If search completed without finding a set, no larger set exists
            # (Assuming search didn't timeout)
            # However, if it timed out, we don't know. 
            # The `search` function returns False on timeout as well.
            # We can distinguish by checking if we hit the deadline.
            if time.time() > deadline:
                break
            else:
                # Search exhausted, no set of this size
                break

    # Convert back to original values (1..22)
    if not best_set:
        best_set = [0] # Fallback to {1}
        
    result = [i + 1 for i in best_set]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result
