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
    
    def is_valid_addition(current_sums, w):
        """Check if adding weight w to current_sums keeps all sums distinct."""
        new_sums = {s + w for s in current_sums}
        if current_sums & new_sums:
            return False, None
        return True, current_sums | new_sums

    def greedy_search(order_indices):
        """
        Greedily add elements in the given order.
        Returns a list of original indices.
        """
        current_sums = {0}
        chosen = []
        for idx in order_indices:
            w = weights[idx]
            valid, new_sums = is_valid_addition(current_sums, w)
            if valid:
                current_sums = new_sums
                chosen.append(idx)
        return chosen

    best_set = []
    best_k = 0

    # 1. Initialize with a good greedy solution
    # Try the standard sorted order (descending weight = ascending x)
    sorted_indices = sorted(indices, key=lambda i: weights[i], reverse=True)
    cand = greedy_search(sorted_indices)
    if len(cand) > best_k:
        best_k = len(cand)
        best_set = cand[:]

    # Try multiple random orders
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        random_order = indices[:]
        random.shuffle(random_order)
        cand = greedy_search(random_order)
        if len(cand) > best_k:
            best_k = len(cand)
            best_set = cand[:]

    # 2. Local search: try to improve by swapping or adding/removing
    def local_search(initial_set):
        """Try to improve the set via local moves."""
        nonlocal best_set, best_k
        
        current = set(initial_set)
        current_sums = {0}
        for idx in current:
            w = weights[idx]
            valid, new_sums = is_valid_addition(current_sums, w)
            if not valid:
                return  # Should not happen
            current_sums = new_sums
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding an element not in current
            for idx in indices:
                if idx in current:
                    continue
                w = weights[idx]
                valid, new_sums = is_valid_addition(current_sums, w)
                if valid:
                    current.add(idx)
                    current_sums = new_sums
                    improved = True
                    if len(current) > best_k:
                        best_k = len(current)
                        best_set = list(current)
                    break
            
            # Try swapping: remove one, add another
            if not improved:
                for remove_idx in list(current):
                    if time.time() > deadline - 5:
                        return
                    # Compute sums without remove_idx
                    # This is expensive; use a simpler heuristic
                    w_remove = weights[remove_idx]
                    # Try to find an add_idx that works after removal
                    for add_idx in indices:
                        if add_idx in current:
                            continue
                        w_add = weights[add_idx]
                        # Check if removing remove_idx and adding add_idx keeps distinctness
                        # Approximate: check if w_add can be added to (current_sums - contributions of remove_idx)
                        # This is complex; skip for now and rely on pure add
                        pass
        return current

    local_search(best_set)

    # 3. Exact search to see if we can improve best_k
    def search(k, start, current_sums, current_indices, order):
        """
        Backtracking search for a set of size k.
        """
        nonlocal best_set, best_k
        
        if time.time() > deadline:
            return False
            
        if len(current_indices) == k:
            best_set = current_indices[:]
            best_k = k
            return True
            
        # Pruning: not enough elements left
        remaining = len(order) - start
        if remaining < k - len(current_indices):
            return False
            
        # Option 1: Take the next element
        idx = order[start]
        w = weights[idx]
        valid, new_sums = is_valid_addition(current_sums, w)
        if valid:
            if search(k, start + 1, new_sums, current_indices + [idx], order):
                return True
                
        # Option 2: Skip the next element
        if search(k, start + 1, current_sums, current_indices, order):
            return True
            
        return False

    # Try to increase the size using sorted order
    target_k = best_k + 1
    while target_k <= 22:
        if time.time() > deadline - 10:
            break
            
        order = sorted(indices, key=lambda i: weights[i], reverse=True)
        found = search(target_k, 0, {0}, [], order)
        
        if found:
            target_k += 1
        else:
            if time.time() > deadline:
                break
            else:
                break

    # Convert back to original values (1..22)
    if not best_set:
        best_set = [0]
        
    result = [i + 1 for i in best_set]
    
    if not result:
        result = [1]
        
    return result
