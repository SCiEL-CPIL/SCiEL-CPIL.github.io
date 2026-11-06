import time
from math import lcm
import bisect
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are L // d
    weights = [L // d for d in denominators]
    
    # Sort by weight descending (smallest denominator first) for better pruning
    # Index 0 -> denom 2, weight L/2 (largest)
    # Index 46 -> denom 48, weight L/48 (smallest)
    # Already in descending weight order!
    
    # Precompute suffix sums: suffix_sum[i] = sum of weights[i:]
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    # Branch and bound DFS
    best_sum = 0
    best_mask = 0
    current_mask = 0
    current_sum = 0
    
    def dfs(idx, current_sum, current_mask):
        nonlocal best_sum, best_mask
        
        # Pruning: if current_sum + remaining can't beat best, prune
        if current_sum + suffix_sum[idx] <= best_sum:
            return
        
        if idx == n:
            if current_sum > best_sum and current_sum < L:
                best_sum = current_sum
                best_mask = current_mask
            return
        
        # Option 1: Skip this element
        dfs(idx + 1, current_sum, current_mask)
        
        # Option 2: Include this element
        new_sum = current_sum + weights[idx]
        if new_sum < L:
            dfs(idx + 1, new_sum, current_mask | (1 << idx))
    
    # Run DFS with time check
    # For 47 items, full DFS is too slow, so we use iterative deepening or limit depth
    # Actually, let's try a different approach: use the meet-in-the-middle but with
    # better local search, OR use a smarter B&B with early termination
    
    # Let's do a hybrid: use B&B but with a node limit, then polish with local search
    node_count = [0]
    max_nodes = 5000000  # Limit nodes to stay within time
    
    def dfs_limited(idx, current_sum, current_mask):
        nonlocal best_sum, best_mask
        
        node_count[0] += 1
        if node_count[0] > max_nodes:
            return False  # Signal timeout
        
        if time.time() > deadline:
            return False
        
        # Pruning: if current_sum + remaining can't beat best, prune
        if current_sum + suffix_sum[idx] <= best_sum:
            return True
        
        if idx == n:
            if current_sum > best_sum and current_sum < L:
                best_sum = current_sum
                best_mask = current_mask
            return True
        
        # Option 1: Skip this element
        if not dfs_limited(idx + 1, current_sum, current_mask):
            return False
        
        # Option 2: Include this element
        new_sum = current_sum + weights[idx]
        if new_sum < L:
            if not dfs_limited(idx + 1, new_sum, current_mask | (1 << idx)):
                return False
        
        return True
    
    # Start DFS
    dfs_limited(0, 0, 0)
    
    # Reconstruct indices from mask
    best_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            best_indices.append(i)
    
    if not best_indices:
        best_indices = [0]
    
    # Local search polish
    if time.time() < deadline - 5.0:
        current_sum = sum(weights[i] for i in best_indices)
        in_set = set(best_indices)
        not_in_set = [i for i in range(n) if i not in in_set]
        
        # Multiple restarts with best-swap
        for restart in range(100):
            if time.time() > deadline - 5.0:
                break
            
            improved = True
            max_iter = 50
            for _ in range(max_iter):
                if time.time() > deadline - 5.0:
                    break
                improved = False
                
                # Try all swaps, pick the best one
                best_swap = None
                best_new_sum = current_sum
                
                for rem in list(in_set):
                    for add in not_in_set:
                        new_sum = current_sum - weights[rem] + weights[add]
                        if new_sum < L and new_sum > best_new_sum:
                            best_new_sum = new_sum
                            best_swap = (rem, add)
                
                if best_swap:
                    rem, add = best_swap
                    best_indices.remove(rem)
                    best_indices.append(add)
                    in_set.remove(rem)
                    not_in_set.remove(add)
                    not_in_set.append(rem)
                    in_set.add(add)
                    current_sum = best_new_sum
                    improved = True
                else:
                    break
            
            if not improved:
                break
    
    # Final validation
    s = sum(weights[i] for i in best_indices)
    if s >= L:
        # Remove elements until valid
        while best_indices and s >= L:
            idx_to_remove = best_indices.pop()
            s -= weights[idx_to_remove]
    
    if not best_indices:
        best_indices = [0]
    
    print(f"Best sum: {s}/{L}")
    print(f"Indices: {best_indices}")
    
    return best_indices
