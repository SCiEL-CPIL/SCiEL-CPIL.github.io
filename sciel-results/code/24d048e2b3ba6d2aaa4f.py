import time
from math import lcm

def construct():
    # Denominators are 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2 to 48
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are L / d_i
    weights = [L // d for d in denominators]
    
    # Sort indices by weight descending for B&B
    sorted_indices = sorted(range(n), key=lambda i: -weights[i])
    sorted_weights = [weights[i] for i in sorted_indices]
    
    # Precompute suffix sums for pruning
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_weights[i]
        
    best_sum = 0
    best_indices = []
    
    start_time = time.time()
    deadline_bb = start_time + 45.0  # Leave 15s for local search
    
    # Global state for recursion
    current_sum = 0
    current_set = []
    
    def search(idx):
        nonlocal best_sum, best_indices, current_sum, current_set
        
        # Time check
        if time.time() > deadline_bb:
            return
            
        # Pruning: even if we take all remaining, we can't beat best
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # Base case
        if idx == n:
            if current_sum < L and current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(current_set)
            return
            
        # Option 1: Include item at idx
        w = sorted_weights[idx]
        if current_sum + w < L:
            current_set.append(sorted_indices[idx])
            current_sum += w
            search(idx + 1)
            current_set.pop()
            current_sum -= w
            
        # Option 2: Exclude item at idx
        # Pruning for exclusion: if we skip this, can we still beat best?
        if current_sum + suffix_sums[idx + 1] > best_sum:
            search(idx + 1)

    # Initial greedy to get a decent lower bound
    greedy_sum = 0
    greedy_set = []
    for i in range(n):
        w = sorted_weights[i]
        if greedy_sum + w < L:
            greedy_sum += w
            greedy_set.append(sorted_indices[i])
    best_sum = greedy_sum
    best_indices = list(greedy_set)
    
    # Run B&B
    search(0)
    
    # Local Search Phase
    deadline_ls = start_time + 58.0  # Leave 2s margin
    
    # Initialize state from best_indices
    current_set_ls = list(best_indices)
    in_set = [False] * n
    for idx in current_set_ls:
        in_set[idx] = True
        
    current_sum_ls = best_sum
    
    # Precompute list of indices not in set for faster iteration?
    # Or just iterate 0..n-1 and check in_set.
    
    while time.time() < deadline_ls:
        best_gain = 0
        best_move = None
        
        # Try Add moves
        for j in range(n):
            if not in_set[j]:
                w_j = weights[j]
                if current_sum_ls + w_j < L:
                    gain = w_j
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('add', j)
        
        # Try Swap moves
        # To optimize, we can sort the items in S by weight ascending (remove small)
        # and items not in S by weight descending (add large)
        # But n is small (47), so O(n^2) is fine.
        
        for i in range(n):
            if in_set[i]:
                w_i = weights[i]
                for j in range(n):
                    if not in_set[j]:
                        w_j = weights[j]
                        new_sum = current_sum_ls - w_i + w_j
                        if new_sum < L and new_sum > current_sum_ls:
                            gain = new_sum - current_sum_ls
                            if gain > best_gain:
                                best_gain = gain
                                best_move = ('swap', i, j)
        
        if best_move is None:
            break
            
        # Apply best move
        if best_move[0] == 'add':
            j = best_move[1]
            in_set[j] = True
            current_set_ls.append(j)
            current_sum_ls += weights[j]
        else: # swap
            i, j = best_move[1], best_move[2]
            in_set[i] = False
            in_set[j] = True
            current_set_ls.remove(i)
            current_set_ls.append(j)
            current_sum_ls -= weights[i]
            current_sum_ls += weights[j]
            
        # Update global best
        if current_sum_ls > best_sum:
            best_sum = current_sum_ls
            best_indices = list(current_set_ls)
            
    # Ensure we have a valid non-empty set
    if not best_indices:
        best_indices = [0] # 1/2 is always valid
        
    return best_indices
