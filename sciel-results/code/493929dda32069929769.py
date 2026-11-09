import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    weights = [L // d for d in denominators]
    
    best_sum = 0
    best_indices = None
    
    rng = random.Random(42)
    
    def local_search(selected, total):
        selected_set = set(selected)
        improved = True
        max_iter = 2000
        
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            # 1. Try 1-add
            added = False
            for j in range(n):
                if j not in selected_set:
                    if total + weights[j] < L:
                        selected.append(j)
                        selected_set.add(j)
                        total += weights[j]
                        improved = True
                        added = True
                        break
            
            if added:
                continue
                
            # 2. Try 1-swap (remove i, add j)
            best_swap_gain = 0
            best_swap_i = -1
            best_swap_j = -1
            
            for i in selected:
                # If we remove i, space available is L - (total - w[i])
                # We need w[j] < L - total + w[i]
                limit_j = L - total + weights[i]
                for j in range(n):
                    if j not in selected_set:
                        if weights[j] < limit_j:
                            gain = weights[j] - weights[i]
                            if gain > best_swap_gain:
                                best_swap_gain = gain
                                best_swap_i = i
                                best_swap_j = j
            
            if best_swap_i != -1:
                selected.remove(best_swap_i)
                selected_set.remove(best_swap_i)
                selected.append(best_swap_j)
                selected_set.add(best_swap_j)
                total += best_swap_gain
                improved = True
                continue
                
            # 3. Try 1-for-2-swap (remove i, add j, k)
            # Gain: w[j] + w[k] - w[i]
            # Constraint: total - w[i] + w[j] + w[k] < L
            # => w[j] + w[k] < L - total + w[i]
            
            best_12_swap_gain = 0
            best_12_i = -1
            best_12_j = -1
            best_12_k = -1
            
            unselected = [x for x in range(n) if x not in selected_set]
            
            for i in selected:
                limit_jk = L - total + weights[i]
                # Find best pair j, k in unselected such that w[j] + w[k] < limit_jk
                # and w[j] + w[k] - w[i] is maximized.
                
                # Since unselected is small, we can just iterate pairs
                # But we can optimize: sort unselected by weight?
                # For now, simple O(U^2) where U is unselected count.
                for idx_j in range(len(unselected)):
                    j = unselected[idx_j]
                    wj = weights[j]
                    if wj >= limit_jk:
                        continue
                    for idx_k in range(idx_j + 1, len(unselected)):
                        k = unselected[idx_k]
                        wk = weights[k]
                        if wj + wk >= limit_jk:
                            continue
                        gain = wj + wk - weights[i]
                        if gain > best_12_swap_gain:
                            best_12_swap_gain = gain
                            best_12_i = i
                            best_12_j = j
                            best_12_k = k
            
            if best_12_i != -1:
                selected.remove(best_12_i)
                selected_set.remove(best_12_i)
                selected.append(best_12_j)
                selected_set.add(best_12_j)
                selected.append(best_12_k)
                selected_set.add(best_12_k)
                total += best_12_swap_gain
                improved = True
                continue
                
            # 4. Try 2-for-1-swap (remove i, k; add j)
            # Gain: w[j] - w[i] - w[k]
            # Constraint: total - w[i] - w[k] + w[j] < L
            # => w[j] < L - total + w[i] + w[k]
            
            best_21_swap_gain = 0
            best_21_i = -1
            best_21_k = -1
            best_21_j = -1
            
            # Iterate pairs in selected
            sel_list = list(selected)
            for idx_i in range(len(sel_list)):
                i = sel_list[idx_i]
                for idx_k in range(idx_i + 1, len(sel_list)):
                    k = sel_list[idx_k]
                    limit_j = L - total + weights[i] + weights[k]
                    # Find best j in unselected with w[j] < limit_j
                    # Maximize w[j] - w[i] - w[k] => Maximize w[j]
                    best_j = -1
                    best_wj = -1
                    for j in unselected:
                        if weights[j] < limit_j and weights[j] > best_wj:
                            best_wj = weights[j]
                            best_j = j
                            
                    if best_j != -1:
                        gain = best_wj - weights[i] - weights[k]
                        if gain > best_21_swap_gain:
                            best_21_swap_gain = gain
                            best_21_i = i
                            best_21_k = k
                            best_21_j = best_j
            
            if best_21_i != -1:
                selected.remove(best_21_i)
                selected_set.remove(best_21_i)
                selected.remove(best_21_k)
                selected_set.remove(best_21_k)
                selected.append(best_21_j)
                selected_set.add(best_21_j)
                total += best_21_swap_gain
                improved = True
                continue

        return selected, total

    # Multiple restarts
    restarts = 0
    while time.time() < deadline - 1.0:
        restarts += 1
        
        # Alternate initialization strategies
        if restarts % 3 == 0:
            # Random
            perm = list(range(n))
            rng.shuffle(perm)
            order = perm
        elif restarts % 3 == 1:
            # Sorted by weight descending
            order = sorted(range(n), key=lambda i: weights[i], reverse=True)
        else:
            # Sorted by weight ascending (might help find different local optima?)
            order = sorted(range(n), key=lambda i: weights[i])
            
        selected = []
        selected_set = set()
        total = 0
        
        for i in order:
            if total + weights[i] < L:
                selected.append(i)
                selected_set.add(i)
                total += weights[i]
        
        selected, total = local_search(selected, total)
        
        if total > best_sum:
            best_sum = total
            best_indices = selected[:]
            
    if not best_indices:
        best_indices = [0]
        
    return best_indices
