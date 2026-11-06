"""Combined program: Integer-weighted Local Search with Random Restarts."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import random

def construct():
    start_time = time.time()
    timeout = 58.0  # Safety margin for 60s limit
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of 2..48
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(2, 49))
    
    # Precompute integer weights
    # index i corresponds to denominator i+2
    # weight[i] = L // (i+2)
    weights = [L // d for d in denominators]
    
    # Sort indices by weight descending to help greedy initialization
    # But we need to return original indices.
    # Let's keep original indices for the output, but maybe sort for search heuristics?
    # For local search, we operate on the set of indices.
    
    # Helper to check time
    def time_exceeded():
        return time.time() - start_time > timeout

    # Greedy Initialization
    # Strategy: Sort by weight descending, add if fits.
    # This tends to pack large items first, leaving small deficit.
    def greedy_init():
        selected = set()
        current_sum = 0
        # Sort indices by weight descending
        sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
        for i in sorted_indices:
            if current_sum + weights[i] < L:
                selected.add(i)
                current_sum += weights[i]
        return selected, current_sum

    # Local Search using Integer Weights
    # Objective: Minimize Deficit D = L - Sum
    # Moves:
    # 1. Add i: if Sum + w_i < L. New D = D - w_i. (Always improves if possible)
    # 2. Swap i (out) with j (in): if Sum - w_i + w_j < L. New D = D - w_j + w_i.
    #    We want to minimize New D, so maximize w_j - w_i subject to w_j - w_i < D.
    
    def local_search(initial_selected, initial_sum, max_iters=100000):
        selected = set(initial_selected)
        current_sum = initial_sum
        current_deficit = L - current_sum
        
        # Precompute list of all indices not in selected for faster lookup?
        # Or just check membership.
        
        # For swap optimization:
        # We want to find j not in selected, i in selected such that:
        # w_j < w_i + current_deficit
        # and w_j - w_i is maximized.
        
        # To speed up swap search, sort all indices by weight.
        # indices_by_weight = sorted(range(n), key=lambda i: weights[i])
        
        indices_by_weight = sorted(range(n), key=lambda i: weights[i])
        
        improved = True
        iters = 0
        while improved and iters < max_iters and not time_exceeded():
            improved = False
            iters += 1
            
            # 1. Try single additions
            # If we can add any item, deficit decreases.
            # We should add the item that causes the smallest decrease in deficit?
            # No, adding ANY item decreases deficit.
            # But adding a small item might block a larger swap later?
            # Actually, if we can add an item, the deficit strictly decreases.
            # So we should always add if possible?
            # Yes, for a fixed set, adding a valid item is always an improvement.
            # However, adding an item changes the set, potentially invalidating previous swaps?
            # No, it just changes the state.
            # So, greedily add any item that fits?
            # Better: Find the item that fits and has the LARGEST weight?
            # If we add the largest weight that fits, we maximize the immediate deficit reduction.
            
            best_add_idx = -1
            best_add_w = 0
            for i in range(n):
                if i not in selected:
                    if current_sum + weights[i] < L:
                        if weights[i] > best_add_w:
                            best_add_w = weights[i]
                            best_add_idx = i
            
            if best_add_idx != -1:
                selected.add(best_add_idx)
                current_sum += best_add_w
                current_deficit -= best_add_w
                improved = True
                continue # Restart loop to check for more adds/swaps
            
            # 2. Try 1-1 Swaps
            # We want to replace i (in) with j (out) such that:
            # w_j < w_i + current_deficit
            # and (w_j - w_i) is maximized.
            
            best_swap_i = -1
            best_swap_j = -1
            best_gain = 0 # Gain = w_j - w_i
            
            # To find best swap efficiently:
            # For each i in selected, we want the largest w_j (j not in selected)
            # such that w_j < w_i + current_deficit.
            
            # Since indices_by_weight is sorted, we can binary search or just iterate?
            # N is 47. Iterating is fine.
            
            # Optimization: Pre-compute list of 'available' weights and indices?
            # Available changes as we swap.
            
            avail_indices = [i for i in range(n) if i not in selected]
            
            for i in selected:
                w_i = weights[i]
                limit = w_i + current_deficit
                # Find j in avail_indices such that weights[j] < limit and weights[j] is max
                # Since avail_indices is small, just loop
                best_j_for_i = -1
                max_w_j = 0
                for j in avail_indices:
                    w_j = weights[j]
                    if w_j < limit:
                        if w_j > max_w_j:
                            max_w_j = w_j
                            best_j_for_i = j
                
                if best_j_for_i != -1:
                    gain = max_w_j - w_i
                    if gain > best_gain:
                        best_gain = gain
                        best_swap_i = i
                        best_swap_j = best_j_for_i
            
            if best_swap_i != -1:
                # Perform swap
                selected.remove(best_swap_i)
                selected.add(best_swap_j)
                current_sum += weights[best_swap_j] - weights[best_swap_i]
                current_deficit -= best_gain
                improved = True
                continue
            
            # 3. Try 2-2 Swaps (Limited search for speed)
            # Remove i1, i2; Add j1, j2
            # Condition: w_j1 + w_j2 < w_i1 + w_i2 + current_deficit
            # Maximize (w_j1 + w_j2) - (w_i1 + w_i2)
            
            # This is O(N^4) roughly. 47^4 is too big.
            # We can restrict to "promising" pairs.
            # Or just do a limited random search for 2-2 swaps.
            
            if iters % 10 == 0: # Do this less frequently
                # Random 2-2 swap attempt
                if len(selected) >= 2 and len(avail_indices) >= 2:
                    # Pick random i1, i2 from selected
                    sel_list = list(selected)
                    i1 = random.choice(sel_list)
                    i2 = random.choice([x for x in sel_list if x != i1])
                    
                    # Pick random j1, j2 from avail
                    j1 = random.choice(avail_indices)
                    j2 = random.choice([x for x in avail_indices if x != j1])
                    
                    w_out = weights[i1] + weights[i2]
                    w_in = weights[j1] + weights[j2]
                    
                    if w_in < w_out + current_deficit:
                        gain = w_in - w_out
                        if gain > 0:
                            selected.remove(i1)
                            selected.remove(i2)
                            selected.add(j1)
                            selected.add(j2)
                            current_sum += w_in - w_out
                            current_deficit -= gain
                            improved = True
                            continue
            
            # If no improvement, try a "kick" to escape local minimum
            # Randomly remove a few items and re-add greedily?
            # Or just break and rely on restarts.
            break
        
        return selected, current_sum, current_deficit

    # Main Loop: Multiple Restarts
    best_indices = []
    best_sum = 0
    best_deficit = L
    
    # Run several restarts
    # 1. Greedy
    # 2. Random Greedy (shuffle weights before sorting? No, shuffle selection order)
    # 3. Random Valid Subset
    
    num_restarts = 100 # Will be limited by time
    
    for r in range(num_restarts):
        if time_exceeded():
            break
        
        if r == 0:
            # Standard Greedy
            sel, s = greedy_init()
        else:
            # Randomized Greedy: Shuffle the order of consideration
            indices = list(range(n))
            random.shuffle(indices)
            # Sort by weight descending, but with some noise?
            # Or just pick randomly from available that fits?
            # Let's do: Pick random item that fits.
            sel = set()
            s = 0
            available = list(range(n))
            random.shuffle(available)
            
            # To make it better, sort available by weight descending, but then shuffle ties?
            # Actually, just sorting by weight descending is the best greedy.
            # For diversity, let's randomly drop some high-weight items in the initial greedy?
            
            # Alternative: Random Subset Initialization
            # Pick k random items, if sum < L, use as start.
            # Or: Greedy with random tie-breaking.
            
            # Let's use: Greedy, but randomly skip the first K items?
            # Or: Start with empty, add random items that fit until no more fit.
            
            # Let's try: Randomized Greedy
            # Sort by weight descending
            sorted_idx = sorted(range(n), key=lambda i: weights[i], reverse=True)
            
            # Randomly perturb the order?
            # Let's just shuffle the input list before sorting? No.
            # Let's try: Start with a random subset of size 5-10, then greedily add.
            
            sel = set()
            s = 0
            # Random start
            start_size = random.randint(0, 5)
            if start_size > 0:
                start_items = random.sample(range(n), start_size)
                for i in start_items:
                    if s + weights[i] < L:
                        sel.add(i)
                        s += weights[i]
            
            # Now greedily add best fitting items
            # We need to re-evaluate "best fitting"
            # Simple greedy: iterate sorted_idx, add if fits
            for i in sorted_idx:
                if i not in sel:
                    if s + weights[i] < L:
                        sel.add(i)
                        s += weights[i]
        
        # Local Search
        sel, s, d = local_search(sel, s)
        
        if d < best_deficit:
            best_deficit = d
            best_sum = s
            best_indices = list(sel)
        
        # Check if we found a perfect solution (deficit 0 is not allowed, strictly below 1)
        # Min deficit is 1/L? No, weights are integers. Min deficit is 1.
        # If best_deficit == 1, we can stop early.
        if best_deficit == 1:
            break

    if not best_indices:
        # Fallback: Single smallest weight?
        # 1/48 is the smallest reciprocal, largest weight.
        # We want sum < 1.
        # Just return [0] (1/2) as fallback, deficit 1/2.
        return [0]
    
    return best_indices
# EVOLVE-BLOCK-END
