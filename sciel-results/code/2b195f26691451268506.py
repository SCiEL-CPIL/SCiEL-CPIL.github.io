import time
import itertools
import random
from functools import lru_cache

# Precompute all triple sums for each pair (i,j) with i<j, store sums
# For a set S, we need:
# 1. x does not divide any triple sum from S
# 2. For each a in S, a does not divide x + u + v for any u,v in S\{a}

def _can_add(x, selected):
    """Check if x can be added to selected without violating the constraint.
    selected: list of selected integers
    """
    n = len(selected)
    
    # Check 1: x must not divide sum of any 3 distinct elements in selected
    if n >= 3:
        # Precompute residues of selected elements modulo x
        # res_idx[r] = list of indices in selected with selected[i] % x == r
        res_idx = [[] for _ in range(x)]
        for i in range(n):
            r = selected[i] % x
            res_idx[r].append(i)
        
        # Iterate over all pairs (i, j)
        for i in range(n):
            si = selected[i]
            for j in range(i + 1, n):
                sj = selected[j]
                s_ij = si + sj
                target_res = (-s_ij) % x
                
                # We need a k such that selected[k] % x == target_res
                # and k != i, k != j
                candidates = res_idx[target_res]
                if not candidates:
                    continue
                
                # Check if there is a candidate k distinct from i and j
                found_k = False
                for k in candidates:
                    if k != i and k != j:
                        found_k = True
                        break
                
                if found_k:
                    return False

    # Check 2: For each a in selected, a must not divide x + u + v for any u, v in selected \ {a}
    for a_idx, a in enumerate(selected):
        # We need u, v in selected \ {a} such that (x + u + v) % a == 0
        # => (u + v) % a == (-x) % a
        target_sum_res = (-x) % a
        
        # Iterate over pairs (u_idx, v_idx) in selected where u_idx != a_idx and v_idx != a_idx
        # To be efficient, we can precompute residues of selected elements modulo a
        # But a changes in the loop, so we must recompute or use a different method.
        # Since n is small, simple double loop is okay, but we avoid slicing.
        
        # Optimization: Use a set of residues?
        # Let's just use a double loop but skip a_idx.
        # We can break early if found.
        
        # Precompute residues for this 'a' to speed up?
        # res_a[i] = selected[i] % a
        # We need res_a[u] + res_a[v] == target_sum_res (mod a)
        
        # Let's collect residues of all elements except a
        # We can just iterate u, v
        for u_idx in range(n):
            if u_idx == a_idx:
                continue
            su = selected[u_idx]
            # We need v such that (su + sv) % a == target_sum_res
            # sv % a == (target_sum_res - su) % a
            target_v_res = (target_sum_res - su) % a
            
            for v_idx in range(u_idx + 1, n):
                if v_idx == a_idx:
                    continue
                sv = selected[v_idx]
                if sv % a == target_v_res:
                    return False
            # Also check v_idx < u_idx?
            # The pair (u, v) is unordered in the sum.
            # My loop `for v_idx in range(u_idx + 1, n)` only checks v > u.
            # This covers all distinct pairs {u, v} exactly once.
            # So this is correct.

    return True

def _greedy(order, deadline):
    """Run greedy in the given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Try to improve the set by swapping or adding elements with better heuristics."""
    selected = list(selected)
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and time.time() < deadline - 2:
        improved = False
        no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
            
        # Try adding any element not in selected
        in_set = set(selected)
        # Sort candidates by value descending (larger numbers are "safer" usually)
        candidates = [x for x in range(192, 0, -1) if x not in in_set]
        
        for x in candidates:
            if time.time() > deadline:
                return selected
            if _can_add(x, selected):
                selected.append(x)
                in_set.add(x)
                improved = True
                no_improve_count = 0
                break
        
        # Try single swap: remove one, add one
        if not improved:
            best_swap_gain = 0
            best_swap_new_set = None
            best_swap_remove_idx = -1
            best_swap_add_x = -1
            
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                candidate = selected[i]
                # Create list without candidate
                # selected_without = selected[:i] + selected[i+1:]
                
                # Instead of creating new list, we can pass a flag or just create it since n is small
                selected_without = selected[:i] + selected[i+1:]
                
                for x in range(192, 0, -1):
                    if x in in_set:
                        continue
                    if not _can_add(x, selected_without):
                        continue
                    
                    # If we can add x, gain is at least 1 (we replaced 1 with 1, size same, but maybe enables more?)
                    # Actually, swap keeps size same. We want to know if it allows *further* additions.
                    # For now, just accept the swap if it doesn't reduce size (it doesn't) 
                    # and hopefully leads to better state.
                    # To decide, we can check if we can add *another* element immediately.
                    new_set = selected_without + [x]
                    new_in_set = set(new_set)
                    
                    # Check if we can add any more
                    extra = 0
                    for y in range(192, 0, -1):
                        if y not in new_in_set:
                            if _can_add(y, new_set):
                                extra = 1
                                break
                                
                    gain = extra # 1 if we can grow, 0 otherwise
                    
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_new_set = new_set
                        best_swap_remove_idx = i
                        best_swap_add_x = x
            
            if best_swap_new_set is not None and best_swap_gain > 0:
                selected = best_swap_new_set
                in_set = set(selected)
                improved = True
                no_improve_count = 0
        
        # Try double swap: remove two, add two
        if not improved and len(selected) >= 2:
            best_double_swap_gain = 0
            best_double_swap_new_set = None
            
            # Limit search space for double swap to avoid explosion
            # Only try removing the "worst" elements? Or just first few?
            # Let's try all pairs but break early if we find a good one.
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(selected)):
                    if time.time() > deadline:
                        break
                    selected_without = selected[:i] + selected[i + 1:j] + selected[j + 1:]
                    
                    for x in range(192, 0, -1):
                        if x in in_set:
                            continue
                        if not _can_add(x, selected_without):
                            continue
                        set_with_x = selected_without + [x]
                        
                        for y in range(192, 0, -1):
                            if y == x or y in in_set:
                                continue
                            if not _can_add(y, set_with_x):
                                continue
                            
                            new_set = set_with_x + [y]
                            new_in_set = set(new_set)
                            
                            # Check if we can add more
                            extra = 0
                            for z in range(192, 0, -1):
                                if z not in new_in_set:
                                    if _can_add(z, new_set):
                                        extra = 1
                                        break
                            
                            gain = extra
                            if gain > best_double_swap_gain:
                                best_double_swap_gain = gain
                                best_double_swap_new_set = new_set
                                if gain > 0:
                                    break
                        if best_double_swap_gain > 0:
                            break
                    if best_double_swap_gain > 0:
                        break
                if best_double_swap_gain > 0:
                    break
            
            if best_double_swap_new_set is not None and best_double_swap_gain > 0:
                selected = best_double_swap_new_set
                in_set = set(selected)
                improved = True
                no_improve_count = 0
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    best = []
    best_size = 0
    
    # Strategy 1: Greedy descending
    result = _greedy(range(192, 0, -1), deadline)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 2: Greedy ascending
    if time.time() < deadline - 10:
        result = _greedy(range(1, 193), deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Strategy 3: Random shuffles
    random.seed(42)
    num_random = 50 # Increase count since _can_add is faster
    for i in range(num_random):
        if time.time() > deadline - 15:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy(order, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Strategy 4: Local search on the best result found so far
    if time.time() < deadline - 10:
        best = _local_search(best, deadline)
        best_size = len(best)
        
    # Strategy 5: Try local search on other top candidates
    # Keep a few best distinct sets
    # For simplicity, just run local search on a few random restarts too if time permits
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            order = list(range(1, 193))
            random.shuffle(order)
            result = _greedy(order, deadline)
            if len(result) >= best_size - 1: # Only if promising
                result = _local_search(result, deadline)
                if len(result) > best_size:
                    best = result
                    best_size = len(result)

    # Final polish: try to add any remaining elements
    if time.time() < deadline - 1:
        in_set = set(best)
        for x in range(192, 0, -1):
            if time.time() > deadline:
                break
            if x not in in_set and _can_add(x, best):
                best.append(x)
                in_set.add(x)
    
    return best
