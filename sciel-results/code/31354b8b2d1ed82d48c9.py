<<<<<<< SEARCH
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute popcount? Not strictly necessary for comparability check, but good for debugging.
    # We need to select 'm' distinct masks.
    
    # Initial family: Empty set and Full set are always good anchors.
    # Empty set (0) is subset of all.
    # Full set ((1<<n)-1) is superset of all.
    # They are comparable to each other.
    
    # Strategy:
    # 1. Start with a few strong candidates.
    # 2. Greedily add the next best candidate.
    
    # To improve results, we can try a few different initial seeds or random restarts if time permits.
    # But greedy from {0, full} is a strong baseline.
    
    # Let's try a slightly more robust greedy:
    # Instead of just adding the best, maybe we should consider that adding a set of size k 
    # makes it comparable to existing sets of size < k and > k.
    
    # We'll run a single greedy pass. It's fast.
    
    # List of all masks
    # We don't need to store all masks in a list if we can iterate 0..4095.
    
    selected = []
    # Start with 0 and full
    selected.append(0)
    selected.append(total_masks - 1)
    
    # We need m - 2 more sets
    # To make it deterministic and robust, we iterate candidates.
    
    # Optimization: 
    # For each candidate c, score = sum(1 for s in selected if comparable(c, s))
    # comparable(c, s) is (c & s) == c or (c & s) == s
    
    # We can speed this up by noting that if we have many sets of a certain size,
    # we should pick sets that are subsets of large sets and supersets of small sets.
    
    # Let's just do the greedy.
    # 38 iterations.
    # Each iteration checks ~4096 candidates.
    # Each check involves ~38 comparisons (bit ops).
    # 38 * 4096 * 38 ~ 5.8 million ops. Very fast.
    
    for _ in range(m - len(selected)):
        if time.time() > deadline:
            break
            
        best_score = -1
        best_mask = -1
        
        # Iterate all possible masks
        # We can skip masks already in selected. 
        # Since selected is small, checking `if mask in selected` is O(k).
        # Alternatively, maintain a set of selected masks for O(1) lookup.
        
        selected_set = set(selected)
        
        for mask in range(total_masks):
            if mask in selected_set:
                continue
                
            score = 0
            # Check comparability with all selected
            for s in selected:
                # Strict subset check
                # c subset s: (c & s) == c and c != s
                # s subset c: (s & c) == s and s != c
                # Since mask != s is guaranteed by the set check, we just check intersection.
                if (mask & s) == mask or (mask & s) == s:
                    score += 1
                    
            if score > best_score:
                best_score = score
                best_mask = mask
                
        if best_mask != -1:
            selected.append(best_mask)
        else:
            break

    # Convert to lists of integers
    # Requirement: "each as a list of distinct integers"
    # "integers must be Python int"
    
    result = []
    for mask in selected:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i)
        result.append(subset)
        
    # Ensure exactly 40 lists. 
    # If the loop broke early due to time, we might have fewer.
    # The problem says "exactly forty". 
    # If we run out of time, we should fill with valid distinct sets to maintain validity, 
    # even if score is lower, rather than returning invalid count.
    # But 5.8M ops is < 0.1s in Python. Time limit is not a risk for this specific algorithm.
    
    # Just in case, if len(result) < 40, add random unused masks.
    if len(result) < 40:
        used_masks = set(selected)
        # Shuffle all masks and pick unused ones
        all_masks = list(range(total_masks))
        random.shuffle(all_masks)
        for mask in all_masks:
            if len(result) >= 40:
                break
            if mask not in used_masks:
                used_masks.add(mask)
                subset = [i for i in range(n) if mask & (1 << i)]
                result.append(subset)
                
    return result
=======
import time
import random
import math

def _count_pairs(selected):
    """Count the number of comparable unordered pairs in the family."""
    count = 0
    m = len(selected)
    for i in range(m):
        for j in range(i + 1, m):
            a, b = selected[i], selected[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _incremental_score(mask, selected):
    """Compute the score of adding 'mask' to 'selected' (number of comparable pairs with existing members)."""
    score = 0
    for s in selected:
        if (mask & s) == mask or (mask & s) == s:
            score += 1
    return score

def _greedy_construct(n, m, deadline):
    """Greedy construction starting from empty and full sets."""
    total_masks = 1 << n
    selected = [0, total_masks - 1]
    selected_set = set(selected)
    
    for _ in range(m - len(selected)):
        if time.time() > deadline:
            break
            
        best_score = -1
        best_mask = -1
        
        for mask in range(total_masks):
            if mask in selected_set:
                continue
                
            score = _incremental_score(mask, selected)
            
            if score > best_score:
                best_score = score
                best_mask = mask
                
        if best_mask != -1:
            selected.append(best_mask)
            selected_set.add(best_mask)
        else:
            break
    
    return selected

def _local_search(selected, n, deadline, max_iters=200):
    """Local search: try swapping one selected set with an unselected set if it improves the score."""
    total_masks = 1 << n
    m = len(selected)
    selected_set = set(selected)
    current_score = _count_pairs(selected)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        
        # Try removing each selected set and adding the best replacement
        for idx in range(m):
            removed = selected[idx]
            remaining = selected[:idx] + selected[idx+1:]
            remaining_set = set(remaining)
            
            # Find the best mask to add that isn't in remaining
            best_add_score = -1
            best_add_mask = -1
            
            for mask in range(total_masks):
                if mask in remaining_set:
                    continue
                # Score = comparability with remaining + (1 if comparable with removed, else 0)
                # Actually, we want to maximize total score after swap
                # New score = old_score - (comparability of removed with others) + (comparability of new with others)
                # But easier: just compute incremental score of new mask with remaining
                score = _incremental_score(mask, remaining)
                
                if score > best_add_score:
                    best_add_score = score
                    best_add_mask = mask
            
            if best_add_mask == -1:
                continue
            
            # Compute new total score
            new_score = current_score
            # Subtract comparability of removed with all other selected
            for j in range(m):
                if j == idx:
                    continue
                if (removed & selected[j]) == removed or (removed & selected[j]) == selected[j]:
                    new_score -= 1
            # Add comparability of new mask with all remaining
            new_score += best_add_score
            
            if new_score > current_score:
                # Perform the swap
                selected[idx] = best_add_mask
                selected_set.remove(removed)
                selected_set.add(best_add_mask)
                current_score = new_score
                improved = True
                break  # Restart the loop with the new configuration
        
        if not improved:
            break
    
    return selected, current_score

def _simulated_annealing(selected, n, deadline, initial_temp=10.0, cooling=0.99, max_iters=1000):
    """Simulated annealing to escape local optima."""
    total_masks = 1 << n
    m = len(selected)
    selected_set = set(selected)
    current_score = _count_pairs(selected)
    best_score = current_score
    best_selected = list(selected)
    
    temp = initial_temp
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Pick a random index to swap
        idx = random.randint(0, m - 1)
        removed = selected[idx]
        
        # Pick a random mask not in selected
        new_mask = random.randint(0, total_masks - 1)
        while new_mask in selected_set:
            new_mask = random.randint(0, total_masks - 1)
        
        # Compute score change
        delta = 0
        # Subtract comparability of removed with all other selected
        for j in range(m):
            if j == idx:
                continue
            if (removed & selected[j]) == removed or (removed & selected[j]) == selected[j]:
                delta -= 1
        # Add comparability of new mask with all other selected
        for j in range(m):
            if j == idx:
                continue
            if (new_mask & selected[j]) == new_mask or (new_mask & selected[j]) == selected[j]:
                delta += 1
        
        new_score = current_score + delta
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if delta > 0 or random.random() < math.exp(-delta / temp):
            # Perform the swap
            selected[idx] = new_mask
            selected_set.remove(removed)
            selected_set.add(new_mask)
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_selected = list(selected)
        
        temp *= cooling
    
    return best_selected, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = 12
    m = 40
    
    # Phase 1: Greedy construction
    selected = _greedy_construct(n, m, deadline)
    
    # Phase 2: Local search
    selected, score = _local_search(selected, n, deadline)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5:
        selected, score = _simulated_annealing(selected, n, deadline)
    
    # Convert to lists of integers
    result = []
    for mask in selected:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i)
        result.append(subset)
    
    # Ensure exactly 40 lists
    if len(result) < 40:
        used_masks = set(selected)
        all_masks = list(range(1 << n))
        random.shuffle(all_masks)
        for mask in all_masks:
            if len(result) >= 40:
                break
            if mask not in used_masks:
                used_masks.add(mask)
                subset = [i for i in range(n) if mask & (1 << i)]
                result.append(subset)
    
    return result
>>>>>>> REPLACE
