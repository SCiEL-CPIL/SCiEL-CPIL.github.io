import math
import time
import sys

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    n = 256
    
    # 1. Identify fixed elements: 1 and primes > n/2
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    large_primes = [p for p in primes if p > n // 2]
    
    fixed = {1}
    fixed.update(large_primes)
    
    candidates = [i for i in range(1, n + 1) if i not in fixed]
    
    # Precompute GCDs for candidates
    # Map candidate index to value
    cand_val = {i: i for i in candidates}
    num_cands = len(candidates)
    
    # Conflict matrix for candidates
    # conflict[i][j] is True if gcd(candidates[i], candidates[j]) > 1
    # We can use a bitset or list of sets for efficiency
    conflicts = [set() for _ in range(num_cands)]
    for i in range(num_cands):
        vi = candidates[i]
        for j in range(i + 1, num_cands):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Initial Solution: Greedy
    # Sort candidates by value descending
    sorted_cands = sorted(range(num_cands), key=lambda i: candidates[i], reverse=True)
    
    selected = [False] * num_cands
    current_sum = 0
    
    # Greedy Add
    for i in sorted_cands:
        # Check if i conflicts with any already selected
        is_conflict = False
        for j in conflicts[i]:
            if selected[j]:
                is_conflict = True
                break
        if not is_conflict:
            selected[i] = True
            current_sum += candidates[i]
            
    # Local Search
    # We will try to improve by swapping.
    # Move 1: Add any unselected node that is now free (if we removed something, but we don't remove in simple add)
    # Actually, the greedy add is maximal independent set. We can't add more without removing.
    
    # Move 2: Swap one selected for one unselected
    # Move 3: Swap two selected for two unselected (or 1 for 2, 2 for 1, etc.)
    
    # To make it fast, we iterate over potential swaps.
    # Since N is small, we can afford O(N^2) or O(N^3) per iteration if iterations are few.
    
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        best_gain = 0
        best_move = None
        
        # Try 1-for-1 swaps
        for i in range(num_cands):
            if not selected[i]:
                continue # i is not selected, we want to remove selected and add unselected
                # Wait, logic: Remove j (selected), Add i (unselected)
                
        # Corrected Loop for 1-for-1:
        # For each selected j, for each unselected i, check if i is compatible with S \ {j}
        
        # Optimization: Precompute neighbors.
        
        for j in range(num_cands):
            if not selected[j]:
                continue
            val_j = candidates[j]
            # Potential gain if we remove j is -val_j.
            # We want to add i such that i is not in S, and i is coprime to all k in S \ {j}.
            # i must not conflict with any k in S \ {j}.
            # This means conflicts[i] should not intersect (S \ {j}).
            # i.e., conflicts[i] & (selected_set - {j}) is empty.
            
            # Efficient check:
            # For each unselected i:
            # If i is compatible with S\{j}, gain = val_i - val_j.
            
            # To speed up, we can iterate unselected i first?
            # Let's stick to nested loops but break early if possible.
            
            for i in range(num_cands):
                if selected[i]:
                    continue
                val_i = candidates[i]
                if val_i <= val_j:
                    continue # No gain in 1-for-1 if val_i <= val_j? 
                    # Actually, if val_i == val_j, no gain. If val_i < val_j, negative gain.
                    # But maybe swapping allows a better 2-for-2 later? 
                    # For 1-for-1, we only accept if val_i > val_j.
                    
                # Check compatibility
                # i must be coprime to all selected k != j
                compatible = True
                for k in conflicts[i]:
                    if selected[k] and k != j:
                        compatible = False
                        break
                if compatible:
                    gain = val_i - val_j
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('1for1', i, j)
                        
        if best_move:
            typ, i, j = best_move
            selected[j] = False
            selected[i] = True
            current_sum += best_gain
            improved = True
            continue
            
        # Try 2-for-2 swaps
        # Remove j1, j2 (selected). Add i1, i2 (unselected).
        # Conditions:
        # 1. i1, i2 not in S
        # 2. i1 coprime to i2
        # 3. i1 coprime to S \ {j1, j2}
        # 4. i2 coprime to S \ {j1, j2}
        # Gain = (val_i1 + val_i2) - (val_j1 + val_j2)
        
        # This is O(N^4) if naive. Too slow?
        # N=232. N^4 is ~ 2.9 * 10^9. Too slow for Python in 50s.
        
        # Optimization:
        # Only consider j1, j2 that are "small" or "blocking" many options?
        # Or limit the search.
        
        # Alternative: 1-for-2 swaps.
        # Remove j (selected). Add i1, i2 (unselected).
        # Gain = val_i1 + val_i2 - val_j.
        # Conditions: i1, i2 coprime to each other, and to S\{j}.
        
        # Let's try 1-for-2 and 2-for-1 with limited search space.
        # We can restrict i1, i2 to the top K largest unselected values.
        
        top_unselected = sorted([i for i in range(num_cands) if not selected[i]], 
                                 key=lambda i: candidates[i], reverse=True)[:20]
        
        for j in range(num_cands):
            if not selected[j]:
                continue
            val_j = candidates[j]
            
            # 1-for-2
            for idx1 in range(len(top_unselected)):
                i1 = top_unselected[idx1]
                val_i1 = candidates[i1]
                # Check i1 compatible with S\{j}
                comp1 = True
                for k in conflicts[i1]:
                    if selected[k] and k != j:
                        comp1 = False
                        break
                if not comp1:
                    continue
                
                for idx2 in range(idx1 + 1, len(top_unselected)):
                    i2 = top_unselected[idx2]
                    val_i2 = candidates[i2]
                    
                    # Check i1 and i2 coprime
                    if math.gcd(candidates[i1], candidates[i2]) > 1:
                        continue
                    
                    # Check i2 compatible with S\{j}
                    comp2 = True
                    for k in conflicts[i2]:
                        if selected[k] and k != j:
                            comp2 = False
                            break
                    if not comp2:
                        continue
                        
                    gain = val_i1 + val_i2 - val_j
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('1for2', i1, i2, j)
                        
        if best_move:
            typ = best_move[0]
            if typ == '1for2':
                _, i1, i2, j = best_move
                selected[j] = False
                selected[i1] = True
                selected[i2] = True
            current_sum += best_gain
            improved = True
            continue
            
        # Try 2-for-1
        # Remove j1, j2. Add i1.
        # Gain = val_i1 - (val_j1 + val_j2)
        
        top_unselected_1 = top_unselected[:5] # Only very large ones
        
        # This is getting complex. The 1-for-1 and 1-for-2 might be enough to reach 1.2.
        
        # If no improvement in 1-for-1 or 1-for-2, break?
        # Maybe try 2-for-2 with very limited candidates?
        
        # For now, break if no 1-for-1 or 1-for-2 improvement.
        if best_gain == 0:
            break

    # Construct result
    result = list(fixed)
    for i in range(num_cands):
        if selected[i]:
            result.append(candidates[i])
            
    # Verify
    # Sort for deterministic output? Not required, but good.
    result = [int(x) for x in result]
    return result
