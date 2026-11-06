import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 57  # 57s safety margin for 60s limit
    
    N = 256
    
    # State:
    # in_set: boolean list
    # conflict_count: list of ints, count[x] = number of pairs (a,b) in S such that {a,b,x} is a 3-AP
    # S: set of elements
    
    in_set = [False] * (N + 1)
    conflict_count = [0] * (N + 1)
    S = set()
    
    def add(x):
        """Add x to S, update conflict counts."""
        if in_set[x]:
            return
        in_set[x] = True
        S.add(x)
        # Update conflict counts for all z in S
        # For each z in S, the pair {x, z} creates new conflicts for potential third elements w.
        # We iterate over current S (before adding x? No, we added x to S already in code above? 
        # Better to iterate over list(S) before adding x to S to avoid confusion, or just use in_set check).
        # Let's use in_set to iterate.
        # Actually, we need to update counts for w based on pairs {x, z} where z is in S_OLD.
        # So we should iterate over z where in_set[z] is True BEFORE we set in_set[x]=True.
        # Let's restructure:
        
    # Let's rewrite add/remove properly.
    
    def _add_element(x):
        if in_set[x]:
            return
        # Calculate updates based on current S
        # Pairs are {x, z} for z in S
        # For each z in S:
        #   a = min(x,z), b = max(x,z)
        #   w1 = 2b - a
        #   w2 = 2a - b
        #   if (a+b) % 2 == 0: w3 = (a+b)//2
        #   For each valid w in [1,N], increment conflict_count[w]
        
        # We need to iterate over z in S.
        # To make it fast, we can maintain a list of elements? 
        # S is small (~100). Iterating over S is fine.
        
        for z in list(S):
            a, b = (x, z) if x < z else (z, x)
            # w1: a, b, w1 => w1 = 2b - a
            w1 = 2 * b - a
            if 1 <= w1 <= N:
                conflict_count[w1] += 1
                
            # w2: w2, a, b => w2 = 2a - b
            w2 = 2 * a - b
            if 1 <= w2 <= N:
                conflict_count[w2] += 1
                
            # w3: a, w3, b => w3 = (a+b)/2
            if (a + b) % 2 == 0:
                w3 = (a + b) // 2
                if 1 <= w3 <= N:
                    conflict_count[w3] += 1
        
        # Now update state
        in_set[x] = True
        S.add(x)

    def _remove_element(x):
        if not in_set[x]:
            return
        # Calculate decrements based on current S (including x)
        # Pairs are {x, z} for z in S, z != x
        for z in list(S):
            if z == x:
                continue
            a, b = (x, z) if x < z else (z, x)
            
            w1 = 2 * b - a
            if 1 <= w1 <= N:
                conflict_count[w1] -= 1
                
            w2 = 2 * a - b
            if 1 <= w2 <= N:
                conflict_count[w2] -= 1
                
            if (a + b) % 2 == 0:
                w3 = (a + b) // 2
                if 1 <= w3 <= N:
                    conflict_count[w3] -= 1
        
        # Update state
        in_set[x] = False
        S.remove(x)

    # Greedy Initialization
    # Sort by value? Or just 1..N?
    # To maximize sum of reciprocals, we prefer small numbers.
    # Greedy: try to add 1, 2, 3...
    for x in range(1, N + 1):
        if conflict_count[x] == 0:
            _add_element(x)
            
    best_set = set(S)
    best_in_set = in_set[:]
    best_conflict_count = conflict_count[:]
    best_score = sum(1.0 / x for x in best_set)
    
    # Local Search
    # Strategy: Remove one element, then greedily fill with smallest possible valid elements.
    # This tends to pull the set towards smaller numbers.
    
    improved = True
    round_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        round_count += 1
        
        # To avoid getting stuck, we can randomize the order of removals
        # But deterministic is safer for reproducibility? 
        # The task doesn't require deterministic, but "score is floating serialization".
        # Let's stick to deterministic first, then maybe add randomness if stuck.
        
        # We will iterate through all x in best_set
        # For each x:
        # 1. Save state
        # 2. Remove x
        # 3. Greedily add y=1..N if conflict_count[y]==0
        # 4. If new score > best score, update best state
        # 5. Restore state to best state (or keep if improved)
        
        # Optimization: 
        # Instead of full save/restore, we can just try moves and revert if bad.
        # But "greedily fill" adds many elements. Reverting is expensive.
        # So we must save state.
        # State size: in_set (257 bools), conflict_count (257 ints), S (set of ~100).
        # Copying is cheap.
        
        for x in list(best_set):
            if time.time() > deadline:
                break
            
            # Save state
            in_set_temp = best_in_set[:]
            conflict_count_temp = best_conflict_count[:]
            S_temp = set(best_set)
            
            # Remove x
            in_set_temp[x] = False
            S_temp.remove(x)
            
            # We need to update conflict counts for removal.
            # We can reuse the logic from _remove_element but on the temp arrays.
            # Let's write a helper or inline it.
            
            # Inline removal logic on temp arrays
            for z in list(S_temp):
                if z == x:
                    continue
                a, b = (x, z) if x < z else (z, x)
                
                w1 = 2 * b - a
                if 1 <= w1 <= N:
                    conflict_count_temp[w1] -= 1
                    
                w2 = 2 * a - b
                if 1 <= w2 <= N:
                    conflict_count_temp[w2] -= 1
                    
                if (a + b) % 2 == 0:
                    w3 = (a + b) // 2
                    if 1 <= w3 <= N:
                        conflict_count_temp[w3] -= 1
            
            # Greedily fill
            # We want to add elements with smallest index to maximize 1/y.
            # Check y from 1 to N.
            for y in range(1, N + 1):
                if in_set_temp[y]:
                    continue
                if conflict_count_temp[y] == 0:
                    # Add y to temp state
                    # Update conflict counts
                    for z in list(S_temp):
                        a, b = (y, z) if y < z else (z, y)
                        
                        w1 = 2 * b - a
                        if 1 <= w1 <= N:
                            conflict_count_temp[w1] += 1
                            
                        w2 = 2 * a - b
                        if 1 <= w2 <= N:
                            conflict_count_temp[w2] += 1
                            
                        if (a + b) % 2 == 0:
                            w3 = (a + b) // 2
                            if 1 <= w3 <= N:
                                conflict_count_temp[w3] += 1
                            
                    in_set_temp[y] = True
                    S_temp.add(y)
            
            new_score = sum(1.0 / y for y in S_temp)
            
            if new_score > best_score + 1e-12:
                best_set = S_temp
                best_in_set = in_set_temp
                best_conflict_count = conflict_count_temp
                best_score = new_score
                improved = True
                # If we improved, we might want to restart the loop over x?
                # Or just continue with the next x in the *old* list?
                # Continuing with old list is fine, but the set changed.
                # It's safer to break and restart the outer loop to re-evaluate all x in the new set.
                # However, that might be slow.
                # Let's just continue. The next x might not be in the new set?
                # If x was removed and not re-added, it's not in S_temp.
                # But we are iterating over `list(best_set)` which was captured *before* the modification.
                # So we might try to remove an element that is no longer in the best set?
                # No, `best_set` is updated to `S_temp`.
                # But the loop variable `x` comes from the old list.
                # If `x` is not in the new `best_set`, we should skip?
                # Actually, if we improved, the state is different.
                # It is better to break and restart the outer loop to ensure we explore the new neighborhood properly.
                break
        
    return list(best_set)
