import time
import random
from itertools import combinations

def _can_add(x, selected):
    """
    Check if x can be added to selected without violating constraints.
    """
    n = len(selected)
    if n == 0:
        return True
        
    # Check 1: x as divisor
    # Sum of any 3 distinct from selected
    # Optimize: Pre-compute pair sums?
    # For now, keep it simple but efficient.
    for i in range(n):
        a = selected[i]
        for j in range(i+1, n):
            b = selected[j]
            ab = a + b
            for k in range(j+1, n):
                if (ab + selected[k]) % x == 0:
                    return False

    # Check 2: existing elements a as divisors
    # For each a in selected, check if a divides (u + v + x)
    # where u, v are distinct elements in selected \ {a}
    for idx_a, a in enumerate(selected):
        # We can pre-compute sums of pairs from selected \ {a}?
        # Or just iterate.
        # To optimize, we can skip if a is very small and we know it divides many things?
        # No, just check.
        for i in range(n):
            if i == idx_a:
                continue
            u = selected[i]
            for j in range(i+1, n):
                if j == idx_a:
                    continue
                v = selected[j]
                if (u + v + x) % a == 0:
                    return False
                    
    return True

def _is_valid(selected):
    """Full validity check for debugging or final verification."""
    n = len(selected)
    for idx, a in enumerate(selected):
        others = [selected[i] for i in range(n) if i != idx]
        for combo in combinations(others, 3):
            if sum(combo) % a == 0:
                return False
    return True

def _greedy(order, deadline):
    """Greedy construction based on given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """
    Improve the set by trying to add elements or swap elements.
    Uses Variable Neighborhood Search (VNS) style swaps.
    """
    best = list(selected)
    best_size = len(best)
    best_set = set(best)
    all_nums = set(range(1, 193))
    
    # Max rounds to prevent infinite loops
    max_rounds = 500
    no_improve_count = 0
    
    for r in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        
        # 1. Try adding a single element (k=0 swap)
        missing = list(all_nums - best_set)
        random.shuffle(missing)
        
        for x in missing:
            if time.time() > deadline:
                break
            if _can_add(x, best):
                best.append(x)
                best_set.add(x)
                best_size = len(best)
                improved = True
                no_improve_count = 0
                # Continue trying to add more in this round?
                # Yes, greedy add.
                # But we should break out to re-shuffle?
                # Let's just continue.
        
        if improved:
            continue # Restart round with new state
        
        # 2. Try swaps: Remove k, Add k+1
        # Start with k=1
        found_swap = False
        for k in range(1, 4): # Try removing 1, 2, or 3 elements
            if time.time() > deadline:
                break
            
            # Generate combinations of k elements to remove
            # To keep it fast, limit the number of combinations checked
            # If k=1, 48 combos.
            # If k=2, 48*47/2 = 1128 combos.
            # If k=3, 48*47*46/6 = 17296 combos.
            
            if k == 1:
                removals = [ [i] for i in range(len(best)) ]
            elif k == 2:
                removals = [ list(c) for c in combinations(range(len(best)), 2) ]
            else:
                # Random sample if too many
                removals = [ list(c) for c in combinations(range(len(best)), 3) ]
                if len(removals) > 100:
                    random.shuffle(removals)
                    removals = removals[:100]
            
            random.shuffle(removals)
            
            for rem_indices in removals:
                if time.time() > deadline:
                    break
                
                # Create temp set
                rem_set = {best[i] for i in rem_indices}
                temp = [x for i, x in enumerate(best) if i not in rem_set]
                temp_set = best_set - rem_set
                
                # Try to add k+1 elements from missing
                missing = list(all_nums - temp_set)
                random.shuffle(missing)
                
                added = 0
                # We need to add k+1 elements.
                # Greedily try to add them.
                # We need to check validity incrementally.
                
                # To be efficient, we can just try to add one by one.
                # But we need to ensure we can add k+1.
                
                candidate_adds = []
                for x in missing:
                    if added >= k + 1:
                        break
                    if _can_add(x, temp):
                        temp.append(x)
                        temp_set.add(x)
                        added += 1
                
                if added == k + 1:
                    # We successfully swapped k elements for k+1 elements
                    best = temp
                    best_set = temp_set
                    best_size = len(best)
                    improved = True
                    found_swap = True
                    no_improve_count = 0
                    break # Break out of removals loop
            
            if found_swap:
                break # Break out of k loop
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 10:
                break # Stuck
        
    return best

def construct():
    start_time = time.time()
    # 60s limit. Use 55s deadline to be safe.
    deadline = start_time + 55.0
    
    # 1. Generate initial candidates using different greedy orders
    candidates = []
    
    # Descending (Large numbers first) - Usually best for this problem
    order_desc = list(range(192, 0, -1))
    s1 = _greedy(order_desc, deadline)
    candidates.append(s1)
    
    # Ascending (Small numbers first)
    if time.time() < deadline:
        order_asc = list(range(1, 193))
        s2 = _greedy(order_asc, deadline)
        candidates.append(s2)
        
    # Randomized Greedy
    for seed in range(20):
        if time.time() > deadline - 10.0: # Leave 10s for local search
            break
        rng = random.Random(seed)
        order_rand = list(range(1, 193))
        rng.shuffle(order_rand)
        s = _greedy(order_rand, deadline)
        candidates.append(s)
        
    # Pick the best candidate
    best = max(candidates, key=len)
    
    # 2. Local Search
    if time.time() < deadline - 1.0:
        best = _local_search(best, deadline)
        
    # 3. Final Validation
    if not _is_valid(best):
        # Fallback to the best greedy solution that was valid
        best = s1
        
    return best
