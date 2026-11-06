import time
import random

def construct():
    start_time = time.time()
    total_deadline = 110.0
    
    fixed = {1, 2, 5, 11}
    
    def get_sums(S):
        sums = set()
        s_list = sorted(S)
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                sums.add(s_list[i] + s_list[j])
        return sums

    best_set = list(fixed)
    
    # Phase 1: Randomized Greedy with Restarts
    # Try to find a large set quickly
    greedy_deadline = start_time + 5.0
    while time.time() < greedy_deadline:
        S = set(fixed)
        sums = get_sums(S)
        
        # Greedy fill
        while True:
            candidates = []
            # Optimization: Only check x > min(S) ? No, x can be anywhere.
            # But since we start with 1,2,5,11, and want to maximize, 
            # picking small numbers packs them.
            for x in range(1, 285):
                if x in S: continue
                # Check compatibility
                # new sums: x+y for y in S, and 2x
                # We need to ensure these are not in 'sums' and distinct from each other
                # Distinct from each other: x+y1 == x+y2 => y1==y2, so distinct if y distinct.
                # 2x == x+y => x==y, but x not in S. So distinct.
                # So we just check intersection with 'sums'
                is_ok = True
                for y in S:
                    if x + y in sums:
                        is_ok = False
                        break
                if is_ok and 2*x in sums:
                    is_ok = False
                if is_ok:
                    candidates.append(x)
            
            if not candidates:
                break
            
            # Heuristic: Pick smallest to pack tight? Or random?
            # Random helps diversity. Smallest helps packing.
            # Let's mix: 50% smallest, 50% random
            if random.random() < 0.5:
                x = min(candidates)
            else:
                x = random.choice(candidates)
                
            S.add(x)
            for y in S:
                if y != x:
                    sums.add(x + y)
            sums.add(2 * x)
            
        if len(S) > len(best_set):
            best_set = list(S)
            if len(best_set) == 20:
                return sorted(best_set)

    # Phase 2: Backtracking Search
    # Try to find size 20, then 19, etc.
    # We only need to go down to len(best_set)
    for target_size in range(20, len(best_set) - 1, -1):
        if time.time() > total_deadline - 5:
            break
            
        found_set = backtracking_search(fixed, target_size, total_deadline - 5)
        if found_set:
            return sorted(found_set)

    return sorted(best_set)

def backtracking_search(fixed, target_size, deadline):
    S = list(fixed)
    sums = set()
    s_sorted = sorted(S)
    for i in range(len(s_sorted)):
        for j in range(i, len(s_sorted)):
            sums.add(s_sorted[i] + s_sorted[j])
            
    def backtrack(last_x, current_S, current_sums):
        if len(current_S) == target_size:
            return list(current_S)
        
        if time.time() > deadline:
            return None
            
        # Heuristic: Try smallest available x first
        # Start from last_x + 1 to avoid permutations (order doesn't matter for set)
        # But we must ensure we don't miss smaller x if we skipped them?
        # No, we are building the set in increasing order.
        # If we pick x, all future picks must be > x.
        # This covers all sets uniquely.
        
        for x in range(last_x + 1, 285):
            if x in current_S: continue # Should not happen if last_x logic is correct
            
            # Check compatibility
            is_ok = True
            for y in current_S:
                if x + y in current_sums:
                    is_ok = False
                    break
            if is_ok and 2 * x in current_sums:
                is_ok = False
                
            if is_ok:
                # Add x
                new_S = current_S + [x]
                new_sums = set(current_sums)
                for y in current_S:
                    new_sums.add(x + y)
                new_sums.add(2 * x)
                
                res = backtrack(x, new_S, new_sums)
                if res:
                    return res
                
        return None

    return backtrack(0, S, sums)
