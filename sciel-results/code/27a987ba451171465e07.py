import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit
    
    N = 284
    PRESCRIBED = [1, 2, 5, 11]
    PRESCRIBED_SET = set(PRESCRIBED)
    
    # Helper: Check if adding x to set S is valid
    # S should be a list, S_set a set, S_sums a set of all pairwise sums
    def is_valid_add(x, S, S_set, S_sums):
        if x in S_set:
            return False
        # Check 2*x
        if 2 * x in S_sums:
            return False
        # Check x + y for all y in S
        for y in S:
            if (x + y) in S_sums:
                return False
        return True

    # Helper: Compute sums for a set
    def get_sums(S):
        s = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                s.add(S[i] + S[j])
        return s

    # Helper: Full validation check
    def is_valid_set(S):
        if len(S) != len(set(S)):
            return False
        if not PRESCRIBED_SET.issubset(set(S)):
            return False
        for v in S:
            if v < 1 or v > N:
                return False
        sums = get_sums(S)
        return len(sums) == len(S) * (len(S) + 1) // 2

    # Phase 1: Randomized Greedy Restarts
    # Goal: Find a large valid set by trying many random orderings of candidates
    best_set = list(PRESCRIBED)
    best_size = len(PRESCRIBED)
    
    candidates = list(range(1, N + 1))
    
    # We will run greedy restarts until we have used up most of the time
    # or until we find a set of size 20 (known max)
    while time.time() < DEADLINE - 20.0:
        # Shuffle candidates for this restart
        random.shuffle(candidates)
        
        current_set = list(PRESCRIBED)
        current_set_set = set(current_set)
        current_sums = get_sums(current_set)
        
        for x in candidates:
            if x in current_set_set:
                continue
            if is_valid_add(x, current_set, current_set_set, current_sums):
                current_set.append(x)
                current_set_set.add(x)
                # Update sums
                for y in current_set[:-1]:
                    current_sums.add(x + y)
                current_sums.add(2 * x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        if best_size == 20:
            break
            
    # Phase 2: Local Search / Polish
    # Use remaining time to try to improve best_set or find even larger sets
    # We will try to grow best_set or find a different larger set
    # Strategy: Take best_set, try to add more elements. If stuck, swap elements.
    
    # If we didn't reach 20, try to polish
    if best_size < 20:
        current_set = list(best_set)
        current_set_set = set(current_set)
        current_sums = get_sums(current_set)
        
        # Try to add any remaining valid elements
        added = True
        while added:
            added = False
            for x in range(1, N + 1):
                if x in current_set_set:
                    continue
                if is_valid_add(x, current_set, current_set_set, current_sums):
                    current_set.append(x)
                    current_set_set.add(x)
                    for y in current_set[:-1]:
                        current_sums.add(x + y)
                    current_sums.add(2 * x)
                    added = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    if best_size == 20:
                        break
            if best_size == 20:
                break
        
        # If still stuck, try local search swaps to escape local optima
        # We allow swapping non-prescribed elements with random others
        # to potentially open up space for larger elements
        if time.time() < DEADLINE - 2.0:
            # Improved Local Search:
            # 1. Try to add any valid x.
            # 2. If fail, try "Swap 1 for 2": Remove one non-prescribed, add two new ones.
            # 3. If fail, try "Swap 1 for 1": Remove one non-prescribed, add one new one (to explore).
            
            non_prescribed = [x for x in current_set if x not in PRESCRIBED_SET]
            
            while time.time() < DEADLINE - 1.0:
                # 1. Try to add
                added = False
                for x in range(1, N + 1):
                    if x in current_set_set:
                        continue
                    if is_valid_add(x, current_set, current_set_set, current_sums):
                        current_set.append(x)
                        current_set_set.add(x)
                        for y in current_set[:-1]:
                            current_sums.add(x + y)
                        current_sums.add(2 * x)
                        added = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                        break
                
                if added:
                    non_prescribed = [x for x in current_set if x not in PRESCRIBED_SET]
                    continue
                
                if not non_prescribed:
                    break
                
                # 2. Try Swap 1 for 2 (Net +1 size)
                found_improvement = False
                candidates = [x for x in range(1, N + 1) if x not in current_set_set]
                
                for u in non_prescribed:
                    if time.time() > DEADLINE - 1.0:
                        break
                    
                    # S_temp = current_set without u
                    S_temp = [x for x in current_set if x != u]
                    S_temp_set = set(S_temp)
                    S_temp_sums = get_sums(S_temp)
                    
                    for v in candidates:
                        if time.time() > DEADLINE - 1.0:
                            break
                        
                        # Check if v can be added to S_temp
                        if is_valid_add(v, S_temp, S_temp_set, S_temp_sums):
                            S_temp2 = S_temp + [v]
                            S_temp2_set = S_temp_set | {v}
                            S_temp2_sums = S_temp_sums.copy()
                            for y in S_temp:
                                S_temp2_sums.add(v + y)
                            S_temp2_sums.add(2 * v)
                            
                            for w in candidates:
                                if w == v:
                                    continue
                                if time.time() > DEADLINE - 1.0:
                                    break
                                
                                # Check if w can be added to S_temp2
                                if is_valid_add(w, S_temp2, S_temp2_set, S_temp2_sums):
                                    # Valid improvement found
                                    current_set = S_temp2 + [w]
                                    current_set_set = S_temp2_set | {w}
                                    current_sums = S_temp2_sums.copy()
                                    for y in S_temp2:
                                        current_sums.add(w + y)
                                    current_sums.add(2 * w)
                                    
                                    if len(current_set) > best_size:
                                        best_size = len(current_set)
                                        best_set = list(current_set)
                                    
                                    found_improvement = True
                                    non_prescribed = [x for x in current_set if x not in PRESCRIBED_SET]
                                    break
                            if found_improvement:
                                break
                    if found_improvement:
                        break
                
                if found_improvement:
                    continue
                
                # 3. Try Swap 1 for 1 (Net 0 size, exploration)
                # Pick a random u to remove and try to find a valid v to add
                u = random.choice(non_prescribed)
                S_temp = [x for x in current_set if x != u]
                S_temp_set = set(S_temp)
                S_temp_sums = get_sums(S_temp)
                
                swap_done = False
                # Shuffle candidates for randomness
                random.shuffle(candidates)
                for v in candidates:
                    if is_valid_add(v, S_temp, S_temp_set, S_temp_sums):
                        current_set = S_temp + [v]
                        current_set_set = S_temp_set | {v}
                        current_sums = S_temp_sums.copy()
                        for y in S_temp:
                            current_sums.add(v + y)
                        current_sums.add(2 * v)
                        
                        non_prescribed = [x for x in current_set if x not in PRESCRIBED_SET]
                        swap_done = True
                        break
                
                if not swap_done:
                    # If no swap worked, we might be stuck.
                    # Try a different random u? Or break?
                    # Let's try a few more random u's before giving up on this round
                    if random.random() < 0.1:
                        break

    # Final Validation
    if not is_valid_set(best_set):
        # Fallback to a simple greedy if something went wrong
        current_set = list(PRESCRIBED)
        current_set_set = set(current_set)
        current_sums = get_sums(current_set)
        for x in range(1, N + 1):
            if x in current_set_set:
                continue
            if is_valid_add(x, current_set, current_set_set, current_sums):
                current_set.append(x)
                current_set_set.add(x)
                for y in current_set[:-1]:
                    current_sums.add(x + y)
                current_sums.add(2 * x)
        best_set = current_set

    return best_set
