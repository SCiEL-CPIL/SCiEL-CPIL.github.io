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
            # We do a simple hill-climbing / random walk on the set
            # Objective: Maximize size. 
            # Since we want max size, we primarily look for Add moves.
            # If no Add move works, we try Swap moves that might lead to an Add.
            
            # To be efficient, we'll just run a few iterations of:
            # 1. Try to add any valid x.
            # 2. If fail, remove a random non-prescribed element, try to add a random x.
            
            non_prescribed = [x for x in current_set if x not in PRESCRIBED_SET]
            
            while time.time() < DEADLINE - 1.0:
                # 1. Try to add
                added = False
                # Check all candidates quickly
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
                    continue
                
                # 2. Try to swap
                if not non_prescribed:
                    break
                
                # Pick a random non-prescribed element to remove
                idx_to_remove = random.randrange(len(non_prescribed))
                val_to_remove = non_prescribed[idx_to_remove]
                
                # Pick a random candidate to add
                candidates_to_try = [x for x in range(1, N + 1) if x not in current_set_set]
                if not candidates_to_try:
                    break
                
                val_to_add = random.choice(candidates_to_try)
                
                # Perform swap
                new_set = [x for x in current_set if x != val_to_remove]
                new_set.append(val_to_add)
                
                # Check validity of new_set
                # Since size is same, we just need to check if it's a valid Sidon set
                # Actually, we want to see if this swap allows future growth.
                # But simple heuristic: accept swap if new_set is valid (it will be if we just swapped and didn't introduce collision? No, swap can introduce collision).
                # Let's check if new_set is valid.
                new_set_set = set(new_set)
                new_sums = get_sums(new_set)
                
                if len(new_sums) == len(new_set) * (len(new_set) + 1) // 2:
                    # Valid set of same size. Accept with some probability to explore?
                    # Or accept if it looks "better"? 
                    # For now, accept to explore neighborhood.
                    current_set = new_set
                    current_set_set = new_set_set
                    current_sums = new_sums
                    non_prescribed = [x for x in current_set if x not in PRESCRIBED_SET]

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
