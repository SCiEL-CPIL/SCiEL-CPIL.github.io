import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    required = [1, 2, 5, 11]
    best_set = set(required)
    best_size = len(best_set)
    
    def get_sums(S):
        sums = set()
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        return sums

    def is_valid_addition(x, S, sums):
        # Check if x can be added to S
        # New sums will be x + y for y in S, and 2*x
        # Check if any of these are in current sums
        # And check if any of these new sums collide with each other (only 2*x can collide with x+y if x=y, but x not in S)
        # Actually, if x not in S, x+y are distinct for distinct y.
        # So we just need to check intersection with existing sums.
        for y in S:
            if (x + y) in sums:
                return False
        if (2 * x) in sums:
            return False
        # Also need to ensure x itself is not in S (handled by caller)
        return True

    def add_to_sums(x, S, sums):
        new_sums = set()
        for y in S:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        sums.update(new_sums)

    def remove_from_sums(x, S, sums):
        # This is tricky. We need to remove sums involving x.
        # It's safer to recompute sums from scratch for small sets.
        pass

    def random_greedy_with_swaps():
        S = set(required)
        sums = get_sums(S)
        
        # Phase 1: Greedy Add
        while True:
            candidates = []
            for x in range(1, 285):
                if x not in S:
                    if is_valid_addition(x, S, sums):
                        candidates.append(x)
            
            if not candidates:
                break
            
            # Pick a candidate
            # Heuristic: pick smallest? Or random?
            # Randomization helps escape local optima.
            # Let's pick randomly from the first 10 candidates to balance speed and diversity?
            # Or just random from all.
            x = random.choice(candidates)
            S.add(x)
            add_to_sums(x, S, sums)
            
        # Phase 2: Local Search (Swaps) to increase size
        # If we are stuck, we try to swap an element to allow an addition.
        # We want to reach size 20.
        # Current size is len(S).
        
        # We will try to find a swap that allows an immediate addition.
        # Or just a swap that maintains validity, hoping it leads to an addition later.
        
        # Better: Try to find a swap (a, b) such that S' = S - {a} + {b} is valid.
        # Then check if S' allows an addition.
        
        # This can be expensive.
        # Let's try a simpler loop:
        # While we can improve:
        #   Try to add.
        #   If fail, try a random valid swap.
        #   If no valid swap, break.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add
            candidates = []
            for x in range(1, 285):
                if x not in S:
                    if is_valid_addition(x, S, sums):
                        candidates.append(x)
            
            if candidates:
                x = random.choice(candidates)
                S.add(x)
                add_to_sums(x, S, sums)
                improved = True
                continue
            
            # 2. If no add, try a swap
            # Find a random a in S and a random b not in S
            # Check if S' is valid.
            # If valid, perform swap and recompute sums.
            # To make this effective, we should bias towards swaps that "open up" space.
            # But random is okay for now.
            
            a = random.choice(list(S))
            # Pick b from a random subset of non-S elements
            # To avoid picking b that is clearly bad (e.g., very close to a), we can just pick random.
            b = random.randint(1, 284)
            
            if b in S:
                continue
                
            # Check validity of S' = S - {a} + {b}
            # We need to check if b is compatible with S - {a}.
            S_temp = S - {a}
            sums_temp = get_sums(S_temp)
            
            if is_valid_addition(b, S_temp, sums_temp):
                # Perform swap
                S.remove(a)
                S.add(b)
                sums = get_sums(S) # Recompute for safety
                improved = True # We changed state, might allow future adds
                
        return S

    # Main Loop: Multiple Restarts
    while time.time() < deadline:
        candidate_set = random_greedy_with_swaps()
        if len(candidate_set) > best_size:
            best_size = len(candidate_set)
            best_set = candidate_set
            if best_size == 20:
                break

    return list(best_set)
