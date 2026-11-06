import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin
    
    prescribed = {1, 2, 5, 11}
    
    def get_initial_sums(s):
        sums = set()
        s_list = sorted(s)
        for i, a in enumerate(s_list):
            for b in s_list[i:]:
                sums.add(a + b)
        return sums

    def can_add(s_set, sums, x):
        # Check if x can be added to s_set with current sums
        # x must not be in s_set
        if x in s_set:
            return False
        # Check sums x+y for y in s_set
        for y in s_set:
            if x + y in sums:
                return False
        # Check 2*x
        if 2 * x in sums:
            return False
        return True

    def update_sums_add(sums, s_set, x):
        for y in s_set:
            sums.add(x + y)
        sums.add(2 * x)

    def update_sums_remove(sums, s_set, x):
        for y in s_set:
            sums.discard(x + y)
        sums.discard(2 * x)

    # 1. Generate a good initial solution using greedy with multiple strategies
    best_set = set()
    best_size = 0
    
    # Strategy A: Standard Greedy
    s = set(prescribed)
    sums = get_initial_sums(s)
    for x in range(12, 285):
        if can_add(s, sums, x):
            s.add(x)
            update_sums_add(sums, s, x)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s.copy()
        
    # Strategy B: Randomized Greedy (multiple restarts)
    random.seed(42)
    for _ in range(20):
        if time.time() > DEADLINE:
            break
        s = set(prescribed)
        sums = get_initial_sums(s)
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(s, sums, x):
                s.add(x)
                update_sums_add(sums, s, x)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s.copy()
            
    # Strategy C: Greedy with "Least Conflicts" heuristic
    for _ in range(10):
        if time.time() > DEADLINE:
            break
        s = set(prescribed)
        sums = get_initial_sums(s)
        remaining = [x for x in range(12, 285)]
        
        while remaining:
            if time.time() > DEADLINE:
                break
            
            # Score candidates by number of conflicts
            # Conflict: x+y in sums or 2x in sums
            # We want to pick x with 0 conflicts if possible.
            # If multiple, pick randomly among them? Or pick the one with min "future" conflicts?
            # Simple: Pick random from those with 0 conflicts.
            valid_candidates = []
            for x in remaining:
                if can_add(s, sums, x):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
                
            # To break ties or add diversity, we can just pick one
            # But to improve, maybe pick the one that creates sums that are "less likely" to conflict?
            # Hard to estimate. Let's just pick randomly from valid.
            x = random.choice(valid_candidates)
            s.add(x)
            update_sums_add(sums, s, x)
            remaining.remove(x)
            
        if len(s) > best_size:
            best_size = len(s)
            best_set = s.copy()

    # 2. Simulated Annealing to improve best_set
    # State: current set, current sums
    cur_set = best_set.copy()
    cur_sums = get_initial_sums(cur_set)
    cur_size = len(cur_set)
    
    best_sa_set = best_set.copy()
    best_sa_size = best_size
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.9995  # Adjust based on expected iterations
    
    # We need to estimate number of iterations.
    # Let's run until deadline.
    
    removable_elements = [x for x in cur_set if x not in prescribed]
    
    while time.time() < DEADLINE:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to ADD
            # Pick a random candidate from 1..284 not in cur_set
            # To speed up, we can maintain a list of "available" or just pick random
            # Picking random from 1..284 is fast enough (284 options)
            x = random.randint(1, 284)
            if x in cur_set:
                continue
            if can_add(cur_set, cur_sums, x):
                # Valid add
                cur_set.add(x)
                update_sums_add(cur_sums, cur_set, x)
                cur_size += 1
                # Always accept improvement
                if cur_size > best_sa_size:
                    best_sa_size = cur_size
                    best_sa_set = cur_set.copy()
            # If invalid, do nothing (stay in state)
            
        elif move_type < 0.7:
            # Try to REMOVE (to escape local minima)
            if not removable_elements:
                continue
            # Pick a random removable element
            idx = random.randint(0, len(removable_elements) - 1)
            x = removable_elements[idx]
            
            # Remove it
            cur_set.remove(x)
            update_sums_remove(cur_sums, cur_set, x)
            cur_size -= 1
            removable_elements.pop(idx)
            
            # Acceptance:
            # Since size decreased, we accept with probability exp(-1/T)
            # But we want to be able to recover.
            # In SA, we accept bad moves with prob exp(deltaE/T).
            # Here deltaSize = -1.
            # Prob = exp(-1/T).
            # If we reject, we must UNDO the move.
            if random.random() > math.exp(-1.0 / T):
                # Reject: Undo
                cur_set.add(x)
                update_sums_add(cur_sums, cur_set, x)
                cur_size += 1
                removable_elements.append(x)
            else:
                # Accept: Keep removed state
                pass

        else:
            # Try to SWAP
            # Remove x, Add y
            if not removable_elements:
                continue
            idx = random.randint(0, len(removable_elements) - 1)
            x = removable_elements[idx]
            
            # Try to find a y that can be added after removing x
            # We can pick a random y and check.
            # To be more efficient, we can pick a few random y's.
            swapped = False
            for _ in range(5): # Try 5 random swaps
                y = random.randint(1, 284)
                if y in cur_set or y == x:
                    continue
                
                # Check if y can be added to (cur_set \ {x})
                # We need to simulate removal of x first to check sums?
                # Or check:
                # 1. y not in cur_set
                # 2. y+x not in cur_sums? No, x is being removed.
                # Let's just do the full check:
                # Temporarily remove x
                cur_set.remove(x)
                update_sums_remove(cur_sums, cur_set, x)
                
                if can_add(cur_set, cur_sums, y):
                    # Valid swap
                    cur_set.add(y)
                    update_sums_add(cur_sums, cur_set, y)
                    # x is removed, y is added. Size unchanged.
                    # Accept with probability 1 (or high) to explore
                    # Since size is same, we accept to move in space.
                    # Update removable list
                    removable_elements[idx] = y
                    swapped = True
                    break
                else:
                    # Undo removal of x
                    cur_set.add(x)
                    update_sums_add(cur_sums, cur_set, x)
                    
            if not swapped:
                # If no swap found, do nothing (x was restored in loop)
                pass
                # Note: if swapped was False, x was restored in the last iteration of the for loop
                # Wait, if the loop finished without break, x was restored.
                # If the loop broke, x was NOT restored (it was swapped).
                # So we are good.

        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end

    return sorted(best_sa_set)

# EVOLVE-BLOCK-END
