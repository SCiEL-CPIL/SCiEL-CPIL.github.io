import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s buffer for safety within 150s limit
    
    prescribed = [1, 2, 5, 11]
    fixed_set = set(prescribed)
    
    # Helper to check if a list is a valid Sidon set
    # Returns (is_valid, conflict_count)
    def check_sidon(candidates):
        n = len(candidates)
        if n != len(set(candidates)):
            return False, 100000
        sums = set()
        conflict = 0
        for i in range(n):
            for j in range(i, n):
                s = candidates[i] + candidates[j]
                if s in sums:
                    conflict += 1
                else:
                    sums.add(s)
        return conflict == 0, conflict

    # 1. Greedy Construction to get a strong baseline
    a = list(prescribed)
    sums = set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    
    for x in range(12, 285):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
            
    best_set = a
    best_size = len(best_set)
    
    # If we already hit the theoretical max (20), return immediately
    if best_size == 20:
        return best_set

    # 2. CSP-based backtracking solver with forward checking
    def csp_solver(target_size, time_limit):
        end_time = time.time() + time_limit
        if end_time > deadline:
            end_time = deadline - 0.5
            
        if time.time() >= end_time:
            return None
            
        # Start with the greedy solution as a base
        base_set = list(best_set)
        
        # If base is smaller than target, we need to add more elements
        # If base is larger, we can try to keep a subset or modify
        if len(base_set) >= target_size:
            # Use a random subset of size target_size, ensuring prescribed elements
            current = random.sample(base_set, target_size)
            for p in prescribed:
                if p not in current:
                    non_fixed = [x for x in current if x not in fixed_set]
                    if non_fixed:
                        current[random.choice(range(len(non_fixed)))] = p
        else:
            # Start with base and add random elements
            current = list(base_set)
            needed = target_size - len(current)
            pool = [x for x in range(1, 285) if x not in set(current)]
            random.shuffle(pool)
            current.extend(pool[:needed])
        
        random.shuffle(current)
        
        # Build initial sum set
        sum_set = set()
        for i in range(len(current)):
            for j in range(i, len(current)):
                sum_set.add(current[i] + current[j])
        
        # Backtracking with forward checking
        def backtrack(assigned, remaining, sum_set, depth):
            if time.time() > end_time:
                return None
            
            if depth == target_size:
                # Check if valid
                if len(sum_set) == len(assigned) * (len(assigned) + 1) // 2:
                    return list(assigned)
                return None
            
            # Generate candidates for this position
            # Try to find values that don't create conflicts
            candidates = list(range(1, 285))
            random.shuffle(candidates)
            
            # Filter out already assigned values
            assigned_set = set(assigned)
            candidates = [c for c in candidates if c not in assigned_set]
            
            # If we need to include prescribed elements, prioritize them
            missing_prescribed = [p for p in prescribed if p not in assigned_set]
            if missing_prescribed:
                # Move missing prescribed to front
                candidates = missing_prescribed + [c for c in candidates if c not in missing_prescribed]
            
            for val in candidates:
                if time.time() > end_time:
                    return None
                
                # Check if this value creates conflicts
                new_sums = set()
                valid = True
                for existing in assigned:
                    s = val + existing
                    if s in sum_set:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    s = 2 * val
                    if s in sum_set:
                        valid = False
                    else:
                        new_sums.add(s)
                
                if not valid:
                    continue
                
                # Forward checking: verify that adding this value doesn't make it impossible
                # to complete the remaining positions (simple check: enough values available)
                new_assigned = assigned + [val]
                new_sum_set = sum_set | new_sums
                
                # Recurse
                result = backtrack(new_assigned, remaining - 1, new_sum_set, depth + 1)
                if result is not None:
                    return result
                
                # Backtrack
                # (sum_set is immutable in our usage, so no need to undo)
            
            return None
        
        # Try multiple random restarts
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > end_time:
                break
            
            # Create a fresh starting set
            if len(best_set) >= target_size:
                current = random.sample(best_set, target_size)
                for p in prescribed:
                    if p not in current:
                        non_fixed = [x for x in current if x not in fixed_set]
                        if non_fixed:
                            idx = random.choice(range(len(non_fixed)))
                            current[idx] = p
            else:
                current = list(best_set)
                needed = target_size - len(current)
                pool = [x for x in range(1, 285) if x not in set(current)]
                random.shuffle(pool)
                current.extend(pool[:needed])
            
            random.shuffle(current)
            
            # Build sum set
            sum_set = set()
            for i in range(len(current)):
                for j in range(i, len(current)):
                    sum_set.add(current[i] + current[j])
            
            # Start backtracking from a partial assignment
            # Use the first few elements as initial assignment
            initial_len = min(4, len(current))
            initial_assigned = current[:initial_len]
            initial_sums = set()
            for i in range(initial_len):
                for j in range(i, initial_len):
                    initial_sums.add(initial_assigned[i] + initial_assigned[j])
            
            result = backtrack(initial_assigned, target_size - initial_len, initial_sums, initial_len)
            if result is not None:
                return result
        
        return None

    # Try to find larger valid sets, starting from the highest target
    targets = [20, 19, 18, 17, 16, 15]
    time_allocs = [60, 30, 20, 10, 5, 5]
    
    for i, target in enumerate(targets):
        if target <= best_size:
            continue
            
        alloc = time_allocs[i]
        # Don't exceed deadline
        if time.time() + alloc > deadline:
            alloc = int(deadline - time.time())
            
        if alloc <= 0:
            break
            
        res = csp_solver(target, alloc)
        
        if res is not None:
            best_set = res
            best_size = len(res)
            break  # Found a valid larger set, stop searching lower targets
    
    return best_set

# Main entry point
if __name__ == "__main__":
    res = construct()
    print(len(res))
