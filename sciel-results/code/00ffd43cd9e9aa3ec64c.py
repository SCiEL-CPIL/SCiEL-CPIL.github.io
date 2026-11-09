import time
import random
import sys

def construct():
    deadline = time.time() + 145.0  # Safety margin from 150s
    fixed = [1, 2, 5, 11]
    N = 284
    fixed_set = set(fixed)
    
    def calc_conflicts(A):
        """
        Returns the number of conflicting sums (excess counts) for the set A.
        A is a list of distinct integers.
        """
        sums = {}
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                if s in sums:
                    sums[s] += 1
                else:
                    sums[s] = 1
        conflicts = 0
        for count in sums.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def greedy_init():
        """
        Greedily build a Sidon set starting with fixed elements,
        trying to add elements from N down to 1.
        """
        current_set = list(fixed)
        current_set.sort()
        # Track used sums
        used_sums = set()
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                used_sums.add(current_set[i] + current_set[j])
        
        # Try to add more elements
        # We want to maximize size, so we try to add elements that don't conflict.
        # Starting from N downwards often yields better rulers (sparse high end).
        for x in range(N, 0, -1):
            if x in current_set:
                continue
            # Check if x can be added
            valid = True
            # Check x + x
            if x + x in used_sums:
                valid = False
                break
            # Check x + y for all y in current_set
            for y in current_set:
                if x + y in used_sums:
                    valid = False
                    break
            if valid:
                current_set.append(x)
                current_set.sort()
                used_sums.add(x + x)
                for y in current_set[:-1]:
                    used_sums.add(x + y)
                # We don't need to stop, we want max size
                # But we might run out of time? Greedy is fast.
                
        return current_set

    def solve_sa(initial_set, time_limit):
        """
        Simulated Annealing with error repair.
        Returns the best set found.
        """
        start_time = time.time()
        current_set = initial_set[:]
        current_set.sort()
        current_cost = calc_conflicts(current_set)
        
        best_set = current_set[:]
        best_cost = current_cost
        
        T = 50.0  # Initial temperature
        cooling = 0.999
        min_T = 0.01
        
        steps = 0
        no_improve_count = 0
        restart_threshold = 5000  # Restart if stuck for this many steps
        
        while time.time() - start_time < time_limit:
            # Check for timeout
            if time.time() > deadline - 2.0:
                break
                
            # Generate a move
            # Strategy: 
            # 70% chance: Random swap (replace a non-fixed element)
            # 30% chance: Error repair (fix a specific conflict)
            
            move_type = random.random()
            
            if len(current_set) == 0:
                break
            
            # Identify non-fixed indices
            # Since fixed elements are small (1,2,5,11) and we add others, 
            # it's safer to check value.
            
            if move_type < 0.7:
                # Random Swap
                # Pick a random index that is NOT fixed
                non_fixed_indices = [i for i, val in enumerate(current_set) if val not in fixed_set]
                if not non_fixed_indices:
                    continue
                idx = random.choice(non_fixed_indices)
                old_val = current_set[idx]
                new_val = random.randint(1, N)
                
                # Avoid duplicates in set
                if new_val in current_set:
                    continue
                
                new_set = current_set[:]
                new_set[idx] = new_val
                new_set.sort()
                
                new_cost = calc_conflicts(new_set)
                
                if new_cost < current_cost or random.random() < pow(2.718281828, -(new_cost - current_cost) / T):
                    current_set = new_set
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_set = current_set[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                # Error Repair
                # Find a conflicting sum
                sums = {}
                pairs = {}
                n = len(current_set)
                for i in range(n):
                    for j in range(i, n):
                        s = current_set[i] + current_set[j]
                        if s not in sums:
                            sums[s] = 0
                            pairs[s] = []
                        sums[s] += 1
                        pairs[s].append((i, j))
                        
                conflicting_sums = [s for s, c in sums.items() if c > 1]
                if not conflicting_sums:
                    continue
                
                target_sum = random.choice(conflicting_sums)
                # Pick a pair from the conflicts for this sum
                # To fix the conflict, we can move one of the elements in the pair.
                # We prefer moving non-fixed elements.
                candidate_pairs = pairs[target_sum]
                
                moved = False
                for _ in range(3): # Try a few times
                    i, j = random.choice(candidate_pairs)
                    # Try moving element at index i or j
                    # Prefer non-fixed
                    move_idx = -1
                    if current_set[i] not in fixed_set and current_set[j] not in fixed_set:
                        move_idx = random.choice([i, j])
                    elif current_set[i] not in fixed_set:
                        move_idx = i
                    elif current_set[j] not in fixed_set:
                        move_idx = j
                    else:
                        continue # Both fixed, cannot repair this way directly (unless we change the fixed set, which is forbidden)
                    
                    if move_idx == -1:
                        continue
                        
                    old_val = current_set[move_idx]
                    new_val = random.randint(1, N)
                    if new_val in current_set:
                        continue
                    
                    new_set = current_set[:]
                    new_set[move_idx] = new_val
                    new_set.sort()
                    
                    new_cost = calc_conflicts(new_set)
                    
                    if new_cost < current_cost or random.random() < pow(2.718281828, -(new_cost - current_cost) / T):
                        current_set = new_set
                        current_cost = new_cost
                        moved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_set = current_set[:]
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        break
                if not moved:
                    no_improve_count += 1

            # Cooling
            T *= cooling
            if T < min_T:
                T = min_T
                # If stuck at low T, restart
                if no_improve_count > restart_threshold:
                    # Restart with a perturbed version of best
                    current_set = best_set[:]
                    # Randomly perturb a few non-fixed elements
                    non_fixed_indices = [i for i, val in enumerate(current_set) if val not in fixed_set]
                    for _ in range(min(3, len(non_fixed_indices))):
                        if non_fixed_indices:
                            idx = random.choice(non_fixed_indices)
                            current_set[idx] = random.randint(1, N)
                            # Ensure no dup
                            while current_set.count(current_set[idx]) > 1:
                                current_set[idx] = random.randint(1, N)
                    current_set.sort()
                    current_cost = calc_conflicts(current_set)
                    T = 50.0
                    no_improve_count = 0
                    restart_threshold = 5000 + random.randint(0, 5000)
                    
            steps += 1
            
            # Check if we found a perfect solution
            if best_cost == 0:
                # We found a valid Sidon set. 
                # Can we add more? 
                # The problem asks to maximize size. 
                # If we have a valid set, we can try to insert new elements into gaps.
                # However, SA usually converges to a fixed size. 
                # If we want to increase size, we need a different mode.
                # For now, if we find cost 0, we can try to "grow" it.
                # But growing is hard. 
                # Let's stick to the current size optimization first.
                # If we have time, we can try to add elements to best_set.
                
                # Try to add elements to best_set to increase size
                # This is a "growth" phase
                growth_set = best_set[:]
                used_sums = set()
                for i in range(len(growth_set)):
                    for j in range(i, len(growth_set)):
                        used_sums.add(growth_set[i] + growth_set[j])
                        
                added_any = False
                # Try to add elements from N downwards
                for x in range(N, 0, -1):
                    if x in growth_set:
                        continue
                    valid = True
                    if x + x in used_sums:
                        valid = False
                        break
                    for y in growth_set:
                        if x + y in used_sums:
                            valid = False
                            break
                    if valid:
                        growth_set.append(x)
                        growth_set.sort()
                        used_sums.add(x + x)
                        for y in growth_set[:-1]:
                            used_sums.add(x + y)
                        added_any = True
                        
                if added_any:
                    best_set = growth_set
                    best_cost = 0 # Still valid
                    # Update current to continue search from this larger set?
                    # Yes, let's set current to best and reset some params to keep looking for even larger?
                    # No, usually once size is maxed for a specific range, you can't add more.
                    # But maybe we can rearrange to allow more?
                    # Let's just return this if it's good.
                    # Actually, let's let the loop run out or restart to see if we can find a better arrangement for the same size.
                    
        return best_set

    # 1. Greedy Initialization
    initial_set = greedy_init()
    
    # 2. SA Optimization
    # We will run SA for the remaining time
    # We want to maximize size. The greedy set might be small.
    # Let's ensure we try to find a set of size at least 15-20.
    
    # If greedy set is small, we might need to force a larger size?
    # No, SA works on the current set size. 
    # If we want to increase size, we need to start with a larger set.
    # Let's create a "random large" set as well to give SA a chance to find a valid large set.
    
    # Strategy: 
    # Run SA on the greedy set (likely valid or near-valid, size ~15-18).
    # If it converges to 0, try to grow it.
    
    # Let's also try a "random large" initialization:
    # Pick K=20 random elements + fixed? No, fixed are included.
    # Pick 16 random elements from 1..284, add fixed (if not already there, but they are distinct).
    # Total size 20.
    # This might have high conflict, but SA might fix it.
    
    # We will alternate or run sequentially?
    # Time is 145s.
    
    # Phase 1: Optimize Greedy Set (10s)
    t_limit_1 = 10.0
    result1 = solve_sa(initial_set, t_limit_1)
    
    # Phase 2: Try Random Large Set (30s)
    # Create a random set of size 20 containing fixed
    random_set = set(fixed)
    while len(random_set) < 20:
        x = random.randint(1, N)
        random_set.add(x)
    random_list = list(random_set)
    
    t_limit_2 = 30.0
    result2 = solve_sa(random_list, t_limit_2)
    
    # Phase 3: Try another Random Large Set (30s)
    random_set2 = set(fixed)
    while len(random_set2) < 20:
        x = random.randint(1, N)
        random_set2.add(x)
    random_list2 = list(random_set2)
    
    t_limit_3 = 30.0
    result3 = solve_sa(random_list2, t_limit_3)
    
    # Phase 4: Optimize the best found so far with remaining time
    # Sort results by size, then by cost (though cost should be 0 for valid)
    # We only care about valid sets (cost 0).
    # If none are valid, we return the one with lowest cost? No, invalid is score 0.
    # We MUST return a valid set.
    
    best_valid = None
    best_size = 0
    
    candidates = [result1, result2, result3]
    for res in candidates:
        if res is None:
            continue
        cost = calc_conflicts(res)
        if cost == 0:
            if len(res) > best_size:
                best_size = len(res)
                best_valid = res
        else:
            # If no valid set found yet, keep track of best "invalid" just in case?
            # No, evaluator scores 0 for invalid.
            # But maybe we can repair it?
            # Let's just look for valid ones.
            pass
            
    if best_valid is None:
        # Fallback: Use the greedy set and try to repair it aggressively
        # Or just return the greedy set? It might be valid.
        # Greedy usually produces a valid set.
        cost_greedy = calc_conflicts(initial_set)
        if cost_greedy == 0:
            best_valid = initial_set
            best_size = len(initial_set)
        else:
            # Try one more quick repair on the best invalid one?
            # Let's assume one of the SA runs found something.
            # If not, we are in trouble.
            # Let's pick the one with lowest conflict.
            min_conf = float('inf')
            best_invalid = None
            for res in candidates:
                if res:
                    c = calc_conflicts(res)
                    if c < min_conf:
                        min_conf = c
                        best_invalid = res
            # If still no valid, return best_invalid? It will score 0.
            # But maybe we can do a final greedy repair on it?
            # Let's try a simple greedy addition/removal to fix conflicts.
            if best_invalid:
                # Simple repair: remove elements causing most conflicts
                # This is complex. 
                # Let's just hope SA found a valid one.
                # If not, we return the greedy set if it's valid, else the best invalid.
                # Wait, the greedy set is generated to be valid. 
                # So initial_set is likely valid.
                # Why did solve_sa not return it? 
                # It returns best_set. If current_cost > 0 and best_cost > 0, it returns best_invalid.
                # But initial_set was valid. So best_cost should be 0.
                # Unless I messed up the logic.
                # Ah, solve_sa starts with initial_set. 
                # So if initial_set is valid, best_cost is 0.
                # So result1 should be valid.
                pass
            best_valid = best_invalid # Fallback, likely 0 score

    # Final Attempt: If we have time, try to grow the best_valid set
    if best_valid is not None:
        # Try to add elements to best_valid
        growth_set = best_valid[:]
        used_sums = set()
        for i in range(len(growth_set)):
            for j in range(i, len(growth_set)):
                used_sums.add(growth_set[i] + growth_set[j])
                
        # Try adding from N down
        for x in range(N, 0, -1):
            if x in growth_set:
                continue
            valid = True
            if x + x in used_sums:
                valid = False
                break
            for y in growth_set:
                if x + y in used_sums:
                    valid = False
                    break
            if valid:
                growth_set.append(x)
                growth_set.sort()
                used_sums.add(x + x)
                for y in growth_set[:-1]:
                    used_sums.add(x + y)
                
        if len(growth_set) > len(best_valid):
            best_valid = growth_set

    return sorted(best_valid)

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Set: {res}")
    # Verify
    sums = set()
    valid = True
    for i in range(len(res)):
        for j in range(i, len(res)):
            s = res[i] + res[j]
            if s in sums:
                valid = False
            sums.add(s)
    print(f"Valid: {valid}")
