import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    # Prescribed elements
    base = [1, 2, 5, 11]
    base_set = set(base)
    
    # Helper to check validity and get sums
    def get_validity_and_sums(candidates):
        candidates = list(candidates)
        n = len(candidates)
        sums = set()
        for i in range(n):
            a = candidates[i]
            for j in range(i, n):
                b = candidates[j]
                s = a + b
                if s in sums:
                    return False, None
                sums.add(s)
        return True, sums

    def try_expand(current_set, max_size=20):
        """
        Greedily try to add elements to current_set.
        Returns the expanded set.
        """
        curr = list(current_set)
        curr.sort()
        valid, sums = get_validity_and_sums(curr)
        if not valid:
            return current_set
        
        remaining = [i for i in range(1, 285) if i not in current_set]
        random.shuffle(remaining)
        
        # Try to add as many as possible
        for x in remaining:
            if len(curr) >= max_size:
                break
            new_sums = {x + y for y in curr} | {2 * x}
            if not (new_sums & sums):
                curr.append(x)
                curr.sort()
                sums |= new_sums
        
        return set(curr)

    def kick_and_expand(current_set):
        """
        Remove a random non-base element, then try to re-expand.
        """
        if len(current_set) <= len(base):
            return current_set
            
        non_base = [x for x in current_set if x not in base_set]
        if not non_base:
            return current_set
            
        x_to_remove = random.choice(non_base)
        new_set = current_set - {x_to_remove}
        
        # Try to expand the new set
        expanded = try_expand(new_set)
        return expanded

    def simulated_annealing_improve(current_set, max_iter=5000, time_limit=30):
        """
        Use simulated annealing with swap moves to try to find a larger valid set.
        The state is the set of non-base elements. We try swaps that might allow
        adding more elements later.
        
        Actually, a better approach: maintain a set of size K, and try to find
        a different set of size K that has more "room" for additional elements.
        
        Alternative: try to directly grow the set by finding a sequence of swaps
        that enables an additional element to be added.
        
        Let's use a different SA formulation:
        - State: a valid Golomb ruler containing base
        - Move: swap one non-base element with a different number, or add/remove
        - Objective: maximize size (primary), minimize "conflict pressure" (secondary)
        
        Simpler approach: try to find a set of size target_size using SA.
        """
        curr = set(current_set)
        best = set(curr)
        best_size = len(curr)
        
        sa_start = time.time()
        
        # Try to grow from current size
        target_sizes = range(len(curr) + 1, 21)
        
        for target_size in target_sizes:
            if time.time() - sa_start > time_limit:
                break
            
            # Try to find a valid set of exactly target_size
            # Start from current best and try to add one element via swaps
            success = False
            for attempt in range(200):
                if time.time() - sa_start > time_limit:
                    break
                    
                # Start from a copy of best
                trial = set(best)
                
                # Try to add a random element
                if len(trial) < target_size:
                    candidates = [i for i in range(1, 285) if i not in trial]
                    if not candidates:
                        break
                    x = random.choice(candidates)
                    
                    # Check if x can be added
                    valid, sums = get_validity_and_sums(trial)
                    if not valid:
                        continue
                    new_sums = {x + y for y in trial} | {2 * x}
                    if not (new_sums & sums):
                        trial.add(x)
                        if len(trial) > best_size:
                            best = trial
                            best_size = len(trial)
                            success = True
                        # Keep going to try to add more
                        continue
                    else:
                        # x conflicts. Try swapping a non-base element with x
                        non_base = [e for e in trial if e not in base_set]
                        if not non_base:
                            continue
                        to_swap = random.choice(non_base)
                        
                        # Try replacing to_swap with x
                        trial2 = trial - {to_swap} | {x}
                        valid2, _ = get_validity_and_sums(trial2)
                        if valid2 and len(trial2) == len(trial):
                            # Check if this allows adding another element later
                            # For now, just accept if it's valid
                            # We'll evaluate by trying to extend
                            pass
                        # Also try just removing to_swap and re-expanding
                        trial3 = trial - {to_swap}
                        expanded = try_expand(trial3, target_size)
                        if len(expanded) > best_size:
                            best = expanded
                            best_size = len(expanded)
                            success = True
            
            if best_size >= target_size:
                continue
            else:
                # Couldn't reach this target size, stop trying larger sizes
                break
        
        return best

    # Main search loop
    best_size_found = len(base)
    best_sol_found = base.copy()
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Phase 1: Randomized greedy expansion from base
        sol = try_expand(base_set)
        
        if len(sol) > best_size_found:
            best_size_found = len(sol)
            best_sol_found = sol.copy()
            if best_size_found == 20:
                return sorted(best_sol_found)
        
        # Phase 2: Kick and expand
        if best_size_found > len(base):
            kicked = kick_and_expand(set(best_sol_found))
            if len(kicked) > best_size_found:
                best_size_found = len(kicked)
                best_sol_found = kicked.copy()
                if best_size_found == 20:
                    return sorted(best_sol_found)
        
        # Phase 3: Simulated annealing improvement
        if best_size_found > len(base) and time.time() < deadline - 10:
            sa_result = simulated_annealing_improve(set(best_sol_found), time_limit=20)
            if len(sa_result) > best_size_found:
                best_size_found = len(sa_result)
                best_sol_found = sa_result.copy()
                if best_size_found == 20:
                    return sorted(best_sol_found)
        
        # Occasionally do a fresh start with different random seed behavior
        if attempts % 10 == 0:
            # Try expanding from a slightly perturbed base
            # Remove one base element temporarily? No, base must be included.
            # Instead, try different expansion orders
            for _ in range(3):
                sol2 = try_expand(base_set)
                if len(sol2) > best_size_found:
                    best_size_found = len(sol2)
                    best_sol_found = sol2.copy()
                    if best_size_found == 20:
                        return sorted(best_sol_found)
        
        if time.time() > deadline - 5:
            break

    return sorted(best_sol_found)
