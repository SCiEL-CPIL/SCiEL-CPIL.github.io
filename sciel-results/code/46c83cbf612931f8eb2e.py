"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145
    prescribed = [1, 2, 5, 11]
    N = 284
    sys.setrecursionlimit(100000)

    def is_valid_sidon(arr):
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Efficient CSP solver using difference bitmask for O(1) conflict checks
    def solve_csp(target_k, time_budget):
        bt_start = time.time()
        prescribed_set = set(prescribed)
        
        # diff_mask: bit d is set if difference d is already used
        # We use a list of ints as a bitset for differences 1..283
        # diff_mask[d] tracks count, but for CSP we just need to know if it's used
        # Use a boolean array for simplicity and speed
        diff_used = [False] * (N + 1)
        
        # Initialize with prescribed elements
        for i in range(len(prescribed)):
            for j in range(i + 1, len(prescribed)):
                d = prescribed[j] - prescribed[i]
                diff_used[d] = True
        
        assigned = list(prescribed)
        used_vals = set(prescribed)
        
        # For each candidate x, precompute the set of differences it would create
        # with each possible assigned value - but this changes as we assign more
        # So we check on the fly
        
        def check_candidate(x, current_assigned, current_diff_used):
            """Check if x can be added without creating duplicate differences"""
            for y in current_assigned:
                d = abs(x - y)
                if current_diff_used[d]:
                    return False
            return True
        
        def get_valid_candidates(current_assigned, current_diff_used, current_used):
            """Get all valid candidates that don't create conflicts"""
            valid = []
            for x in range(1, N + 1):
                if x in current_used:
                    continue
                ok = True
                for y in current_assigned:
                    d = abs(x - y)
                    if current_diff_used[d]:
                        ok = False
                        break
                if ok:
                    valid.append(x)
            return valid
        
        def dfs(depth, assigned_list, diff_used_arr, used_vals_set):
            if time.time() - bt_start > time_budget:
                return None
            
            if depth == target_k:
                return list(assigned_list)
            
            # Get valid candidates
            candidates = get_valid_candidates(assigned_list, diff_used_arr, used_vals_set)
            
            if not candidates:
                return None
            
            # Heuristic ordering: prefer candidates that are "far" from existing ones
            # to leave more room for future placements
            # Score: sum of distances to existing elements (higher = more spread)
            def score_candidate(x):
                total_dist = 0
                for y in assigned_list:
                    total_dist += abs(x - y)
                return total_dist
            
            # Sort by score descending (most spread first) with some randomization
            candidates.sort(key=score_candidate, reverse=True)
            
            # Add randomness to break ties and explore different branches
            # Shuffle the top candidates slightly
            if len(candidates) > 10:
                # Keep top 10 in order, shuffle the rest
                top = candidates[:10]
                rest = candidates[10:]
                random.shuffle(rest)
                candidates = top + rest
            
            # Limit search breadth
            max_candidates = min(len(candidates), 60)
            candidates = candidates[:max_candidates]
            
            for x in candidates:
                if time.time() - bt_start > time_budget:
                    return None
                
                # Check validity (should already be valid from get_valid_candidates, but double-check)
                new_diffs = []
                valid = True
                for y in assigned_list:
                    d = abs(x - y)
                    if diff_used_arr[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Apply assignment
                for d in new_diffs:
                    diff_used_arr[d] = True
                assigned_list.append(x)
                used_vals_set.add(x)
                
                result = dfs(depth + 1, assigned_list, diff_used_arr, used_vals_set)
                if result is not None:
                    return result
                
                # Undo assignment
                for d in new_diffs:
                    diff_used_arr[d] = False
                assigned_list.pop()
                used_vals_set.discard(x)
            
            return None
        
        # Try with different random seeds for diversity
        for attempt in range(100):
            if time.time() - bt_start > time_budget:
                break
            random.seed(42 + attempt * 137)
            result = dfs(len(prescribed), list(prescribed), list(diff_used), set(prescribed))
            if result is not None:
                if is_valid_sidon(result):
                    return result
        
        return None

    # Greedy construction as fallback
    def greedy_construct():
        a = list(prescribed)
        used = set(prescribed)
        diffs = set()
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                diffs.add(a[j] - a[i])
        
        for x in range(1, N + 1):
            if x in used:
                continue
            new_diffs = set()
            ok = True
            for y in a:
                d = abs(x - y)
                if d in diffs or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if ok:
                a.append(x)
                used.add(x)
                diffs |= new_diffs
            if len(a) >= 20:
                break
        
        return a

    # Local search to improve a partial solution
    def local_search(initial_sol, time_budget):
        ls_start = time.time()
        n = len(initial_sol)
        sol = list(initial_sol)
        sol.sort()
        
        # Build diff state
        def build_diffs(arr):
            d = set()
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    d.add(arr[j] - arr[i])
            return d
        
        diff_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                d = sol[j] - sol[i]
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        conflicts = sum(1 for c in diff_counts.values() if c > 1)
        best_sol = list(sol)
        best_conflicts = conflicts
        
        prescribed_set = set(prescribed)
        movable_indices = [i for i in range(n) if sol[i] not in prescribed_set]
        
        if not movable_indices:
            return sol, conflicts
        
        T = 2.0
        T_min = 0.01
        cooling = 0.9999
        
        while time.time() - ls_start < time_budget:
            if conflicts == 0:
                break
            
            if len(movable_indices) < 2:
                idx = random.choice(movable_indices)
                old_val = sol[idx]
                new_val = random.randint(1, N)
                if new_val == old_val or new_val in sol:
                    continue
                
                # Remove old differences
                for j in range(n):
                    if j == idx:
                        continue
                    d = abs(sol[j] - old_val)
                    diff_counts[d] -= 1
                    if diff_counts[d] == 1:
                        conflicts -= 1
                sol[idx] = new_val
                
                # Add new differences
                for j in range(n):
                    if j == idx:
                        continue
                    d = abs(sol[j] - new_val)
                    if d not in diff_counts:
                        diff_counts[d] = 0
                    diff_counts[d] += 1
                    if diff_counts[d] == 2:
                        conflicts += 1
                
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_sol = list(sol)
                
                if conflicts > 0 and random.random() >= math.exp(-conflicts / T):
                    # Revert
                    for j in range(n):
                        if j == idx:
                            continue
                        d = abs(sol[j] - new_val)
                        diff_counts[d] -= 1
                        if diff_counts[d] == 1:
                            conflicts -= 1
                    sol[idx] = old_val
                    for j in range(n):
                        if j == idx:
                            continue
                        d = abs(sol[j] - old_val)
                        if d not in diff_counts:
                            diff_counts[d] = 0
                        diff_counts[d] += 1
                        if diff_counts[d] == 2:
                            conflicts += 1
            else:
                i1, i2 = random.sample(movable_indices, 2)
                v1, v2 = sol[i1], sol[i2]
                if v1 == v2:
                    continue
                
                # Remove v1 and v2 differences
                for j in range(n):
                    if j == i1 or j == i2:
                        continue
                    d = abs(sol[j] - v1)
                    diff_counts[d] -= 1
                    if diff_counts[d] == 1:
                        conflicts -= 1
                    d2 = abs(sol[j] - v2)
                    diff_counts[d2] -= 1
                    if diff_counts[d2] == 1:
                        conflicts -= 1
                d_swap = abs(v1 - v2)
                diff_counts[d_swap] -= 1
                if diff_counts[d_swap] == 1:
                    conflicts -= 1
                
                # Swap
                sol[i1], sol[i2] = v2, v1
                
                # Add new differences
                for j in range(n):
                    if j == i1 or j == i2:
                        continue
                    d = abs(sol[j] - v2)
                    if d not in diff_counts:
                        diff_counts[d] = 0
                    diff_counts[d] += 1
                    if diff_counts[d] == 2:
                        conflicts += 1
                    d2 = abs(sol[j] - v1)
                    if d2 not in diff_counts:
                        diff_counts[d2] = 0
                    diff_counts[d2] += 1
                    if diff_counts[d2] == 2:
                        conflicts += 1
                d_swap = abs(v1 - v2)
                if d_swap not in diff_counts:
                    diff_counts[d_swap] = 0
                diff_counts[d_swap] += 1
                if diff_counts[d_swap] == 2:
                    conflicts += 1
                
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_sol = list(sol)
                
                if conflicts > 0 and random.random() >= math.exp(-conflicts / T):
                    # Revert
                    for j in range(n):
                        if j == i1 or j == i2:
                            continue
                        d = abs(sol[j] - v2)
                        diff_counts[d] -= 1
                        if diff_counts[d] == 1:
                            conflicts -= 1
                        d2 = abs(sol[j] - v1)
                        diff_counts[d2] -= 1
                        if diff_counts[d2] == 1:
                            conflicts -= 1
                    d_swap = abs(v1 - v2)
                    diff_counts[d_swap] -= 1
                    if diff_counts[d_swap] == 1:
                        conflicts -= 1
                    
                    sol[i1], sol[i2] = v1, v2
                    
                    for j in range(n):
                        if j == i1 or j == i2:
                            continue
                        d = abs(sol[j] - v1)
                        if d not in diff_counts:
                            diff_counts[d] = 0
                        diff_counts[d] += 1
                        if diff_counts[d] == 2:
                            conflicts += 1
                        d2 = abs(sol[j] - v2)
                        if d2 not in diff_counts:
                            diff_counts[d2] = 0
                        diff_counts[d2] += 1
                        if diff_counts[d2] == 2:
                            conflicts += 1
                    d_swap = abs(v1 - v2)
                    if d_swap not in diff_counts:
                        diff_counts[d_swap] = 0
                    diff_counts[d_swap] += 1
                    if diff_counts[d_swap] == 2:
                        conflicts += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_sol, best_conflicts

    # Main logic
    best_a = None
    best_size = len(prescribed)

    # Step 1: Try CSP for size 20
    remaining = deadline - time.time() - 5
    time_budget_csp = min(60, remaining * 0.6)
    result = solve_csp(20, time_budget_csp)
    if result is not None:
        best_a = sorted(result)
        best_size = 20
    else:
        # Step 2: Try CSP for smaller sizes
        for target_k in [19, 18, 17]:
            if time.time() > deadline - 25:
                break
            time_budget = min(20, deadline - time.time() - 25)
            result = solve_csp(target_k, time_budget)
            if result is not None:
                best_a = sorted(result)
                best_size = target_k
                break

    # Step 3: Fallback to greedy + local search
    if best_a is None or best_size < 20:
        greedy_sol = greedy_construct()
        if len(greedy_sol) > best_size:
            best_size = len(greedy_sol)
            best_a = sorted(greedy_sol)
        
        # Try to improve with local search
        for target_k in [20, 19, 18]:
            if time.time() > deadline - 15:
                break
            if target_k <= best_size:
                continue
            
            extra_needed = target_k - len(prescribed)
            greedy_extra = [x for x in greedy_sol if x not in prescribed]
            chosen = list(greedy_extra[:extra_needed]) if len(greedy_extra) >= extra_needed else list(greedy_extra)
            if len(chosen) < extra_needed:
                available = [x for x in range(1, N + 1) if x not in prescribed and x not in chosen]
                random.shuffle(available)
                chosen += available[:extra_needed - len(chosen)]
            
            sol = list(prescribed) + chosen
            sol.sort()
            
            time_budget_ls = min(15, deadline - time.time() - 10)
            improved_sol, conflicts = local_search(sol, time_budget_ls)
            
            if conflicts == 0 and is_valid_sidon(improved_sol):
                if target_k > best_size:
                    best_size = target_k
                    best_a = sorted(improved_sol)
                break
            elif is_valid_sidon(improved_sol) and len(improved_sol) > best_size:
                best_size = len(improved_sol)
                best_a = sorted(improved_sol)

    # Final verification
    if best_a is not None and is_valid_sidon(best_a):
        return best_a
    return sorted(prescribed)
# EVOLVE-BLOCK-END
