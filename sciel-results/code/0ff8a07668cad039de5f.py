import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Known good solution: {66, ..., 100}, size 35
    # Max sum of 2 elements is 99+100=199.
    # Min sum of 3 elements is 66+67+68=201.
    # Thus, no subset sums to 200.
    best_set = set(range(66, 101))
    best_size = 35

    def is_valid(s):
        """Check if no subset of s sums to 200."""
        sums = {0}
        for x in s:
            new = {y + x for y in sums if y + x <= 200}
            if 200 in new:
                return False
            sums |= new
        return True

    def compute_reachable(s):
        """Compute all subset sums up to 200 for set s."""
        reachable = {0}
        for x in s:
            new_sums = {y + x for y in reachable if y + x <= 200}
            reachable |= new_sums
        return reachable

    # Precompute all conflict sets (subsets of [1..100] summing to 200)
    # For each element x, precompute which other elements form conflicts with it
    # A conflict for element x is a set C such that x + sum(C) = 200, i.e., sum(C) = 200-x
    
    # Precompute for each target sum t (1 to 199), all subsets of [1..100] that sum to t
    # This is expensive, so we'll be smart about it
    # For small t, only small elements can be in the subset
    # For large t, we can use the complement idea
    
    # Actually, let's use a different approach:
    # For each element x, precompute the set of "conflict partners" - elements y such that
    # there exists a subset containing x and y that sums to 200.
    
    # Even simpler: for each element x, precompute which elements y cannot be in the same
    # set as x if we want to avoid sum=200. But this is not quite right because conflicts
    # can involve more than 2 elements.
    
    # Let's use the greedy heuristic with conflict density:
    # For each element x, count how many subsets of [1..100]\{x} sum to 200-x.
    # Elements with fewer such subsets are "safer" to include.
    
    # Precompute subset sum counts for each target
    def count_subsets_summing_to(target, elements):
        """Count subsets of elements that sum to target."""
        if target == 0:
            return 1
        if target < 0 or not elements:
            return 0
        
        # Use DP
        n = len(elements)
        # dp[j] = number of ways to make sum j using first i elements
        # But this can be expensive for large target
        # For our case, target <= 199, so it's manageable
        
        dp = [0] * (target + 1)
        dp[0] = 1
        for elem in elements:
            if elem > target:
                continue
            for j in range(target, elem - 1, -1):
                dp[j] += dp[j - elem]
        return dp[target]
    
    # Precompute conflict counts for each element
    all_elements = list(range(1, 101))
    conflict_counts = {}
    for x in all_elements:
        target = 200 - x
        if target < 0:
            conflict_counts[x] = 0
        else:
            other_elements = [y for y in all_elements if y != x]
            conflict_counts[x] = count_subsets_summing_to(target, other_elements)
    
    # Sort elements by conflict count (ascending - fewer conflicts first)
    sorted_by_conflicts = sorted(all_elements, key=lambda x: conflict_counts[x])
    
    # Greedy construction: add elements in order of fewest conflicts
    def greedy_by_conflict_order(order):
        """Greedy construction using a given order of elements."""
        s = set()
        reachable = {0}
        
        for x in order:
            if x in s:
                continue
            if (200 - x) in reachable:
                continue
            new_sums = {y + x for y in reachable if y + x <= 200}
            if 200 in new_sums:
                continue
            s.add(x)
            reachable |= new_sums
        
        return s, reachable
    
    # Try greedy with conflict-based ordering
    s_conflict, _ = greedy_by_conflict_order(sorted_by_conflicts)
    if len(s_conflict) > best_size and is_valid(s_conflict):
        best_size = len(s_conflict)
        best_set = s_conflict
    
    # Also try reverse conflict order (most conflicts first) as a heuristic
    sorted_by_conflicts_desc = sorted(all_elements, key=lambda x: -conflict_counts[x])
    s_conflict_desc, _ = greedy_by_conflict_order(sorted_by_conflicts_desc)
    if len(s_conflict_desc) > best_size and is_valid(s_conflict_desc):
        best_size = len(s_conflict_desc)
        best_set = s_conflict_desc
    
    # Local search to improve
    def local_search(initial_set, max_iters=500):
        """Local search to improve the set."""
        s = set(initial_set)
        reachable = compute_reachable(s)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add a single element
            candidates = [x for x in range(1, 101) if x not in s]
            # Sort by conflict count for better heuristic
            candidates.sort(key=lambda x: conflict_counts.get(x, 0))
            
            for x in candidates:
                if (200 - x) in reachable:
                    continue
                new_sums = {y + x for y in reachable if y + x <= 200}
                if 200 in new_sums:
                    continue
                s.add(x)
                reachable |= new_sums
                improved = True
                break
            
            if improved:
                continue
            
            # Try to swap: remove one, add two
            elements_in_s = list(s)
            random.shuffle(elements_in_s)
            
            for x_out in elements_in_s:
                if time.time() > deadline:
                    break
                
                s_temp = s - {x_out}
                reachable_temp = compute_reachable(s_temp)
                
                # Try to add two elements
                candidates = [x for x in range(1, 101) if x not in s_temp]
                candidates.sort(key=lambda x: conflict_counts.get(x, 0))
                
                added = []
                s_curr = s_temp
                r_curr = reachable_temp
                
                for x1 in candidates:
                    if (200 - x1) in r_curr:
                        continue
                    new_sums = {y + x1 for y in r_curr if y + x1 <= 200}
                    if 200 in new_sums:
                        continue
                    s_curr.add(x1)
                    r_curr |= new_sums
                    added.append(x1)
                    break
                
                if len(added) == 1:
                    for x2 in candidates:
                        if x2 in s_curr:
                            continue
                        if (200 - x2) in r_curr:
                            continue
                        new_sums = {y + x2 for y in r_curr if y + x2 <= 200}
                        if 200 in new_sums:
                            continue
                        s_curr.add(x2)
                        r_curr |= new_sums
                        added.append(x2)
                        break
                
                if len(added) == 2:
                    s = s_curr
                    reachable = r_curr
                    improved = True
                    break
            
            if not improved:
                break
        
        return s
    
    # Run local search on best solution found so far
    if time.time() < deadline:
        improved_set = local_search(best_set, max_iters=300)
        if len(improved_set) > best_size and is_valid(improved_set):
            best_size = len(improved_set)
            best_set = improved_set
    
    # Try a few random restarts with conflict-based greedy
    for _ in range(5):
        if time.time() > deadline:
            break
        
        # Perturb the conflict order slightly
        perturbed_order = sorted_by_conflicts.copy()
        for i in range(len(perturbed_order) - 1):
            if random.random() < 0.1:  # 10% chance to swap
                j = i + random.randint(0, 2)
                if j < len(perturbed_order):
                    perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
        
        s_perturbed, _ = greedy_by_conflict_order(perturbed_order)
        if len(s_perturbed) > best_size and is_valid(s_perturbed):
            best_size = len(s_perturbed)
            best_set = s_perturbed
        
        # Try local search on this
        if len(s_perturbed) >= best_size - 1 and time.time() < deadline - 10:
            improved_perturbed = local_search(s_perturbed, max_iters=100)
            if len(improved_perturbed) > best_size and is_valid(improved_perturbed):
                best_size = len(improved_perturbed)
                best_set = improved_perturbed
    
    # Final validation
    if is_valid(best_set):
        return list(best_set)
    else:
        return list(range(66, 101))
