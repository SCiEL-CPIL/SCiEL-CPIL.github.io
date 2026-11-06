import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    # Prescribed elements
    base = [1, 2, 5, 11]
    base_set = set(base)
    
    # Precompute: for a given set, what sums are used
    def compute_sums(candidates):
        n = len(candidates)
        sums = set()
        for i in range(n):
            a = candidates[i]
            for j in range(i, n):
                b = candidates[j]
                s = a + b
                sums.add(s)
        return sums

    def is_valid(candidates):
        n = len(candidates)
        sums = set()
        for i in range(n):
            a = candidates[i]
            for j in range(i, n):
                b = candidates[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def count_conflicts(x, current_set):
        """Count how many sums would conflict if x were added."""
        conflicts = 0
        sums = compute_sums(current_set)
        for y in current_set:
            s = x + y
            if s in sums:
                conflicts += 1
        s2 = 2 * x
        if s2 in sums:
            conflicts += 1
        return conflicts

    def try_expand(current_set, max_size=20):
        """
        Greedily try to add elements to current_set.
        Uses a conflict-aware heuristic: prefer candidates with fewer conflicts.
        """
        curr = list(current_set)
        curr.sort()
        if not is_valid(curr):
            return current_set
        
        sums = compute_sums(curr)
        remaining = [i for i in range(1, 285) if i not in current_set]
        random.shuffle(remaining)
        
        # Try to add as many as possible
        # Sort remaining by number of conflicts (fewer first) for better greedy
        # But this is expensive, so we'll just shuffle and try
        
        added = 0
        for x in remaining:
            if len(curr) >= max_size:
                break
            new_sums = {x + y for y in curr} | {2 * x}
            if not (new_sums & sums):
                curr.append(x)
                curr.sort()
                sums |= new_sums
                added += 1
        
        return set(curr)

    def try_expand_heuristic(current_set, max_size=20):
        """
        Greedy expansion that prioritizes candidates with fewer conflicts.
        This may find larger sets than pure random.
        """
        curr = list(current_set)
        curr.sort()
        if not is_valid(curr):
            return current_set
        
        sums = compute_sums(curr)
        remaining = [i for i in range(1, 285) if i not in current_set]
        
        # Score each candidate by number of conflicts
        scored = []
        for x in remaining:
            new_sums = {x + y for y in curr} | {2 * x}
            conflict_count = len(new_sums & sums)
            scored.append((conflict_count, x))
        
        # Sort by conflict count, then randomize ties
        random.shuffle(scored)
        scored.sort(key=lambda t: t[0])
        
        added = 0
        for conflict_count, x in scored:
            if len(curr) >= max_size:
                break
            new_sums = {x + y for y in curr} | {2 * x}
            if not (new_sums & sums):
                curr.append(x)
                curr.sort()
                sums |= new_sums
                added += 1
                # Recompute scores for remaining? Too expensive. Just continue.
        
        return set(curr)

    def kick_and_expand(current_set, max_size=20):
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
        expanded = try_expand(new_set, max_size)
        return expanded

    def local_search(current_set, max_size=20, max_iterations=1000):
        """
        Local search: try swaps of non-base elements to enable adding more.
        """
        curr = set(current_set)
        best = set(curr)
        best_size = len(curr)
        
        for _ in range(max_iterations):
            non_base = [x for x in curr if x not in base_set]
            if not non_base:
                break
            
            # Pick a random non-base element to potentially swap
            x = random.choice(non_base)
            
            # Try replacing x with a different number
            candidates = [i for i in range(1, 285) if i not in curr and i != x]
            if not candidates:
                break
            
            y = random.choice(candidates)
            
            # Try the swap
            trial = curr - {x} | {y}
            if is_valid(trial):
                # Now try to expand
                expanded = try_expand(trial, max_size)
                if len(expanded) > best_size:
                    best = expanded
                    best_size = len(expanded)
                    curr = expanded
                elif len(expanded) == best_size and random.random() < 0.3:
                    # Accept same-size with some probability to escape local optima
                    curr = expanded
        
        return best

    # Main search loop
    best_size_found = len(base)
    best_sol_found = base.copy()
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Strategy 1: Randomized greedy expansion from base
        sol = try_expand(base_set)
        
        if len(sol) > best_size_found:
            best_size_found = len(sol)
            best_sol_found = sol.copy()
            if best_size_found == 20:
                return sorted(best_sol_found)
        
        # Strategy 2: Heuristic greedy expansion (conflict-aware)
        sol_h = try_expand_heuristic(base_set)
        
        if len(sol_h) > best_size_found:
            best_size_found = len(sol_h)
            best_sol_found = sol_h.copy()
            if best_size_found == 20:
                return sorted(best_sol_found)
        
        # Strategy 3: Kick and expand on best found
        if best_size_found > len(base):
            kicked = kick_and_expand(set(best_sol_found))
            if len(kicked) > best_size_found:
                best_size_found = len(kicked)
                best_sol_found = kicked.copy()
                if best_size_found == 20:
                    return sorted(best_sol_found)
        
        # Strategy 4: Local search on best found
        if best_size_found > len(base) and time.time() < deadline - 15:
            ls_result = local_search(set(best_sol_found), max_iterations=500)
            if len(ls_result) > best_size_found:
                best_size_found = len(ls_result)
                best_sol_found = ls_result.copy()
                if best_size_found == 20:
                    return sorted(best_sol_found)
        
        # Strategy 5: Multiple random kicks
        if best_size_found > len(base) and attempts % 5 == 0:
            for _ in range(5):
                kicked = kick_and_expand(set(best_sol_found))
                if len(kicked) > best_size_found:
                    best_size_found = len(kicked)
                    best_sol_found = kicked.copy()
                    if best_size_found == 20:
                        return sorted(best_sol_found)
        
        if time.time() > deadline - 5:
            break

    return sorted(best_sol_found)
