"""Sidon set extension in [1, 284] containing {1,2,5,11}."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    seed_elements = [1, 2, 5, 11]
    
    def get_sums(a):
        """Compute all pairwise sums a[i]+a[j] for i<=j."""
        s = set()
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                s.add(a[i] + a[j])
        return s
    
    def is_valid(a):
        """Check if a is a valid Sidon set (all pairwise sums distinct)."""
        return len(get_sums(a)) == len(a) * (len(a) + 1) // 2
    
    def greedy_extend(a, start, end, direction=1):
        """Greedy extension: try to add elements in order."""
        a = list(a)
        sums = get_sums(a)
        if direction == 1:
            candidates = range(start, end + 1)
        else:
            candidates = range(end, start - 1, -1)
        
        for x in candidates:
            if x in a:
                continue
            new_sums = {x + y for y in a} | {2 * x}
            if len(new_sums) == len(a) + 1 and not new_sums & sums:
                a.append(x)
                a.sort()
                sums |= new_sums
        return a
    
    def csp_backtrack(selected, sums, target_size, max_depth=50, deadline=None):
        """
        CSP-based backtracking with forward checking.
        selected: sorted list of selected elements
        sums: set of all pairwise sums
        target_size: desired size
        Returns a valid Sidon set of size >= target_size, or None if not found within budget.
        """
        if len(selected) >= target_size:
            return list(selected)
        
        if deadline and time.time() > deadline:
            return None
        
        # If we're close to target, try direct greedy completion
        remaining_needed = target_size - len(selected)
        
        # Find all candidates that can be added
        candidates = []
        for x in range(1, 285):
            if x in selected:
                continue
            # Check if x can be added
            new_sums = {x + y for y in selected} | {2 * x}
            if len(new_sums) == len(selected) + 1 and not new_sums & sums:
                candidates.append(x)
        
        if not candidates:
            return None
        
        # Dynamic variable ordering: choose candidate with fewest remaining valid partners
        # For each candidate, count how many other candidates it conflicts with
        def count_conflicts(cand, other_candidates):
            conflicts = 0
            cand_sums = {cand + y for y in selected} | {2 * cand}
            for other in other_candidates:
                if other == cand:
                    continue
                other_sums = {other + y for y in selected} | {2 * other}
                # Check if adding both would cause sum collision
                if cand_sums & other_sums:
                    conflicts += 1
            return conflicts
        
        # Sort candidates by number of conflicts (fewest first)
        candidates.sort(key=lambda x: count_conflicts(x, candidates))
        
        # Try candidates in order, with some randomization
        for x in candidates:
            if deadline and time.time() > deadline:
                return None
            
            new_selected = selected + [x]
            new_selected.sort()
            new_sums = sums | {x + y for y in selected} | {2 * x}
            
            result = csp_backtrack(new_selected, new_sums, target_size, max_depth - 1, deadline)
            if result is not None:
                return result
        
        return None
    
    def local_search(a, max_iters=5000, deadline=None):
        """Local search: try swapping elements to improve size or enable more additions."""
        a = sorted(a)
        best = list(a)
        best_size = len(a)
        sums = get_sums(a)
        
        # First, try greedy additions
        for x in range(1, 285):
            if x in a:
                continue
            new_sums = {x + y for y in a} | {2 * x}
            if len(new_sums) == len(a) + 1 and not new_sums & sums:
                a.append(x)
                a.sort()
                sums |= new_sums
                if len(a) > best_size:
                    best = list(a)
                    best_size = len(a)
        
        # Now try local search: remove one element, add another
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
            
            improved = False
            # Try removing a random element and adding a different one
            if len(a) > len(seed_elements):
                idx = random.randint(0, len(a) - 1)
                removed = a[idx]
                a_new = a[:idx] + a[idx+1:]
                a_new.sort()
                sums_new = get_sums(a_new)
                
                # Try to add a random element
                for _ in range(20):
                    x = random.randint(1, 284)
                    if x in a_new:
                        continue
                    ns = {x + y for y in a_new} | {2 * x}
                    if len(ns) == len(a_new) + 1 and not ns & sums_new:
                        a_new.append(x)
                        a_new.sort()
                        sums_new |= ns
                        # Try greedy extension
                        a_new = greedy_extend(a_new, 1, 284, 1)
                        if len(a_new) > len(a):
                            a = a_new
                            sums = sums_new
                            improved = True
                            if len(a) > best_size:
                                best = list(a)
                                best_size = len(a)
                            break
            
            if not improved and random.random() < 0.1:
                # Occasionally restart with fresh greedy
                pass
        
        return best
    
    # Strategy 1: Simple greedy from seed
    best_result = greedy_extend(seed_elements, 12, 284, 1)
    
    # Strategy 2: Greedy from different starting points
    for start in range(12, 285, 10):
        a = greedy_extend(seed_elements, start, 284, 1)
        if len(a) > len(best_result):
            best_result = a
    
    # Strategy 3: Randomized greedy with multiple restarts
    random.seed(42)
    num_restarts = 50
    per_restart_time = 2.0
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Randomize the order of candidates
        a = list(seed_elements)
        sums = get_sums(a)
        
        # Build a random priority list
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        # Also try some structured orders
        if r % 3 == 0:
            candidates = list(range(12, 285))
        elif r % 3 == 1:
            candidates = list(range(284, 11, -1))
        # else: already shuffled
        
        for x in candidates:
            if x in a:
                continue
            new_sums = {x + y for y in a} | {2 * x}
            if len(new_sums) == len(a) + 1 and not new_sums & sums:
                a.append(x)
                a.sort()
                sums |= new_sums
        
        if len(a) > len(best_result):
            best_result = a
        
        # Local search improvement
        if len(a) >= len(best_result):
            a_improved = local_search(a, max_iters=2000, deadline=deadline - 5)
            if len(a_improved) > len(best_result):
                best_result = a_improved
    
    # Strategy 4: CSP-based backtracking to reach size 19-20
    for target_size in [20, 19]:
        if len(best_result) >= target_size:
            break
        if time.time() > deadline - 15:
            break
        
        # Use CSP backtracking with the current best as starting point
        csp_result = csp_backtrack(
            list(best_result), 
            get_sums(best_result), 
            target_size, 
            max_depth=30, 
            deadline=deadline - 5
        )
        
        if csp_result is not None and len(csp_result) > len(best_result):
            best_result = csp_result
    
    # Strategy 5: Try to extend best_result further
    a = list(best_result)
    a = greedy_extend(a, 1, 284, 1)
    if len(a) > len(best_result):
        best_result = a
    
    # Final verification
    final_a = sorted(best_result)
    assert all(x in final_a for x in seed_elements), "Missing seed elements"
    assert all(1 <= x <= 284 for x in final_a), "Out of range"
    assert len(set(final_a)) == len(final_a), "Duplicates"
    assert is_valid(final_a), "Invalid Sidon set"
    
    return final_a
# EVOLVE-BLOCK-END
