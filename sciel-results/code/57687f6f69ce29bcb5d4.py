import time
import random
import sys

def _is_sidon(set_a):
    """Check if all positive differences are distinct."""
    n = len(set_a)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = set_a[j] - set_a[i]
            if d in seen:
                return False
            seen.add(d)
    return True

def _has_all_diffs(set_a, max_d=30):
    """Check if all d in 1..max_d occur as differences."""
    diffs = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            d = set_a[j] - set_a[i]
            diffs.add(d)
    return all(d in diffs for d in range(1, max_d + 1))

def _score(set_a):
    """Score: minimize max element. Return float('inf') if invalid."""
    if not _is_sidon(set_a):
        return float('inf')
    if not _has_all_diffs(set_a):
        return float('inf')
    return max(set_a)

def _solve_backtrack(seed=0, time_limit=10.0):
    """
    CSP solver with randomized backtracking and forward checking.
    State: current set of elements, set of used differences.
    Goal: Include all differences 1..30 while maintaining Sidon property.
    Minimize: max element.
    """
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initial state
    current_set = [0]
    used_diffs = set()
    
    # We want to cover 1..30.
    # To minimize max element, we should try to cover small differences first?
    # Or rather, the structure of the set determines the max.
    # A good heuristic is to try to add elements that cover the most "urgent" missing differences.
    # However, a simpler CSP approach:
    # Variables: The set of elements.
    # Constraints: Sidon, Covers 1..30.
    
    # Let's use a recursive backtracking search where we decide which element to add next.
    # To ensure we find a small max, we can limit the search space to [0, current_bound].
    # We can iteratively increase the bound.
    
    # Start with a small bound, say 100, and increase if no solution found.
    # But we don't know the optimal bound.
    # Instead, let's search for any valid set, then try to optimize?
    # No, the backtracking itself can be guided by the max element.
    
    # Strategy:
    # 1. Maintain a list of missing differences.
    # 2. Pick a missing difference d.
    # 3. Generate candidate pairs (x, y) such that y - x = d.
    # 4. Try adding x and y (if not already present) to the set.
    # 5. Check Sidon property.
    # 6. Recurse.
    
    # This is a bit complex because adding one element might satisfy multiple differences.
    # A better state for CSP:
    # Current set S.
    # Used diffs D.
    # Missing diffs M = {1..30} \ D.
    
    # If M is empty, we have a solution.
    # If time is up, return None.
    
    # Heuristic for choosing next missing diff:
    # Pick the smallest missing diff? Or the one with fewest candidate pairs?
    # Let's try picking the smallest missing diff first, as it constrains the local structure more.
    
    # Heuristic for choosing candidate pair (x, y) for diff d:
    # We want to keep the max element small.
    # So prefer x, y that are small.
    # Also, x and y must not create duplicate differences.
    
    # To avoid infinite loops and excessive branching, we limit the number of elements.
    # Max size 32.
    
    best_solution = None
    best_max = float('inf')
    
    def backtrack(current_set, used_diffs, depth):
        nonlocal best_solution, best_max
        
        if time.time() > deadline:
            return False
            
        current_set = sorted(current_set)
        max_elem = max(current_set)
        
        # Check if we have a valid solution
        missing = [d for d in range(1, 31) if d not in used_diffs]
        if not missing:
            if max_elem < best_max:
                best_max = max_elem
                best_solution = list(current_set)
            return True # Found a solution, can stop this branch? 
            # Actually, we want the BEST solution, so we should continue searching for smaller max?
            # But backtracking usually returns on first solution. 
            # To find the minimum max, we need to search more or use branch and bound.
            # Given time limits, finding ANY small solution is good. 
            # We can rely on multiple restarts with different seeds/strategies.
            # For now, let's accept the first valid solution found in this run, 
            # but we will run multiple seeds.
            return True
            
        if len(current_set) >= 32:
            return False
            
        # Choose a missing difference to satisfy
        # Heuristic: pick the one with the fewest valid candidate pairs
        # Or just the smallest one to keep numbers small?
        # Let's try the smallest missing difference.
        d_target = missing[0]
        
        # Generate candidates for d_target
        # We need x, y in current_set U {new} such that y - x = d_target.
        # Cases:
        # 1. x in current_set, y not in current_set. y = x + d_target.
        # 2. y in current_set, x not in current_set. x = y - d_target.
        # 3. Neither in current_set. x, y are new.
        
        candidates = []
        
        # Case 1: Add y
        for x in current_set:
            y = x + d_target
            if y not in current_set:
                # Check if adding y is valid (Sidon)
                # We need to check all new differences involving y
                # New diffs: y - z for z in current_set
                # And ensure they are not in used_diffs
                valid = True
                for z in current_set:
                    diff = y - z
                    if diff in used_diffs:
                        valid = False
                        break
                if valid:
                    # Also check if y creates a diff with another new element? No, only adding one.
                    candidates.append((y, 1)) # (value, num_new_elements)
                    
        # Case 2: Add x
        for y in current_set:
            x = y - d_target
            if x >= 0 and x not in current_set:
                valid = True
                for z in current_set:
                    diff = z - x
                    if diff in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append((x, 1))
                    
        # Case 3: Add x and y (both new)
        # This is more expensive. Let's limit this to small values.
        # x can be any non-negative integer not in current_set.
        # y = x + d_target.
        # To keep max small, try small x.
        max_x_to_try = max(current_set) + d_target + 10 # Heuristic limit
        for x in range(0, max_x_to_try):
            if x in current_set:
                continue
            y = x + d_target
            if y in current_set:
                continue
            
            # Check Sidon for adding x and y
            # New diffs:
            # y - x = d_target (already known to be missing, so ok)
            # y - z for z in current_set
            # z - x for z in current_set
            
            valid = True
            new_diffs = {d_target}
            
            for z in current_set:
                d1 = y - z
                if d1 in used_diffs or d1 in new_diffs:
                    valid = False
                    break
                new_diffs.add(d1)
                
                d2 = z - x
                if d2 in used_diffs or d2 in new_diffs:
                    valid = False
                    break
                new_diffs.add(d2)
            
            if valid:
                # Also check if x and y create duplicate diffs between themselves?
                # y - x = d_target. That's the only pair.
                candidates.append((x, 2)) # Mark as 2 new elements? 
                # Actually, we need to store both x and y.
                # Let's store tuples (values_list, count)
                # Re-structure candidates to be (list_of_new_vals)
                
        # Re-do candidate generation to store lists
        candidates = []
        
        for x in current_set:
            y = x + d_target
            if y not in current_set:
                valid = True
                for z in current_set:
                    diff = y - z
                    if diff in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append([y])
                    
        for y in current_set:
            x = y - d_target
            if x >= 0 and x not in current_set:
                valid = True
                for z in current_set:
                    diff = z - x
                    if diff in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append([x])
                    
        # Case 3: Add x, y
        # To avoid explosion, limit x
        # We can also randomize the order of x
        # Let's try a few small x values
        x_candidates = list(range(0, max(current_set) + d_target + 20))
        rng.shuffle(x_candidates)
        
        for x in x_candidates:
            if x in current_set:
                continue
            y = x + d_target
            if y in current_set:
                continue
            
            valid = True
            new_diffs = {d_target}
            
            for z in current_set:
                d1 = y - z
                if d1 in used_diffs or d1 in new_diffs:
                    valid = False
                    break
                new_diffs.add(d1)
                
                d2 = z - x
                if d2 in used_diffs or d2 in new_diffs:
                    valid = False
                    break
                new_diffs.add(d2)
            
            if valid:
                candidates.append([x, y])
                
        if not candidates:
            return False
            
        # Heuristic: sort candidates by max(new_vals) to prefer smaller growth
        candidates.sort(key=lambda c: max(c))
        
        # Randomize slightly to escape local optima?
        # For CSP, deterministic is often better if the heuristic is good.
        # But we have multiple seeds, so let's shuffle a bit.
        if rng.random() < 0.3:
            # Shuffle the first 10 candidates
            k = min(10, len(candidates))
            for i in range(k):
                j = rng.randint(i, k-1)
                candidates[i], candidates[j] = candidates[j], candidates[i]
                
        for new_vals in candidates:
            # Apply move
            new_set = current_set + new_vals
            new_used_diffs = set(used_diffs)
            
            # Update used_diffs
            # We already checked validity, so we just need to add the diffs
            # Actually, we should re-calculate or add specifically.
            # For efficiency, add the diffs involving new elements.
            
            # Sort new_set for consistent diff calculation?
            # The diffs are just pairwise.
            for v in new_vals:
                for u in current_set:
                    diff = abs(v - u)
                    new_used_diffs.add(diff)
            for i in range(len(new_vals)):
                for j in range(i+1, len(new_vals)):
                    diff = abs(new_vals[i] - new_vals[j])
                    new_used_diffs.add(diff)
                    
            # Recurse
            if backtrack(new_set, new_used_diffs, depth + 1):
                return True
                
        return False
        
    # Start backtracking
    # To improve success rate, we can try different initial bounds or heuristics.
    # For now, just run it.
    
    backtrack([0], set(), 0)
    
    if best_solution:
        return best_solution
    return None

def _greedy_construct(seed=0):
    """Fast greedy construction to get a valid starting point."""
    rng = random.Random(seed)
    a = [0]
    
    def diffs_of(s):
        s_sorted = sorted(s)
        return [y - x for i, x in enumerate(s_sorted) for y in s_sorted[i + 1:]]
    
    for d in range(1, 31):
        if d in diffs_of(a):
            continue
        # Try adding one element
        candidates = set()
        for v in a:
            candidates.add(v + d)
            candidates.add(v - d)
        candidates = [x for x in candidates if x >= 0 and x not in a]
        rng.shuffle(candidates)
        found = False
        for x in candidates:
            trial = sorted(a + [x])
            if _is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            # Try adding two elements
            for _ in range(100):
                y = max(a) + rng.randint(1, 50)
                trial = sorted(a + [y, y + d])
                if _is_sidon(trial):
                    a = trial
                    found = True
                    break
            if not found:
                # Fallback: large gap
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if _is_sidon(trial):
                        a = trial
                        break
                    y += 1
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 118.0 # Safety margin
    
    best_set = None
    best_score = float('inf')
    
    # Run multiple CSP instances with different seeds
    # Each run has a time limit to allow restarts.
    
    seed = 0
    while time.time() < deadline - 5.0:
        run_deadline = min(deadline - 2.0, time.time() + 5.0)
        time_limit = run_deadline - time.time()
        
        if time_limit < 0.5:
            break
            
        candidate = _solve_backtrack(seed, time_limit=time_limit)
        
        if candidate:
            s = _score(candidate)
            if s < best_score:
                best_score = s
                best_set = candidate
                # If we found a very good solution, we might stop early?
                # But let's keep searching for better.
                
        seed += 1
        
    # If CSP didn't find anything, fallback to greedy
    if best_set is None or best_score == float('inf'):
        best_set = _greedy_construct(0)
        
    return best_set

# EVOLVE-BLOCK-END
