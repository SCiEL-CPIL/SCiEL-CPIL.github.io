import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin before 150s kill
    
    TARGETS = set(range(1, 31))
    N_TARGETS = 30
    
    def _diffs_set(s):
        """Return set of all positive differences of sorted list s."""
        n = len(s)
        d = set()
        for i in range(n):
            si = s[i]
            for j in range(i + 1, n):
                d.add(s[j] - si)
        return d

    def _is_sidon(s):
        """True iff all positive differences are distinct."""
        n = len(s)
        seen = set()
        for i in range(n):
            si = s[i]
            for j in range(i + 1, n):
                d = s[j] - si
                if d in seen:
                    return False
                seen.add(d)
        return True

    def _covers_1_to_30(s):
        """Check if all differences 1..30 are present."""
        diffs = _diffs_set(s)
        return TARGETS.issubset(diffs)

    def _valid(s):
        """Check full validity."""
        if 0 not in s:
            return False
        if len(s) > 32:
            return False
        if max(s) > 10**6:
            return False
        if not _is_sidon(s):
            return False
        if not _covers_1_to_30(s):
            return False
        return True

    def _backtrack_search(max_elem_bound, time_limit, order=None):
        """
        Backtracking search with forward checking.
        Try to build a Sidon set containing 0 that covers 1..30,
        with max element <= max_elem_bound.
        Returns solution or None.
        """
        if order is None:
            order = list(range(1, 31))
        
        # We'll build the set incrementally
        # State: current sorted list, set of covered differences, set of all differences
        best_solution = [None]
        
        def search(current, covered_diffs, all_diffs, depth, max_depth):
            if time.time() > time_limit:
                return False
            
            # Check if we've covered all targets
            if covered_diffs >= TARGETS:
                if max(current) < (best_solution[0][0] if best_solution[0] else 10**9):
                    best_solution[0] = list(current)
                return True
            
            # If we've used up elements or exceeded bound, fail
            if len(current) >= 32 or max(current) >= max_elem_bound:
                return False
            
            # Forward checking: can we still cover remaining targets?
            remaining = TARGETS - covered_diffs
            if not remaining:
                return True
            
            # Determine which differences are still achievable
            # For each remaining target d, we need two elements a,b in current∪future with |a-b|=d
            # This is complex; use a simpler heuristic: if we can't add any element that creates
            # a new covered difference, we might be stuck
            
            # Try adding a new element
            # Candidate values: for each uncovered difference d, try x such that x - c = d or c - x = d
            # for some c in current. Also try random small values.
            
            candidates = set()
            for d in remaining:
                for c in current:
                    v1 = c + d
                    v2 = c - d
                    if 0 < v1 < max_elem_bound:
                        candidates.add(v1)
                    if 0 < v2 < max_elem_bound:
                        candidates.add(v2)
            
            # Also add some random candidates for diversity
            for _ in range(5):
                v = random.randint(1, min(max_elem_bound, 200))
                if v not in current:
                    candidates.add(v)
            
            # Sort candidates by value (prefer smaller)
            for cand in sorted(candidates):
                if cand in current:
                    continue
                
                # Check if adding cand maintains Sidon property
                trial = sorted(current + [cand])
                new_diffs = set()
                valid = True
                for i in range(len(trial)):
                    ti = trial[i]
                    for j in range(i + 1, len(trial)):
                        d = trial[j] - ti
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                new_covered = covered_diffs | (new_diffs & TARGETS)
                
                # Pruning: if this doesn't help cover any new target and we're deep, skip
                if len(new_covered) == len(covered_diffs) and depth > 5:
                    continue
                
                if search(trial, new_covered, new_diffs, depth + 1, max_depth):
                    return True
            
            return False
        
        # Start with [0]
        initial_diffs = set()
        search([0], set(), initial_diffs, 0, 32)
        
        return best_solution[0]

    def _local_search(initial, time_limit):
        """Local search to improve the maximum element."""
        best = list(initial)
        best_max = max(best)
        current = list(best)
        current_max = max(current)
        
        T = 100.0
        T_min = 1.0
        cooling = 0.9995
        
        def _energy(s):
            if not _is_sidon(s):
                # Count collisions
                diffs = _diffs_set(s)
                n = len(s)
                collisions = n * (n - 1) // 2 - len(diffs)
                return max(s) + 100000 * collisions
            missing = len(TARGETS - _diffs_set(s))
            return max(s) + 100000 * missing
        
        best_e = _energy(best)
        current_e = _energy(current)
        
        while time.time() < time_limit:
            T = max(T_min, T * cooling)
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Replace an element with a smaller value
                if len(current) <= 1:
                    continue
                idx = random.randint(1, len(current) - 1)
                old_val = current[idx]
                if old_val <= 1:
                    continue
                # Try a smaller random value
                new_val = random.randint(1, old_val - 1)
                if new_val in current:
                    continue
                trial = current.copy()
                trial[idx] = new_val
                trial = sorted(set(trial))
                if len(trial) != len(current):
                    continue
                trial_e = _energy(trial)
                delta = trial_e - current_e
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = trial
                    current_e = trial_e
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
                        best_max = max(best)
            elif move_type < 0.7:
                # Remove an element
                if len(current) <= 8:
                    continue
                idx = random.randint(1, len(current) - 1)
                trial = current[:idx] + current[idx + 1:]
                trial_e = _energy(trial)
                delta = trial_e - current_e
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = trial
                    current_e = trial_e
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
                        best_max = max(best)
            elif move_type < 0.9:
                # Add an element
                if len(current) >= 32:
                    continue
                max_so_far = max(current)
                new_val = random.randint(1, max_so_far + 20)
                if new_val in current:
                    continue
                trial = sorted(current + [new_val])
                trial_e = _energy(trial)
                delta = trial_e - current_e
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = trial
                    current_e = trial_e
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
                        best_max = max(best)
            else:
                # Perturbation: replace multiple elements
                if len(current) <= 2:
                    continue
                n_replace = random.randint(1, min(3, len(current) - 1))
                indices = random.sample(range(1, len(current)), n_replace)
                trial = current.copy()
                for idx in indices:
                    old_val = trial[idx]
                    new_val = random.randint(1, max(1, old_val + 10))
                    if new_val not in trial or new_val == old_val:
                        continue
                    trial[idx] = new_val
                trial = sorted(set(trial))
                if len(trial) != len(current):
                    continue
                trial_e = _energy(trial)
                delta = trial_e - current_e
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = trial
                    current_e = trial_e
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
                        best_max = max(best)
        
        return best

    random.seed(42)
    
    # Phase 1: Backtracking search with different orderings
    best_initial = None
    best_initial_max = 10**9
    
    # Try several different orderings
    for attempt in range(10):
        if time.time() > DEADLINE - 30:
            break
        
        if attempt == 0:
            order = list(range(1, 31))
        elif attempt == 1:
            order = list(range(30, 0, -1))
        elif attempt == 2:
            # Alternating
            order = []
            for i in range(1, 31):
                if i % 2 == 1:
                    order.append(i)
            for i in range(1, 31):
                if i % 2 == 0:
                    order.append(i)
        else:
            order = list(range(1, 31))
            random.shuffle(order)
        
        # Start with a reasonable bound and tighten
        bound = 500
        for _ in range(5):
            if time.time() > DEADLINE - 30:
                break
            
            sol = _backtrack_search(bound, DEADLINE - 35, order)
            if sol is not None:
                if max(sol) < best_initial_max:
                    best_initial = sol
                    best_initial_max = max(sol)
                break
            bound = max(100, bound // 2)
        
        if best_initial is not None and best_initial_max <= 100:
            break
    
    # Fallback: if backtracking didn't find anything, use a simple greedy
    if best_initial is None:
        a = [0]
        current_diffs = set()
        for d in range(1, 31):
            if d in current_diffs:
                continue
            found = False
            candidates = set()
            for x in a:
                candidates.add(x + d)
                if x - d > 0:
                    candidates.add(x - d)
            for cand in sorted(candidates):
                if cand in a:
                    continue
                trial = sorted(a + [cand])
                if _is_sidon(trial):
                    new_diffs = _diffs_set(trial)
                    if d in new_diffs:
                        a = trial
                        current_diffs = new_diffs
                        found = True
                        break
            if not found:
                y = 2 * max(a) + d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if _is_sidon(trial):
                        a = trial
                        current_diffs = _diffs_set(a)
                        break
                    y += 1
        best_initial = a
        best_initial_max = max(a)
    
    # Phase 2: Local search to refine
    time_limit = DEADLINE - 5.0
    result = _local_search(best_initial, time_limit)
    
    # Ensure validity
    if not _valid(result):
        result = best_initial
    
    result = sorted(set(result))
    if 0 not in result:
        result = sorted([0] + result)
    
    return result
