import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    TARGETS = set(range(1, 31))
    
    def get_diffs_set(s):
        """Get set of positive differences from sorted list s."""
        diffs = set()
        n = len(s)
        for i in range(n):
            si = s[i]
            for j in range(i+1, n):
                diffs.add(s[j] - si)
        return diffs

    def is_valid(s):
        """Check if s is a valid solution."""
        if 0 not in s:
            return False
        if len(s) > 32:
            return False
        if max(s) > 10**6:
            return False
        diffs = get_diffs_set(s)
        # Sidon condition: number of pairs must equal number of unique diffs
        n = len(s)
        if len(diffs) != n * (n - 1) // 2:
            return False
        if not TARGETS.issubset(diffs):
            return False
        return True

    def energy(s):
        """
        Energy function: 
        1. Max element (primary objective)
        2. Missing targets (hard penalty)
        3. Collisions (hard penalty)
        """
        diffs = get_diffs_set(s)
        n = len(s)
        collisions = n * (n - 1) // 2 - len(diffs)
        missing = len(TARGETS - diffs)
        max_elem = max(s)
        # We want to minimize max_elem, but only if valid.
        # If invalid, penalize heavily.
        e = max_elem + 100000 * (missing + collisions)
        return e

    # Phase 1: Greedy Construction
    def greedy_construct():
        a = [0]
        current_diffs = set()
        for d in range(1, 31):
            if d in current_diffs:
                continue
            found = False
            # Try to add x such that x - a_i = d or a_i - x = d
            # This implies x = a_i + d or x = a_i - d
            candidates = set()
            for x in a:
                candidates.add(x + d)
                if x - d > 0: # x-d > 0 to keep positive, though 0 is already in a
                    candidates.add(x - d)
            
            # Sort candidates to try smaller ones first
            for cand in sorted(candidates):
                if cand in a:
                    continue
                trial = sorted(a + [cand])
                # Check Sidon condition for trial
                # Optimization: just check new differences involving cand
                # But for simplicity and safety with small N, recompute
                if is_sidon(trial):
                    # Check if it actually adds d
                    new_diffs = set()
                    for i in range(len(trial)):
                        for j in range(i+1, len(trial)):
                            new_diffs.add(trial[j] - trial[i])
                    if d in new_diffs:
                        a = trial
                        current_diffs = new_diffs
                        found = True
                        break
            if not found:
                # Fallback: add a large element to create d
                # Try y such that y - 0 = d is already covered if d not in diffs?
                # If d not in diffs, we need to create it.
                # Add y = max(a) + d + 1?
                # Or just pick a random large number and hope?
                # Better: add two elements y, y+d
                y = 2 * max(a) + d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if is_sidon(trial):
                        a = trial
                        new_diffs = set()
                        for i in range(len(a)):
                            for j in range(i+1, len(a)):
                                new_diffs.add(a[j] - a[i])
                        current_diffs = new_diffs
                        break
                    y += 1
        return a

    def is_sidon(s):
        n = len(s)
        diffs = set()
        for i in range(n):
            si = s[i]
            for j in range(i+1, n):
                d = s[j] - si
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Phase 2: Local Search
    def local_search(initial, time_limit):
        best = list(initial)
        best_e = energy(best)
        current = list(best)
        current_e = best_e
        T = 500.0
        T_min = 1.0
        cooling = 0.999
        
        while time.time() < time_limit:
            T = max(T_min, T * cooling)
            
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.6:
                # Replace an element
                if len(current) <= 1:
                    continue
                idx = random.randint(1, len(current) - 1)
                old_val = current[idx]
                # Try smaller value
                new_val = random.randint(1, old_val) if old_val > 1 else 1
                if new_val in current:
                    continue
                trial = current.copy()
                trial[idx] = new_val
                trial = sorted(set(trial))
                if len(trial) != len(current):
                    continue
                trial_e = energy(trial)
                delta = trial_e - current_e
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = trial
                    current_e = trial_e
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
            elif move_type < 0.8:
                # Remove an element (if valid and not losing coverage too much? 
                # Actually, removing elements might lose coverage. 
                # Only remove if we can re-add smaller ones later?)
                # For now, skip removal to maintain coverage, or allow it with high penalty
                if len(current) <= 8:
                    continue
                idx = random.randint(1, len(current) - 1)
                trial = current[:idx] + current[idx+1:]
                trial_e = energy(trial)
                delta = trial_e - current_e
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = trial
                    current_e = trial_e
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
            else:
                # Add an element
                if len(current) >= 32:
                    continue
                max_so_far = max(current)
                new_val = random.randint(1, max_so_far + 10)
                if new_val in current:
                    continue
                trial = sorted(current + [new_val])
                trial_e = energy(trial)
                delta = trial_e - current_e
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current = trial
                    current_e = trial_e
                    if current_e < best_e:
                        best = list(current)
                        best_e = current_e
            
            # Check time
            if time.time() > time_limit:
                break

        return best

    random.seed(42)
    
    # Get initial solution
    initial = greedy_construct()
    
    if not is_valid(initial):
        # Try a few more greedy attempts with randomization
        for _ in range(5):
            random.seed(random.randint(0, 10000))
            # Shuffle target order?
            targets = list(range(1, 31))
            random.shuffle(targets)
            a = [0]
            current_diffs = set()
            for d in targets:
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
                    if is_sidon(trial):
                        new_diffs = set()
                        for i in range(len(trial)):
                            for j in range(i+1, len(trial)):
                                new_diffs.add(trial[j] - trial[i])
                        if d in new_diffs:
                            a = trial
                            current_diffs = new_diffs
                            found = True
                            break
                if not found:
                    y = 2 * max(a) + d + 1
                    while True:
                        trial = sorted(a + [y, y + d])
                        if is_sidon(trial):
                            a = trial
                            new_diffs = set()
                            for i in range(len(a)):
                                for j in range(i+1, len(a)):
                                    new_diffs.add(a[j] - a[i])
                            current_diffs = new_diffs
                            break
                        y += 1
            if is_valid(a):
                initial = a
                break

    time_limit = deadline - 2.0
    result = local_search(initial, time_limit)
    
    if not is_valid(result):
        result = initial

    result = sorted(set(result))
    if 0 not in result:
        result = sorted([0] + result)
        
    return result
