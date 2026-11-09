"""Simulated annealing search for truncated perfect difference sets (Sidon sets covering 1..30)."""
import time
import random
import math

def _validate_and_max(S):
    """Return (is_valid, max_element) for a sorted set S. Valid means Sidon + covers 1..30."""
    n = len(S)
    if n < 2:
        return False, 0
    seen = set()
    max_elem = S[-1]
    for i in range(n):
        si = S[i]
        for j in range(i + 1, n):
            d = S[j] - si
            if d > 30:
                continue
            if d in seen:
                return False, max_elem
            seen.add(d)
    # Check coverage of 1..30
    for d in range(1, 31):
        if d not in seen:
            return False, max_elem
    return True, max_elem

def _count_covered(S):
    """Count how many of 1..30 are covered as differences."""
    n = len(S)
    covered = set()
    for i in range(n):
        si = S[i]
        for j in range(i + 1, n):
            d = S[j] - si
            if 1 <= d <= 30:
                covered.add(d)
    return len(covered)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_set = None
    best_max = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Greedy initialization with multiple seeds
    def greedy_init(seed):
        r = random.Random(seed)
        S = [0]
        used_diffs = set()
        
        for _ in range(200):
            # Find which of 1..30 are still missing
            missing = [d for d in range(1, 31) if d not in used_diffs]
            if not missing:
                break
            
            target = min(missing)
            # Try to find x that covers target and maintains Sidon
            found = False
            attempts = 0
            while attempts < 2000 and not found:
                attempts += 1
                x = r.randint(1, 100)
                if x in S:
                    continue
                # Check Sidon
                valid = True
                new_diffs = []
                for s in S:
                    d = abs(x - s)
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if not valid:
                    continue
                # Check it covers target
                if target not in new_diffs:
                    continue
                # Accept
                S.append(x)
                S.sort()
                used_diffs.update(new_diffs)
                found = True
        
        # Verify
        valid, mx = _validate_and_max(S)
        if valid:
            return S, mx
        return None, float('inf')
    
    # Try multiple greedy seeds
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        S, mx = greedy_init(seed)
        if S is not None and mx < best_max:
            best_max = mx
            best_set = S[:]
    
    if best_set is None:
        # Fallback: simple greedy
        S = [0]
        used_diffs = set()
        for d in range(1, 31):
            if d in used_diffs:
                continue
            found = False
            for x in range(1, 500):
                if x in S:
                    continue
                valid = True
                new_diffs = []
                for s in S:
                    dd = abs(x - s)
                    if dd in used_diffs or dd in new_diffs:
                        valid = False
                        break
                    new_diffs.append(dd)
                if not valid:
                    continue
                if d in new_diffs:
                    S.append(x)
                    S.sort()
                    used_diffs.update(new_diffs)
                    found = True
                    break
            if not found:
                # Force add
                x = max(S) + d + 1
                while True:
                    valid = True
                    new_diffs = []
                    for s in S:
                        dd = abs(x - s)
                        if dd in used_diffs or dd in new_diffs:
                            valid = False
                            break
                        new_diffs.append(dd)
                    if valid:
                        break
                    x += 1
                S.append(x)
                S.sort()
                used_diffs.update([abs(x - s) for s in S])
        best_set = S
        best_max = max(S)
    
    # Phase 2: Simulated annealing to reduce max element
    def simulated_annealing(start_S, start_time_local, end_time_local, rng_local):
        nonlocal best_set, best_max
        
        S = start_S[:]
        S.sort()
        current_max = max(S)
        current_covered = _count_covered(S)
        
        # Track if currently valid
        cur_valid, _ = _validate_and_max(S)
        
        best_local = S[:]
        best_local_max = current_max
        
        T = 100.0
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() < end_time_local and T > T_min:
            T *= cooling
            S_set = set(S)
            max_elem = S[-1]
            
            # Generate move
            move_type = rng_local.randint(1, 10)
            new_S = S[:]
            
            if move_type <= 4:
                # Replace max element with smaller
                idx = len(new_S) - 1
                if len(new_S) > 2:
                    new_val = rng_local.randint(1, max_elem - 1)
                    if new_val in S_set:
                        continue
                    new_S[idx] = new_val
            elif move_type <= 7:
                # Replace a random non-zero element
                idx = rng_local.randint(1, len(new_S) - 1)
                old_val = new_S[idx]
                new_val = rng_local.randint(1, max_elem - 1)
                if new_val in S_set:
                    continue
                new_S[idx] = new_val
            elif move_type <= 9:
                # Add a new element
                new_val = rng_local.randint(1, max_elem)
                if new_val not in S_set:
                    new_S.append(new_val)
            else:
                # Remove a non-zero element (if we have more than needed)
                if len(new_S) > 8:
                    idx = rng_local.randint(1, len(new_S) - 1)
                    new_S.pop(idx)
            
            new_S.sort()
            
            # Validate
            new_valid, new_max_elem = _validate_and_max(new_S)
            
            if new_valid:
                if new_max_elem < best_local_max:
                    best_local_max = new_max_elem
                    best_local = new_S[:]
                    if best_local_max < best_max:
                        best_max = best_local_max
                        best_set = best_local[:]
                # Accept if valid and not too large
                if new_max_elem <= current_max + 5:
                    S = new_S
                    current_max = new_max_elem
                    cur_valid = True
            else:
                # Check coverage for acceptance
                new_covered = _count_covered(new_S)
                # Compute Sidon violations
                n = len(new_S)
                seen = set()
                sidon_ok = True
                for i in range(n):
                    si = new_S[i]
                    for j in range(i + 1, n):
                        d = new_S[j] - si
                        if d > 30:
                            continue
                        if d in seen:
                            sidon_ok = False
                            break
                        seen.add(d)
                    if not sidon_ok:
                        break
                
                # Accept with probability based on T
                if new_covered >= current_covered - 2 and sidon_ok:
                    delta_energy = max(0, new_max_elem - current_max) + max(0, current_covered - new_covered) * 3
                    if rng_local.random() < math.exp(-delta_energy / T):
                        S = new_S
                        current_max = new_max_elem
                        cur_valid = new_valid
                        current_covered = new_covered
        
        return best_local, best_local_max
    
    # Run SA from the best greedy solution
    if best_set is not None:
        sa_end = min(deadline - 2, start_time + 143.0)
        _, _ = simulated_annealing(best_set, time.time(), sa_end, rng)
    
    # Phase 3: Targeted local search - try to reduce max element
    if best_set is not None and time.time() < deadline - 3:
        S = best_set[:]
        S.sort()
        
        for _ in range(10000):
            if time.time() > deadline - 3:
                break
            
            max_elem = S[-1]
            S_set = set(S)
            improved = False
            
            # Try to replace max element with smaller values
            for new_val in range(max_elem - 1, 0, -1):
                if new_val in S_set:
                    continue
                trial = S[:-1] + [new_val]
                trial.sort()
                valid, mx = _validate_and_max(trial)
                if valid and mx < max_elem:
                    S = trial
                    best_set = S[:]
                    best_max = mx
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try other replacements
            for i in range(1, len(S)):
                old_val = S[i]
                for new_val in range(1, max_elem):
                    if new_val in S_set:
                        continue
                    trial = S[:i] + [new_val] + S[i+1:]
                    trial.sort()
                    valid, mx = _validate_and_max(trial)
                    if valid and mx < best_max:
                        S = trial
                        best_set = S[:]
                        best_max = mx
                        S_set = set(S)
                        max_elem = S[-1]
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try adding an element then removing
                for new_val in range(1, max_elem + 20):
                    if new_val in S_set:
                        continue
                    trial = S + [new_val]
                    trial.sort()
                    # Try removing each element
                    for rem in range(1, len(trial)):
                        trial2 = trial[:rem] + trial[rem+1:]
                        trial2.sort()
                        valid, mx = _validate_and_max(trial2)
                        if valid and mx < best_max:
                            S = trial2
                            best_set = S[:]
                            best_max = mx
                            S_set = set(S)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
                
                # Stuck - try random perturbation
                idx = rng.randint(1, len(S) - 1)
                S[idx] = rng.randint(1, max_elem + 10)
                S.sort()
                valid, mx = _validate_and_max(S)
                if not valid:
                    S = best_set[:]
                    S.sort()
    
    return best_set

# EVOLVE-BLOCK-END
