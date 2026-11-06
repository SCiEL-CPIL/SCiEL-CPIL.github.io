"""B2[2] set construction in [1, 1000] using simulated annealing over binary masks with penalty."""
import time
import random
import math

MAX_VAL = 1000
MAX_SUM = 2 * MAX_VAL  # 2000

def _greedy_build():
    """Standard greedy: add x if it doesn't cause any sum to exceed 2 reps."""
    chosen = []
    counts = [0] * (MAX_SUM + 1)
    for x in range(1, MAX_VAL + 1):
        ok = True
        for y in chosen:
            s = x + y
            if counts[s] >= 2:
                ok = False
                break
        if ok:
            s = 2 * x
            if counts[s] >= 2:
                ok = False
        if ok:
            for y in chosen:
                counts[x + y] += 1
            counts[2 * x] += 1
            chosen.append(x)
    return chosen

def _build_counts(set_list):
    """Build count array for a set."""
    counts = [0] * (MAX_SUM + 1)
    n = len(set_list)
    for i in range(n):
        a = set_list[i]
        for j in range(i, n):
            b = set_list[j]
            counts[a + b] += 1
    return counts

def _penalty(counts):
    """Penalty = sum over all sums of max(0, count[s] - 2)."""
    p = 0
    for c in counts:
        if c > 2:
            p += c - 2
    return p

def _simulated_annealing(initial, deadline, seed=42):
    """SA over binary mask with penalty function.
    State: set of selected elements (represented as list).
    Objective: maximize |S| - lambda * penalty(S), where penalty counts violations.
    Moves: add one element, remove one element, or swap one element.
    """
    rng = random.Random(seed)
    current = list(initial)
    current_set = set(current)
    counts = _build_counts(current)
    current_penalty = _penalty(counts)
    current_obj = len(current) - 0.5 * current_penalty  # lambda = 0.5
    
    best = list(current)
    best_size = len(current)
    best_penalty = current_penalty
    
    all_vals = list(range(1, MAX_VAL + 1))
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.01
    total_iters = 50000  # Will adjust based on time
    # We'll run until deadline, adjusting temperature geometrically
    
    start_time = time.time()
    elapsed = 0
    T = T_start
    cooling_rate = 0.9995  # Per iteration
    
    while time.time() < deadline:
        # Check time every 1000 iters to avoid overhead
        if rng.random() < 0.001:
            remaining_frac = max(0, (deadline - time.time()) / (deadline - start_time))
            # Adjust cooling rate to hit T_end at deadline
            # For now, use fixed cooling rate
        
        op = rng.random()
        
        if op < 0.4:
            # Add an element
            x = rng.choice(all_vals)
            if x in current_set:
                continue
            # Compute delta penalty
            delta_p = 0
            for y in current:
                s = x + y
                old_c = counts[s]
                new_c = old_c + 1
                delta_p += max(0, new_c - 2) - max(0, old_c - 2)
            s = 2 * x
            old_c = counts[s]
            new_c = old_c + 1
            delta_p += max(0, new_c - 2) - max(0, old_c - 2)
            
            # Objective change: +1 for size, -0.5 * delta_p for penalty
            delta_obj = 1.0 - 0.5 * delta_p
            
            if delta_obj >= 0 or rng.random() < math.exp(delta_obj / T):
                # Accept: update counts
                for y in current:
                    counts[x + y] += 1
                counts[2 * x] += 1
                current.append(x)
                current_set.add(x)
                current_penalty += delta_p
                current_obj += delta_obj
                
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                    best_penalty = current_penalty
        elif op < 0.7:
            # Remove an element
            if not current:
                continue
            x = rng.choice(current)
            # Compute delta penalty (removing reduces counts)
            delta_p = 0
            for y in current:
                if y == x:
                    continue
                s = x + y
                old_c = counts[s]
                new_c = old_c - 1
                delta_p += max(0, new_c - 2) - max(0, old_c - 2)
            s = 2 * x
            old_c = counts[s]
            new_c = old_c - 1
            delta_p += max(0, new_c - 2) - max(0, old_c - 2)
            
            # Objective change: -1 for size, -0.5 * delta_p for penalty (delta_p is negative or zero)
            delta_obj = -1.0 - 0.5 * delta_p
            
            if delta_obj >= 0 or rng.random() < math.exp(delta_obj / T):
                # Accept: update counts
                for y in current:
                    if y != x:
                        counts[x + y] -= 1
                counts[2 * x] -= 1
                current.remove(x)
                current_set.discard(x)
                current_penalty += delta_p
                current_obj += delta_obj
                
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                    best_penalty = current_penalty
        else:
            # Swap: remove one, add one
            if not current:
                continue
            remove_x = rng.choice(current)
            add_x = rng.choice(all_vals)
            if add_x == remove_x or add_x in current_set:
                continue
            
            # Compute combined delta penalty
            # First remove remove_x, then add add_x
            delta_p = 0
            # Remove remove_x
            for y in current:
                if y == remove_x:
                    continue
                s = remove_x + y
                old_c = counts[s]
                new_c = old_c - 1
                delta_p += max(0, new_c - 2) - max(0, old_c - 2)
            s = 2 * remove_x
            old_c = counts[s]
            new_c = old_c - 1
            delta_p += max(0, new_c - 2) - max(0, old_c - 2)
            
            # Now add add_x to the set without remove_x
            for y in current:
                if y == remove_x:
                    continue
                s = add_x + y
                # Need to use the updated count after removal
                # This is tricky; let's approximate by computing from scratch for these sums
                # Actually, let's just compute the net effect more carefully
                pass
            
            # For simplicity, let's just try the swap and check if it's valid
            # This is slower but correct
            temp_counts = list(counts)
            for y in current:
                if y != remove_x:
                    temp_counts[remove_x + y] -= 1
            temp_counts[2 * remove_x] -= 1
            
            ok = True
            for y in current:
                if y == remove_x:
                    continue
                s = add_x + y
                if temp_counts[s] >= 2:
                    ok = False
                    break
            if ok and temp_counts[2 * add_x] >= 2:
                ok = False
            
            if ok:
                # Valid swap, always accept
                for y in current:
                    if y != remove_x:
                        counts[remove_x + y] -= 1
                counts[2 * remove_x] -= 1
                for y in current:
                    if y != remove_x:
                        counts[add_x + y] += 1
                counts[2 * add_x] += 1
                current.remove(remove_x)
                current_set.discard(remove_x)
                current.append(add_x)
                current_set.add(add_x)
                # Recompute penalty for best tracking
                # This is expensive; skip for now
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
    
    return best

def _fast_is_valid(set_list):
    """Faster validity check using counting array."""
    n = len(set_list)
    if n == 0:
        return True
    s = sorted(set_list)
    counts = [0] * (MAX_SUM + 1)
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            counts[sm] += 1
            if counts[sm] > 2:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Step 1: Greedy baseline
    greedy_set = _greedy_build()
    best = list(greedy_set)
    best_size = len(best)
    
    # Step 2: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 20:
        chunk = remaining * 0.7
        result = _simulated_annealing(best, time.time() + chunk, seed=42)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Step 3: Another SA run with different seed
    remaining = deadline - time.time()
    if remaining > 10:
        chunk = remaining * 0.8
        result = _simulated_annealing(best, time.time() + chunk, seed=123)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Final validation
    if not _fast_is_valid(best):
        # Filter to keep only valid subset
        # Greedy filter
        valid = []
        counts = [0] * (MAX_SUM + 1)
        for x in sorted(best):
            ok = True
            for y in valid:
                if counts[x + y] >= 2:
                    ok = False
                    break
            if ok and counts[2 * x] >= 2:
                ok = False
            if ok:
                for y in valid:
                    counts[x + y] += 1
                counts[2 * x] += 1
                valid.append(x)
        best = valid
        best_size = len(best)
    
    return best
