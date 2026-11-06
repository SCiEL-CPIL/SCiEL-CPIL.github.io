import time
import random
import math

def _check_valid(seq):
    """Verify strictly increasing, in [1,256], and all consecutive-block sums distinct."""
    n = len(seq)
    if n == 0:
        return False
    for i in range(n):
        if seq[i] < 1 or seq[i] > 256:
            return False
        if i > 0 and seq[i] <= seq[i - 1]:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_cost(seq):
    """
    Returns a cost value. Lower is better.
    Cost = (number of sum collisions) * 1000 - len(seq)
    """
    n = len(seq)
    if n == 0:
        return 0
    
    sums = []
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.append(s)
            
    collisions = len(sums) - len(set(sums))
    return collisions * 1000 - n

def _greedy_build(preferred_order):
    """
    Build a strictly increasing sequence with distinct consecutive-block sums.
    """
    values = []
    sums = set()
    candidates = list(preferred_order)
    
    while candidates:
        best_val = None
        for x in candidates:
            if values and x <= values[-1]:
                continue
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                best_val = x
                break
        
        if best_val is None:
            break
        
        values.append(best_val)
        tails = [best_val]
        total = best_val
        for a in reversed(values[:-1]):
            total += a
            tails.append(total)
        sums.update(tails)
        
        candidates.remove(best_val)
    
    return values

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # 1. Get multiple good initial solutions using different greedy orders
    initial_candidates = []
    
    # Smallest first
    initial_candidates.append(_greedy_build(list(range(1, 257))))
    # Largest first
    initial_candidates.append(_greedy_build(list(range(256, 0, -1))))
    # Random shuffles
    rng_init = random.Random(1)
    for _ in range(20):
        order = list(range(1, 257))
        rng_init.shuffle(order)
        initial_candidates.append(_greedy_build(order))
        
    # Pick the best initial candidate
    best_seq = initial_candidates[0]
    best_cost = _get_cost(best_seq)
    for seq in initial_candidates:
        c = _get_cost(seq)
        if c < best_cost:
            best_cost = c
            best_seq = seq
            
    # Ensure best_seq is valid
    if not _check_valid(best_seq):
        best_seq = [1]
        best_cost = _get_cost(best_seq)
        
    # 2. Simulated Annealing
    current_seq = list(best_seq)
    current_cost = best_cost
    
    used_set = set(current_seq)
    all_numbers = set(range(1, 257))
    
    T_start = 100.0
    T_end = 0.1
    T = T_start
    # Adjust decay based on expected steps
    # Let's assume 500,000 steps in 85s (approx 6000 steps/sec)
    decay = (T_end / T_start) ** (1.0 / 500000.0)
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Propose move
        move_type = rng.random()
        
        new_seq = current_seq
        new_used = used_set
        
        if move_type < 0.5:
            # Local move: Perturb an existing element
            if not new_seq:
                continue
            idx = rng.randint(0, len(new_seq) - 1)
            x_old = new_seq[idx]
            
            # Try to replace x_old with x_old + delta
            delta = rng.randint(-10, 10)
            if delta == 0:
                continue
            x_new = x_old + delta
            
            if x_new < 1 or x_new > 256:
                continue
            if x_new in new_used:
                continue
            
            prev_val = new_seq[idx-1] if idx > 0 else 0
            next_val = new_seq[idx+1] if idx < len(new_seq)-1 else 257
            
            if prev_val < x_new < next_val:
                new_seq = new_seq[:idx] + [x_new] + new_seq[idx+1:]
                new_used = new_used - {x_old} | {x_new}
            else:
                continue
                
        elif move_type < 0.75:
            # Global Add
            unused = list(all_numbers - new_used)
            if not unused:
                continue
            x = rng.choice(unused)
            lo, hi = 0, len(new_seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if new_seq[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            new_seq = new_seq[:lo] + [x] + new_seq[lo:]
            new_used = new_used | {x}
            
        elif move_type < 0.95:
            # Global Remove
            if not new_seq:
                continue
            idx = rng.randint(0, len(new_seq) - 1)
            x = new_seq[idx]
            new_seq = new_seq[:idx] + new_seq[idx+1:]
            new_used = new_used - {x}
            
        else:
            # Global Swap
            if not new_seq:
                continue
            unused = list(all_numbers - new_used)
            if not unused:
                continue
            idx = rng.randint(0, len(new_seq) - 1)
            x_out = new_seq[idx]
            x_in = rng.choice(unused)
            
            new_seq = new_seq[:idx] + new_seq[idx+1:]
            lo, hi = 0, len(new_seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if new_seq[mid] < x_in:
                    lo = mid + 1
                else:
                    hi = mid
            new_seq = new_seq[:lo] + [x_in] + new_seq[lo:]
            
            new_used = new_used - {x_out} | {x_in}
        
        # Calculate new cost
        new_cost = _get_cost(new_seq)
        
        # Acceptance
        delta = new_cost - current_cost
        if delta < 0:
            current_seq = new_seq
            current_cost = new_cost
            used_set = new_used
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_seq = current_seq
        else:
            if T > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_seq = new_seq
                    current_cost = new_cost
                    used_set = new_used
        
        # Cool down
        T *= decay
        if T < T_end:
            T = T_end
            
    # 3. Post-processing: Greedy extension
    extended_seq = list(best_seq)
    
    # Try adding small numbers
    for x in range(1, 257):
        if x in extended_seq:
            continue
        if not extended_seq or x > extended_seq[-1]:
            new_seq = extended_seq + [x]
            if _check_valid(new_seq):
                extended_seq = new_seq
        
    # Try adding large numbers
    for x in range(256, 0, -1):
        if x in extended_seq:
            continue
        if not extended_seq or x > extended_seq[-1]:
            new_seq = extended_seq + [x]
            if _check_valid(new_seq):
                extended_seq = new_seq

    if len(extended_seq) > len(best_seq):
        best_seq = extended_seq

    # Final validation
    if _check_valid(best_seq):
        return best_seq
    
    # Fallback
    return [1]
