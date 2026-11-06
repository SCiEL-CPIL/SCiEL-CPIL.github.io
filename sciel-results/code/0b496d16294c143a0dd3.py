"""Increasing sequences with unique consecutive-block sums.
Uses randomized greedy + simulated annealing on selected indices to escape
local optima and maximize sequence length."""
import time
import random

def _all_block_sums(seq):
    """Compute all nonempty consecutive-block sums for a sequence."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _can_append_fast(values, x, sums):
    """Check if x can be appended without creating duplicate block sums.
    Returns the set of new sums if valid, else None."""
    new_sums = set()
    total = x
    new_sums.add(total)
    for a in reversed(values):
        total += a
        new_sums.add(total)
    if new_sums & sums:
        return None
    return new_sums

def _greedy_greedy(values, sums, start_val, max_val, deadline):
    """Greedy: pick smallest valid x >= start_val."""
    for x in range(start_val, max_val + 1):
        if time.time() > deadline:
            return False
        new_sums = _can_append_fast(values, x, sums)
        if new_sums is not None:
            values.append(x)
            sums.update(new_sums)
            return True
    return False

def _randomized_greedy(start_val, max_val, deadline, seed=0):
    """Greedy with random tie-breaking among valid candidates.
    Returns (values, sums, best_len)."""
    rng = random.Random(seed)
    values = []
    sums = set()
    
    while True:
        if time.time() > deadline:
            break
        # Collect all valid candidates
        valid = []
        for x in range(max(values[-1] + 1 if values else 1, start_val), max_val + 1):
            new_sums = _can_append_fast(values, x, sums)
            if new_sums is not None:
                valid.append((x, new_sums))
        if not valid:
            break
        # Pick randomly from valid candidates (bias toward smaller values)
        weights = [1.0 / (1.0 + (x - start_val) * 0.05) for x, _ in valid]
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        chosen_idx = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                chosen_idx = i
                break
        x, new_sums = valid[chosen_idx]
        values.append(x)
        sums.update(new_sums)
    
    return values, sums, len(values)

def _simulated_annealing(initial_seq, deadline, rng, max_iters=5000):
    """Simulated annealing on the set of selected indices.
    State: a strictly increasing sequence with distinct consecutive-block sums.
    Moves: add a new element, remove an element, swap two elements.
    Objective: maximize sequence length (primary), then minimize max element (secondary)."""
    import math
    
    current = list(initial_seq)
    cur_sums = _all_block_sums(current)
    best = list(current)
    best_sums = set(cur_sums)
    best_len = len(best)
    
    # Initial temperature
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    iters = 0
    while time.time() < deadline and iters < max_iters:
        iters += 1
        if T < T_min:
            T = T_min
        
        # Choose a random move
        move_type = rng.random()
        n = len(current)
        
        new_seq = None
        new_sums = None
        
        if move_type < 0.3 and n < 256:
            # Try to add a new element at a random position
            # Generate a candidate value
            if n == 0:
                lo, hi = 1, 256
            else:
                # Try inserting between existing elements or at ends
                pos = rng.randint(0, n)
                if pos == 0:
                    lo, hi = 1, current[0] - 1
                elif pos == n:
                    lo, hi = current[-1] + 1, 256
                else:
                    lo, hi = current[pos-1] + 1, current[pos] - 1
            
            if lo <= hi:
                x = rng.randint(lo, hi)
                test = list(current)
                test.insert(pos, x)
                test_sums = _all_block_sums(test)
                if len(test_sums) == len(test) * (len(test) + 1) // 2:
                    new_seq = test
                    new_sums = test_sums
        
        elif move_type < 0.5 and n > 1:
            # Try to remove an element
            pos = rng.randint(0, n - 1)
            test = list(current)
            test.pop(pos)
            if test:  # non-empty
                test_sums = _all_block_sums(test)
                if len(test_sums) == len(test) * (len(test) + 1) // 2:
                    new_seq = test
                    new_sums = test_sums
        
        elif move_type < 0.8 and n >= 2:
            # Try to swap two elements (if result is still strictly increasing)
            i, j = rng.sample(range(n), 2)
            if i > j:
                i, j = j, i
            test = list(current)
            test[i], test[j] = test[j], test[i]
            # Check if still strictly increasing
            valid = all(test[k] < test[k+1] for k in range(len(test)-1))
            if valid:
                test_sums = _all_block_sums(test)
                if len(test_sums) == len(test) * (len(test) + 1) // 2:
                    new_seq = test
                    new_sums = test_sums
        
        elif move_type < 0.95 and n >= 1:
            # Try to replace an element with a nearby value
            pos = rng.randint(0, n - 1)
            delta = rng.randint(-5, 5)
            new_val = current[pos] + delta
            if 1 <= new_val <= 256:
                test = list(current)
                test[pos] = new_val
                # Check strict increasing
                valid = True
                if pos > 0 and test[pos] <= test[pos-1]:
                    valid = False
                if pos < n-1 and test[pos] >= test[pos+1]:
                    valid = False
                if valid:
                    test_sums = _all_block_sums(test)
                    if len(test_sums) == len(test) * (len(test) + 1) // 2:
                        new_seq = test
                        new_sums = test_sums
        
        if new_seq is None:
            T *= cooling
            continue
        
        # Evaluate the move
        old_len = len(current)
        new_len = len(new_seq)
        
        # Score: primarily length, secondarily negative of max element (prefer smaller)
        old_score = old_len * 1000 - max(current) if current else 0
        new_score = new_len * 1000 - max(new_seq) if new_seq else 0
        delta_score = new_score - old_score
        
        # Accept if better, or with probability exp(delta_score / T) if worse
        if delta_score >= 0:
            current = new_seq
            cur_sums = new_sums
            if len(current) > best_len:
                best = list(current)
                best_sums = set(cur_sums)
                best_len = len(best)
        else:
            accept_prob = math.exp(delta_score / T)
            if rng.random() < accept_prob:
                current = new_seq
                cur_sums = new_sums
        
        T *= cooling
    
    return best, best_len

def _try_extend(values, sums, deadline):
    """Try to extend the sequence by appending valid elements."""
    extended = False
    while True:
        if time.time() > deadline:
            break
        found = False
        for x in range(values[-1] + 1 if values else 1, 257):
            new_sums = _can_append_fast(values, x, sums)
            if new_sums is not None:
                values.append(x)
                sums.update(new_sums)
                extended = True
                found = True
                break
        if not found:
            break
    return extended

def _local_search(values, sums, deadline, rng, max_iters=200):
    """Iterative local search: try replacing elements to enable further extension."""
    best = list(values)
    best_sums = set(sums)
    best_len = len(best)
    iters = 0
    
    while time.time() < deadline and iters < max_iters:
        iters += 1
        current = list(values)
        cur_sums = set(sums)
        
        # Try to extend from current state
        _try_extend(current, cur_sums, deadline)
        
        if len(current) > best_len:
            best = list(current)
            best_sums = _all_block_sums(current)
            best_len = len(best)
            values = current
            sums = best_sums
            continue
        
        # Try a random perturbation: replace one element
        if len(current) > 1:
            pos = rng.randint(0, len(current) - 1)
            old_val = current[pos]
            for delta in [1, -1, 2, -2, 3, -3]:
                new_val = old_val + delta
                if new_val < 1 or new_val > 256:
                    continue
                if pos > 0 and new_val <= current[pos - 1]:
                    continue
                if pos < len(current) - 1 and new_val >= current[pos + 1]:
                    continue
                test = list(current)
                test[pos] = new_val
                test_sums = _all_block_sums(test)
                if len(test_sums) == len(test) * (len(test) + 1) // 2:
                    ext = list(test)
                    ext_sums = set(test_sums)
                    _try_extend(ext, ext_sums, deadline)
                    if len(ext) > best_len:
                        best = list(ext)
                        best_sums = _all_block_sums(ext)
                        best_len = len(best)
                        values = ext
                        sums = best_sums
                        break
    
    return best, best_sums, best_len

def _multi_start_search(deadline, n_starts=10):
    """Run multiple randomized greedy starts and local search from each."""
    best = []
    best_len = 0
    rng = random.Random(42)
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
        start_val = rng.randint(1, 5)
        values, sums, length = _randomized_greedy(start_val, 256, deadline, seed=i)
        
        if length > best_len:
            best = values
            best_len = length
        
        # Local search from this state
        if time.time() < deadline - 2:
            improved_vals, improved_sums, improved_len = _local_search(
                values, sums, deadline, rng, max_iters=50
            )
            if improved_len > best_len:
                best = improved_vals
                best_len = improved_len
    
    return best, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Phase 1: Fast greedy baseline (deterministic)
    values = []
    sums = set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        new_sums = _can_append_fast(values, x, sums)
        if new_sums is not None:
            values.append(x)
            sums.update(new_sums)
    
    best = list(values)
    best_len = len(best)
    
    # Phase 2: Multi-start randomized search + local search
    if time.time() < deadline - 20:
        ms_best, ms_len = _multi_start_search(deadline - 20, n_starts=8)
        if ms_len > best_len:
            best = ms_best
            best_len = ms_len
    
    # Phase 3: Simulated annealing on the best sequence
    if time.time() < deadline - 10 and best_len < 256:
        rng = random.Random(99)
        sa_best, sa_len = _simulated_annealing(
            best, deadline - 10, rng, max_iters=10000
        )
        if sa_len > best_len:
            best = sa_best
            best_len = sa_len
    
    # Phase 4: Final local search refinement
    if time.time() < deadline - 3 and best_len < 256:
        rng = random.Random(123)
        refined_vals, refined_sums, refined_len = _local_search(
            best, _all_block_sums(best), deadline - 3, rng, max_iters=100
        )
        if refined_len > best_len:
            best = refined_vals
            best_len = refined_len
    
    return best
