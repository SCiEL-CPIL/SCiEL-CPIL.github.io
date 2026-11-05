import time
import random
import bisect

def _compute_block_sums(candidate):
    """Compute all non-empty consecutive block sums for a candidate sequence."""
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            sums.add(s)
    return sums

def _is_valid_candidate(candidate):
    """Check if all consecutive block sums are distinct."""
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _count_duplicate_sums(candidate):
    """Count number of duplicate consecutive block sums (excess over distinct)."""
    n = len(candidate)
    sums = {}
    dup = 0
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            if s in sums:
                dup += 1
            else:
                sums[s] = 1
    return dup

def _greedy_build(order):
    """Greedy build: iterate through order, add x if it maintains validity."""
    selected = []
    for x in order:
        candidate = selected + [x]
        candidate.sort()
        if _is_valid_candidate(candidate):
            selected = candidate
    return selected

def _try_insert(selected, x):
    """Try to insert x into sorted selected and check validity."""
    pos = bisect.bisect_left(selected, x)
    if pos < len(selected) and selected[pos] == x:
        return None
    new_candidate = selected[:pos] + [x] + selected[pos:]
    if _is_valid_candidate(new_candidate):
        return new_candidate
    return None

def _local_search(selected, available, deadline):
    """Local search: try to insert available numbers into the sequence."""
    current = selected[:]
    available_set = set(available)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        avail_list = list(available_set)
        random.shuffle(avail_list)
        for x in avail_list:
            if x in current:
                continue
            new_seq = _try_insert(current, x)
            if new_seq:
                current = new_seq
                available_set.remove(x)
                improved = True
    return current

def _energy(seq):
    """
    Energy function for simulated annealing.
    Lower is better. Penalizes duplicate sums heavily, rewards length.
    """
    n = len(seq)
    dup = _count_duplicate_sums(seq)
    # Heavy penalty for duplicates, small reward for length
    return dup * 10000 - n

def _simulated_annealing(initial_seq, deadline, rng):
    """
    Simulated annealing to escape local optima.
    State: sorted sequence of integers in [1,256].
    Moves: add element, remove element, swap two elements.
    Energy: penalize duplicate block sums, reward length.
    """
    current = initial_seq[:]
    current_energy = _energy(current)
    best = current[:]
    best_energy = current_energy
    
    T = 50.0
    T_min = 0.01
    cooling_rate = 0.995
    max_iters = 200000
    
    used_set = set(current)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if T < T_min:
            break
        
        # Choose a random move
        move_type = rng.random()
        
        new_seq = current[:]
        
        if move_type < 0.4 and len(current) < 256:
            # Add a random element
            available = [x for x in range(1, 257) if x not in used_set]
            if available:
                x = rng.choice(available)
                pos = bisect.bisect_left(new_seq, x)
                new_seq = new_seq[:pos] + [x] + new_seq[pos:]
                used_set.add(x)
            else:
                continue
                
        elif move_type < 0.7 and len(current) > 1:
            # Remove a random element
            idx = rng.randrange(len(current))
            x = current[idx]
            new_seq = current[:idx] + current[idx+1:]
            used_set.discard(x)
            
        elif move_type < 0.9 and len(current) >= 2:
            # Swap two elements (keep sorted order by re-sorting)
            i, j = rng.sample(range(len(current)), 2)
            # For a strictly increasing sequence, swapping doesn't make sense
            # Instead, let's do a "replace" move: remove one, add another
            x_remove = current[i]
            available = [x for x in range(1, 257) if x not in used_set]
            if available:
                x_add = rng.choice(available)
                new_seq = current[:i] + current[i+1:]
                pos = bisect.bisect_left(new_seq, x_add)
                new_seq = new_seq[:pos] + [x_add] + new_seq[pos:]
                used_set.discard(x_remove)
                used_set.add(x_add)
            else:
                continue
                
        else:
            # Perturbation: remove a random element and add a different one
            if len(current) > 0:
                idx = rng.randrange(len(current))
                x_remove = current[idx]
                available = [x for x in range(1, 257) if x not in used_set]
                if available:
                    x_add = rng.choice(available)
                    new_seq = current[:idx] + current[idx+1:]
                    pos = bisect.bisect_left(new_seq, x_add)
                    new_seq = new_seq[:pos] + [x_add] + new_seq[pos:]
                    used_set.discard(x_remove)
                    used_set.add(x_add)
                else:
                    continue
            else:
                continue
        
        # Check validity of new_seq (must be strictly increasing)
        valid_order = True
        for k in range(len(new_seq) - 1):
            if new_seq[k] >= new_seq[k+1]:
                valid_order = False
                break
        
        if not valid_order:
            # Revert the move
            continue
        
        new_energy = _energy(new_seq)
        delta_e = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta_e / T)
        if delta_e < 0 or (T > 0 and rng.random() < pow(2.718281828, -delta_e / T)):
            current = new_seq
            current_energy = new_energy
            
            if new_energy < best_energy:
                best = new_seq[:]
                best_energy = new_energy
        
        T *= cooling_rate
    
    return best

def _dfs_backtrack(selected, available, max_depth, deadline, memo=None):
    """
    DFS backtracking search.
    """
    if memo is None:
        memo = {}
    
    if len(selected) >= max_depth:
        return selected[:]
    
    if time.time() > deadline:
        return selected[:]
    
    current_sums = _compute_block_sums(selected)
    last_val = selected[-1] if selected else 0
    
    sums_tuple = tuple(sorted(current_sums))
    key = (last_val, sums_tuple)
    
    if key in memo:
        return selected[:]
    
    best_result = selected[:]
    avail_list = sorted(available)
    
    for x in avail_list:
        if time.time() > deadline:
            break
        if x in selected:
            continue
        if selected and x <= selected[-1]:
            continue
        
        new_selected = selected + [x]
        new_sums = _compute_block_sums(new_selected)
        
        if len(new_sums) == len(new_selected) * (len(new_selected) + 1) // 2:
            new_available = available - {x}
            result = _dfs_backtrack(new_selected, new_available, max_depth, deadline, memo)
            if len(result) > len(best_result):
                best_result = result
    
    memo[key] = True
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    
    # Baseline: greedy builds
    seq = _greedy_build(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    seq = _greedy_build(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq
    
    # DFS backtracking from a good starting point
    if len(best_seq) > 0:
        available = set(range(1, 257)) - set(best_seq)
        target_len = len(best_seq) + 1
        dfs_result = _dfs_backtrack(best_seq, available, target_len, deadline)
        if len(dfs_result) > len(best_seq):
            best_seq = dfs_result
    
    # Random restarts with greedy + local search + simulated annealing
    rng = random.Random(42)
    
    while time.time() < deadline:
        order = list(range(1, 257))
        rng.shuffle(order)
        
        seq = _greedy_build(order)
        
        if len(seq) > len(best_seq):
            best_seq = seq
        
        available = set(range(1, 257)) - set(seq)
        improved_seq = _local_search(seq, available, deadline)
        
        if len(improved_seq) > len(best_seq):
            best_seq = improved_seq
        
        # Try DFS extension from this improved sequence
        if len(improved_seq) > 0 and time.time() < deadline - 5:
            avail2 = set(range(1, 257)) - set(improved_seq)
            target2 = len(improved_seq) + 1
            dfs_res2 = _dfs_backtrack(improved_seq, avail2, target2, deadline)
            if len(dfs_res2) > len(best_seq):
                best_seq = dfs_res2
        
        # Simulated annealing to escape local optima
        if time.time() < deadline - 10:
            sa_result = _simulated_annealing(improved_seq, deadline - 5, rng)
            if len(sa_result) > len(best_seq):
                best_seq = sa_result
            # Also try SA from the greedy result directly
            sa_result2 = _simulated_annealing(seq, deadline - 5, rng)
            if len(sa_result2) > len(best_seq):
                best_seq = sa_result2
    
    return best_seq
