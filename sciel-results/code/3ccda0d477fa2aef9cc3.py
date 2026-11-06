"""Combined program: greedy baseline + randomized greedy + targeted backtracking
with deadline guard for strictly increasing sequences in [1,256] with distinct
consecutive-block sums."""
import time
import random

def _get_all_block_sums(values):
    """Get all consecutive block sums for a sequence."""
    sums = set()
    n = len(values)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += values[j]
            sums.add(s)
    return sums

def _can_append(values, x, sums):
    """Check if x can be appended to values without creating duplicate block sums.
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

def _greedy_smallest(values, sums, start_val, max_val, deadline):
    """Greedy: pick smallest valid x >= start_val, extend as far as possible."""
    while time.time() < deadline:
        added = False
        for x in range(start_val, max_val + 1):
            new_sums = _can_append(values, x, sums)
            if new_sums is not None:
                values.append(x)
                sums.update(new_sums)
                start_val = x + 1
                added = True
                break
        if not added:
            break
    return values, sums

def _greedy_randomized(seed, start_val, max_val, deadline):
    """Randomized greedy: at each step, try candidates in random order."""
    random.seed(seed)
    values = []
    sums = set()
    remaining = list(range(start_val, max_val + 1))
    
    while remaining and time.time() < deadline:
        random.shuffle(remaining)
        added = False
        for x in remaining:
            if values and x <= values[-1]:
                continue
            new_sums = _can_append(values, x, sums)
            if new_sums is not None:
                values.append(x)
                sums.update(new_sums)
                remaining.remove(x)
                added = True
                break
        if not added:
            break
    return values

def _backtrack_extend(values, sums, min_val, max_val, deadline, max_depth=12):
    """Backtracking search to extend a sequence. Returns best found."""
    best = list(values)
    
    if time.time() > deadline:
        return best
    
    if len(values) >= max_depth:
        return best
    
    # Collect valid candidates
    candidates = []
    for x in range(min_val, max_val + 1):
        if time.time() > deadline:
            break
        new_sums = _can_append(values, x, sums)
        if new_sums is not None:
            candidates.append((x, new_sums))
    
    # Try each candidate
    for x, new_sums in candidates:
        if time.time() > deadline:
            break
        
        values.append(x)
        sums.update(new_sums)
        
        if len(values) > len(best):
            best = list(values)
        
        # Recurse
        result = _backtrack_extend(values, sums, x + 1, max_val, deadline, max_depth)
        if len(result) > len(best):
            best = result
        
        # Undo
        values.pop()
        for s in new_sums:
            sums.discard(s)
    
    return best

def _local_search(values, sums, deadline):
    """Try removing elements and re-extending to find longer sequences."""
    best = list(values)
    best_len = len(best)
    n = len(values)
    
    for i in range(n - 1, max(n - 5, 0), -1):
        if time.time() > deadline:
            break
        
        # Try removing element at position i and re-extending
        trial = list(values[:i])
        trial_sums = _get_all_block_sums(trial)
        min_val = trial[-1] + 1 if trial else 1
        
        result = _backtrack_extend(trial, trial_sums, min_val, 256, deadline, max_depth=10)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8 second safety margin before 90s kill
    
    # Phase 1: Standard greedy (smallest valid)
    values = []
    sums = set()
    values, sums = _greedy_smallest(values, sums, 1, 256, deadline)
    best = list(values)
    best_len = len(best)
    
    # Phase 2: Randomized greedy with multiple seeds
    for seed in range(30):
        if time.time() > deadline - 5:
            break
        result = _greedy_randomized(seed, 1, 256, deadline)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 3: Backtracking to extend the best sequence found
    if time.time() < deadline - 3:
        best_sums = _get_all_block_sums(best)
        min_val = best[-1] + 1 if best else 1
        result = _backtrack_extend(list(best), set(best_sums), min_val, 256, deadline, max_depth=12)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 4: Local search - try removing tail elements and re-extending
    if time.time() < deadline - 2:
        best_sums = _get_all_block_sums(best)
        result = _local_search(list(best), set(best_sums), deadline)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 5: Try backtracking from shorter prefixes of the original greedy
    if time.time() < deadline - 1:
        # Recompute original greedy
        g_values = []
        g_sums = set()
        g_values, g_sums = _greedy_smallest(g_values, g_sums, 1, 256, deadline)
        
        for trunc in range(len(g_values) - 1, max(len(g_values) - 4, 0), -1):
            if time.time() > deadline - 0.5:
                break
            prefix = list(g_values[:trunc])
            prefix_sums = _get_all_block_sums(prefix)
            min_val = prefix[-1] + 1 if prefix else 1
            result = _backtrack_extend(prefix, prefix_sums, min_val, 256, deadline, max_depth=10)
            if len(result) > best_len:
                best = result
                best_len = len(result)
    
    return best
