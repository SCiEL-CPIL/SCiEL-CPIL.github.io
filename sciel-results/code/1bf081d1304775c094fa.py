"""Improved program: backtracking with fast numpy-based sum checking and local search."""
import time
import random
import numpy as np

def _compute_block_sums(arr):
    """Compute all consecutive block sums using prefix sums. Returns a set."""
    if len(arr) == 0:
        return set()
    prefix = np.concatenate(([0], np.cumsum(arr)))
    sums = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n+1):
            sums.add(int(prefix[j] - prefix[i]))
    return sums

def _can_append_fast(values, x, existing_sums):
    """Check if appending x creates any duplicate block sum. Fast using prefix sums."""
    if not values:
        return True
    # New block sums when appending x:
    # x itself, x + values[-1], x + values[-1] + values[-2], ..., x + sum(values)
    # These must not be in existing_sums, and must be distinct from each other
    # (they are distinct since values are positive and increasing)
    total = x
    new_sums = [total]
    for a in reversed(values):
        total += a
        new_sums.append(total)
    # Check distinctness among new_sums (they are strictly increasing since values > 0)
    # So just check against existing_sums
    for s in new_sums:
        if s in existing_sums:
            return False
    return True

def _get_new_sums(values, x):
    """Get the new block sums when appending x to values."""
    total = x
    new_sums = {total}
    for a in reversed(values):
        total += a
        new_sums.add(total)
    return new_sums

def _greedy_min(start_val=1, max_val=256):
    """Greedy: pick smallest valid value at each step."""
    values = []
    sums = set()
    for x in range(start_val, max_val + 1):
        if values and x <= values[-1]:
            continue
        if _can_append_fast(values, x, sums):
            values.append(x)
            sums |= _get_new_sums(values[:-1], x)
    return values

def _greedy_randomized(seed, start_val=1, max_val=256):
    """Randomized greedy."""
    random.seed(seed)
    values = []
    sums = set()
    candidates = list(range(start_val, max_val + 1))
    while candidates:
        random.shuffle(candidates)
        added = False
        for x in candidates:
            if values and x <= values[-1]:
                continue
            if _can_append_fast(values, x, sums):
                values.append(x)
                sums |= _get_new_sums(values[:-1], x)
                candidates.remove(x)
                added = True
                break
        if not added:
            break
    return values

def _backtrack_extend(values, sums, min_val, max_val, deadline, max_depth=200):
    """Backtracking to extend the sequence. Returns best found."""
    best = list(values)
    best_len = len(best)
    
    def _search(vals, sums_set, min_v, depth):
        nonlocal best, best_len
        if time.time() > deadline:
            return
        if depth > max_depth:
            return
        if len(vals) > best_len:
            best = list(vals)
            best_len = len(best)
        
        # Try to extend
        for x in range(min_v, max_val + 1):
            if time.time() > deadline:
                return
            if vals and x <= vals[-1]:
                continue
            if _can_append_fast(vals, x, sums_set):
                new_sums = sums_set | _get_new_sums(vals, x)
                vals.append(x)
                _search(vals, new_sums, x + 1, depth + 1)
                vals.pop()
                # Prune: if we found a longer sequence, no need to continue at this level
                if len(best) > len(vals) + (max_val - x):
                    # Can't beat best even with all remaining
                    return
    
    _search(list(values), sums, min_val, 0)
    return best

def _local_search_extend(values, sums, deadline, max_iterations=5000):
    """Try to extend by trying all possible values, then doing local search."""
    best = list(values)
    best_sums = set(sums)
    best_len = len(best)
    
    # First try simple extension
    for x in range(best[-1] + 1 if best else 1, 257):
        if _can_append_fast(best, x, best_sums):
            new_sums = best_sums | _get_new_sums(best, x)
            best.append(x)
            best_sums = new_sums
            best_len = len(best)
    
    # Now try to improve by replacing elements
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        improved = False
        # Try replacing each element with a larger one
        for i in range(len(best)):
            if time.time() > deadline:
                break
            old_val = best[i]
            # Try replacing with larger values
            for new_val in range(old_val + 1, 257):
                if i > 0 and new_val <= best[i-1]:
                    continue
                if i < len(best) - 1 and new_val >= best[i+1]:
                    break
                # Check if replacement is valid
                trial = list(best)
                trial[i] = new_val
                # Recompute sums
                trial_sums = _compute_block_sums(np.array(trial, dtype=np.int64))
                if len(trial_sums) == len(trial) * (len(trial) + 1) // 2:
                    # Valid!
                    best = trial
                    best_sums = trial_sums
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try inserting a new element
            inserted = False
            for pos in range(len(best) + 1):
                if time.time() > deadline:
                    break
                min_allowed = best[pos-1] + 1 if pos > 0 else 1
                max_allowed = best[pos] - 1 if pos < len(best) else 256
                for x in range(min_allowed, max_allowed + 1):
                    trial = list(best)
                    trial.insert(pos, x)
                    trial_sums = _compute_block_sums(np.array(trial, dtype=np.int64))
                    if len(trial_sums) == len(trial) * (len(trial) + 1) // 2:
                        best = trial
                        best_sums = trial_sums
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                break
    
    return best

def construct():
    deadline = time.time() + 80.0  # 10s safety margin
    
    # Phase 1: Get a good baseline with greedy
    best = _greedy_min()
    best_len = len(best)
    
    # Phase 2: Try randomized greedy with many seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        result = _greedy_randomized(seed)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 3: Backtracking extension from best
    if time.time() < deadline:
        sums = _compute_block_sums(np.array(best, dtype=np.int64))
        min_val = best[-1] + 1 if best else 1
        result = _backtrack_extend(best, sums, min_val, 256, deadline, max_depth=30)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 4: Local search to extend/improve
    if time.time() < deadline:
        sums = _compute_block_sums(np.array(best, dtype=np.int64))
        result = _local_search_extend(best, sums, deadline, max_iterations=2000)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 5: Final backtracking from the improved result
    if time.time() < deadline:
        sums = _compute_block_sums(np.array(best, dtype=np.int64))
        min_val = best[-1] + 1 if best else 1
        result = _backtrack_extend(best, sums, min_val, 256, deadline, max_depth=50)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Validate final result
    if best:
        arr = np.array(best, dtype=np.int64)
        all_sums = _compute_block_sums(arr)
        expected = len(best) * (len(best) + 1) // 2
        if len(all_sums) != expected:
            # Fallback to greedy
            best = _greedy_min()
    
    return best
