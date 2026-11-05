import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0
START_TIME = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)

def _build_counts(chosen, cap=1000):
    """Build sum-count array from a list of chosen integers."""
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    if not chosen:
        return counts
    c = np.asarray(chosen, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    np.add.at(counts, sums, 1)
    return counts

def _is_valid_addition(x, chosen_np, counts):
    """Check if x can be added without violating the B2[2] constraint."""
    if chosen_np.size == 0:
        new_sums = np.array([2 * x], dtype=np.int64)
    else:
        new_sums = np.concatenate((chosen_np + x, np.array([2 * x], dtype=np.int64)))
    return np.all(counts[new_sums] < 2)

def _scored_greedy(cap=1000, seed=42, time_fn=None, stop_fn=None):
    """
    Greedy construction with a scoring heuristic:
    - At each step, consider a batch of candidate elements.
    - Score each candidate by how 'tight' it makes the sums (prefer those that create fewer count==2 sums).
    - Pick the best candidate.
    """
    rng = random.Random(seed)
    pool = list(range(1, cap + 1))
    rng.shuffle(pool)
    pool_set = set(pool)
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    
    while pool:
        if time_fn is not None and time_fn() < 5.0:
            break
        if stop_fn is not None and stop_fn():
            break
        
        # Take a batch of candidates to evaluate
        batch_size = min(100, len(pool))
        batch = pool[:batch_size]
        
        if chosen_np.size == 0:
            # First element: any is fine, pick the first
            x = batch[0]
            chosen.append(x)
            pool = pool[1:]
            pool_set.discard(x)
            chosen_np = np.array([x], dtype=np.int64)
            counts[2 * x] += 1
            continue
        
        # Evaluate each candidate in the batch
        best_x = None
        best_score = float('inf')
        best_new_sums = None
        
        for x in batch:
            new_sums = np.concatenate((chosen_np + x, np.array([2 * x], dtype=np.int64)))
            # Check feasibility
            if np.any(counts[new_sums] >= 2):
                continue
            # Score: number of sums that would become count==2 (tighter is better? or worse?)
            # We want to minimize the number of new count==2 sums to leave room for future additions.
            # Alternatively, we might want to maximize density. Let's try minimizing tight sums.
            new_counts = counts[new_sums] + 1
            tight = (new_counts == 2).sum()
            # Also consider the max count reached
            max_count = new_counts.max()
            # Score: primarily min tight, secondarily min max_count
            score = tight * 1000 + max_count
            if score < best_score:
                best_score = score
                best_x = x
                best_new_sums = new_sums
        
        if best_x is None:
            # No feasible candidate in this batch, discard them
            for x in batch:
                pool_set.discard(x)
            pool = pool[batch_size:]
            continue
        
        # Add the best candidate
        chosen.append(best_x)
        pool = pool[batch_size:]  # Remove the entire batch, including best_x
        pool_set.discard(best_x)
        chosen_np = np.concatenate((chosen_np, np.array([best_x], dtype=np.int64)))
        counts[best_new_sums] += 1
    
    return chosen

def _local_improve(chosen, cap=1000, time_fn=None, stop_fn=None, max_iter=50000):
    """
    Local search improvement:
    1. Try to add any remaining element.
    2. Try swaps: remove one element, add one or two others.
    """
    if not chosen:
        return chosen
    
    current = set(chosen)
    counts = _build_counts(list(current), cap)
    best_size = len(current)
    best_set = set(current)
    
    rng = random.Random(123)
    candidates = [x for x in range(1, cap + 1) if x not in current]
    
    no_improve = 0
    max_no_improve = 1000
    
    for it in range(max_iter):
        if time_fn is not None and time_fn() < 5.0:
            break
        if stop_fn is not None and stop_fn():
            break
        
        # Strategy 1: Try to add a random candidate if possible
        if candidates:
            try_add = rng.sample(candidates, min(10, len(candidates)))
            added = False
            for x in try_add:
                chosen_arr = np.asarray(sorted(current), dtype=np.int64)
                if _is_valid_addition(x, chosen_arr, counts):
                    current.add(x)
                    candidates.remove(x)
                    new_sums = np.concatenate((chosen_arr + x, np.array([2 * x], dtype=np.int64)))
                    counts[new_sums] += 1
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
                    no_improve = 0
                    added = True
                    break
            if added:
                continue
        
        # Strategy 2: Try a 1-for-1 swap
        if current and candidates:
            rem = rng.choice(list(current))
            other = [y for y in current if y != rem]
            other_arr = np.asarray(sorted(other), dtype=np.int64)
            
            # Calculate sums removed by taking out 'rem'
            removed_sums = np.concatenate((other_arr + rem, np.array([2 * rem], dtype=np.int64)))
            
            # Tentative counts
            tent_counts = counts.copy()
            tent_counts[removed_sums] -= 1
            
            # Try to find a candidate x that is valid under tent_counts
            found_swap = False
            swap_candidates = rng.sample(candidates, min(20, len(candidates)))
            for x in swap_candidates:
                if x == rem:
                    continue
                new_sums = np.concatenate((other_arr + x, np.array([2 * x], dtype=np.int64)))
                if np.all(tent_counts[new_sums] < 2):
                    # Perform swap
                    current.remove(rem)
                    current.add(x)
                    if rem in candidates:
                        candidates.remove(rem)
                    if x in candidates:
                        candidates.remove(x)
                    
                    counts = tent_counts
                    counts[new_sums] += 1
                    
                    found_swap = True
                    no_improve = 0
                    break
            
            if not found_swap:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
    
    if len(best_set) > len(current):
        return list(best_set)
    return list(current)

def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 115.0
    
    cap = 1000

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 15.0

    best = []
    best_len = -1

    # Phase 1: Multiple scored greedy runs with different seeds
    for seed in range(20):
        if _time_left() < 20.0:
            break
        res = _scored_greedy(cap=cap, seed=seed, time_fn=time_left, stop_fn=stopped)
        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: Local improvement on the best greedy solution
    if _time_left() > 25.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved

    # Phase 3: Try simple structured patterns (quick checks)
    patterns = [
        list(range(1, cap + 1, 2)),  # every 2nd
        [i for i in range(1, cap + 1) if i % 3 != 0],  # skip multiples of 3
        [i for i in range(1, cap + 1) if i % 4 in (1, 2)],  # 2 per 4
    ]
    for pat in patterns:
        if len(pat) > best_len:
            # Validate pattern
            c = np.asarray(pat, dtype=np.int64)
            n = len(c)
            ii, jj = np.triu_indices(n)
            sums = c[ii] + c[jj]
            if sums.size > 0:
                counts = np.bincount(sums, minlength=2 * cap + 1)
                if counts.max() <= 2:
                    best_len = len(pat)
                    best = pat

    # Final validation and return
    if best:
        c = np.asarray(best, dtype=np.int64)
        n = len(c)
        ii, jj = np.triu_indices(n)
        sums = c[ii] + c[jj]
        if sums.size > 0:
            counts = np.bincount(sums, minlength=2 * cap + 1)
            if counts.max() > 2:
                # Fallback to empty or a simple valid set
                return list(range(1, cap + 1, 3))
    
    return best
