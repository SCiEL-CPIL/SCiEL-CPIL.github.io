import time
import random
import numpy as np

DEADLINE_SECONDS = 120.0
START_TIME = time.time()


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen):
    n = len(chosen)
    if n == 0:
        return True
    c = np.asarray(chosen, dtype=np.int64)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    max_s = int(c.max()) * 2 if n else 0
    if max_s < 1:
        return True
    counts = np.bincount(sums, minlength=max_s + 1)
    return bool(counts.max() <= 2)


def _build_counts(chosen, cap=1000):
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    if not chosen:
        return counts
    c = np.asarray(chosen, dtype=np.int64)
    ii, jj = np.triu_indices(len(c))
    sums = c[ii] + c[jj]
    np.add.at(counts, sums, 1)
    return counts


def _scored_greedy(cap=1000, time_fn=None, stop_fn=None, seed=42):
    rng = random.Random(seed)
    pool = set(range(1, cap + 1))
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    
    while pool:
        if time_fn is not None and time_fn() < 3.0:
            break
        if stop_fn is not None and stop_fn():
            break
        pool_list = list(pool)
        if len(pool_list) <= 300:
            batch = pool_list
        else:
            batch = rng.sample(pool_list, 300)
        arr = np.array(batch, dtype=np.int64)
        if chosen_np.size == 0:
            new_sums = 2 * arr[:, None]
        else:
            cross = arr[:, None] + chosen_np[None, :]
            new_sums = np.concatenate([cross, 2 * arr[:, None]], axis=1)
        gathered = counts[new_sums] + 1
        cand_max = gathered.max(axis=1)
        feasible = np.where(cand_max <= 2)[0]
        if feasible.size == 0:
            for b in batch:
                pool.discard(b)
            continue
        maxes = cand_max[feasible]
        min_max = maxes.min()
        tight_candidates = feasible[maxes == min_max]
        tight_counts = (gathered[tight_candidates] == 2).sum(axis=1)
        best_local = tight_candidates[np.argmin(tight_counts)]
        x = int(arr[best_local])
        chosen.append(x)
        pool.discard(x)
        if chosen_np.size == 0:
            new_sums_x = np.array([2 * x], dtype=np.int64)
        else:
            new_sums_x = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
        counts[new_sums_x] += 1
    return chosen


def _local_improve(chosen, cap=1000, time_fn=None, stop_fn=None, max_iter=100000):
    """
    Optimized local search: vectorized scoring, batch candidate testing,
    and re-consideration of previously removed elements.
    """
    if not chosen:
        return chosen
    
    current = set(chosen)
    counts = _build_counts(list(current), cap)
    best_size = len(current)
    best_set = set(current)
    
    rng = random.Random(123)
    all_elements = list(range(1, cap + 1))
    candidates = [x for x in all_elements if x not in current]
    removed_pool = []  # Track removed elements for re-consideration
    
    no_improve = 0
    max_no_improve = 2000
    
    for it in range(max_iter):
        if time_fn is not None and time_fn() < 5.0:
            break
        if stop_fn is not None and stop_fn():
            break
        
        # Strategy 1: Try to add candidates (including from removed_pool)
        if candidates or removed_pool:
            # Mix candidates with recently removed elements
            pool = candidates[:]
            if removed_pool:
                # Add some removed elements back to the pool for re-consideration
                k = min(len(removed_pool), 10)
                pool.extend(rng.sample(removed_pool, k))
            
            if pool:
                # Test up to 20 candidates at once using vectorization
                test_batch = rng.sample(pool, min(20, len(pool)))
                chosen_arr = np.asarray(sorted(current), dtype=np.int64)
                
                for x in test_batch:
                    new_sums = np.concatenate([chosen_arr + x, np.array([2 * x], dtype=np.int64)])
                    if (counts[new_sums] + 1).max() <= 2:
                        current.add(x)
                        if x in candidates:
                            candidates.remove(x)
                        if x in removed_pool:
                            removed_pool.remove(x)
                        counts[new_sums] += 1
                        best_size = len(current)
                        best_set = set(current)
                        no_improve = 0
                        break
                else:
                    no_improve += 1
                if no_improve > max_no_improve:
                    break
                continue
        
        # Strategy 2: Vectorized swap search
        if not current:
            break
        
        # Compute tight counts for each element using vectorized approach
        curr_sorted = sorted(current)
        curr_arr = np.asarray(curr_sorted, dtype=np.int64)
        n = len(curr_sorted)
        
        # For each element, count how many sums it participates in that have count == 2
        # An element x participates in sums: x+y for y>=x in current, and 2x
        tight_counts = np.zeros(n, dtype=np.int32)
        
        for i in range(n):
            x = curr_arr[i]
            # Sums involving x: x + curr_arr[i:] (since a<=b)
            involved_sums = x + curr_arr[i:]
            tight_counts[i] = (counts[involved_sums] == 2).sum()
        
        # Try removing elements with highest tight counts
        # Sort indices by tight count descending
        order = np.argsort(-tight_counts)
        
        found_swap = False
        max_remove_attempts = min(15, n)
        
        for idx in order[:max_remove_attempts]:
            rem = curr_arr[idx]
            
            # Compute other elements
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            other_arr = curr_arr[mask]
            
            # Removed sums
            removed_sums = np.concatenate([other_arr + rem, np.array([2 * rem], dtype=np.int64)])
            
            # Tentative counts
            tent_counts = counts.copy()
            tent_counts[removed_sums] -= 1
            
            # Try to find a valid replacement from candidates + removed_pool
            if candidates or removed_pool:
                pool = candidates[:]
                if removed_pool:
                    k = min(len(removed_pool), 50)
                    pool.extend(rng.sample(removed_pool, k))
                
                # Test candidates in batches
                batch_size = min(len(pool), 200)
                test_candidates = np.array(pool[:batch_size], dtype=np.int64)
                
                # Vectorized check: for each candidate x, check if all new sums are valid
                # new_sums for x: other_arr + x and 2*x
                cross = other_arr[:, None] + test_candidates[None, :]
                doubles = (2 * test_candidates)[:, None]
                all_new_sums = np.concatenate([cross, doubles], axis=0)
                
                gathered = tent_counts[all_new_sums] + 1
                max_gathered = gathered.max(axis=0)
                valid_mask = max_gathered <= 2
                
                if valid_mask.any():
                    valid_indices = np.where(valid_mask)[0]
                    # Pick the one that creates fewest new tight sums
                    tight_created = (gathered[valid_indices] == 2).sum(axis=0)
                    best_local = valid_indices[np.argmin(tight_created)]
                    x = int(test_candidates[best_local])
                    
                    # Perform the swap
                    current.remove(rem)
                    current.add(x)
                    if rem in candidates:
                        candidates.remove(rem)
                    if rem in removed_pool:
                        removed_pool.remove(rem)
                    else:
                        removed_pool.append(rem)
                    if x in candidates:
                        candidates.remove(x)
                    if x in removed_pool:
                        removed_pool.remove(x)
                    
                    counts = tent_counts
                    new_sums_x = np.concatenate([other_arr + x, np.array([2 * x], dtype=np.int64)])
                    counts[new_sums_x] += 1
                    
                    found_swap = True
                    no_improve = 0
                    break
            
            if found_swap:
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
    DEADLINE_SECONDS = 120.0

    cap = 1000

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 10.0

    best = []
    best_len = -1
    all_greedy_results = []

    # Phase 1: Multiple scored greedy runs with different seeds
    for seed in range(20):
        if _time_left() < 20.0:
            break
        res = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped, seed=seed)
        all_greedy_results.append(res)
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
        if _valid(pat):
            if len(pat) > best_len:
                best_len = len(pat)
                best = pat

    # Phase 4: Local improvement on second-best greedy if time permits
    if _time_left() > 30.0 and len(all_greedy_results) > 1:
        # Sort greedy results by length descending
        all_greedy_results.sort(key=len, reverse=True)
        # Try local improve on the second best if it's close to the best
        if len(all_greedy_results[1]) >= best_len - 5:
            improved2 = _local_improve(all_greedy_results[1], cap=cap, time_fn=time_left, stop_fn=stopped)
            if len(improved2) > best_len:
                best_len = len(improved2)
                best = improved2

    # Phase 5: Final polish - another local improve pass
    if _time_left() > 25.0 and best:
        improved3 = _local_improve(best, cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(improved3) > best_len:
            best_len = len(improved3)
            best = improved3

    # Final validation
    if best and not _valid(best):
        fallback = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped, seed=0)
        if _valid(fallback):
            best = fallback

    return best
