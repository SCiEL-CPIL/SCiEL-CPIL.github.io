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
    Local search: try to add elements, or swap elements to maintain/improve size.
    Optimized: maintain sorted array, use vectorized checks, avoid unnecessary copies.
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
    max_no_improve = 5000
    current_arr = np.sort(np.asarray(list(current), dtype=np.int64))
    
    for it in range(max_iter):
        if time_fn is not None and time_fn() < 8.0:
            break
        if stop_fn is not None and stop_fn():
            break
        
        # Strategy 1: Try to add a random candidate if possible
        if candidates:
            # Pick more candidates to try (up to 20)
            n_try = min(20, len(candidates))
            try_add = rng.sample(candidates, n_try)
            added = False
            
            # Vectorize: check all try_add at once against current_arr
            # For each x in try_add, new_sums = current_arr + x and 2*x
            # We need max(counts[new_sums] + 1) <= 2, i.e., counts[new_sums] <= 1
            
            for x in try_add:
                new_sums = np.concatenate([current_arr + x, np.array([2 * x], dtype=np.int64)])
                if (counts[new_sums] + 1).max() <= 2:
                    current.add(x)
                    candidates.remove(x)
                    counts[new_sums] += 1
                    current_arr = np.sort(np.concatenate([current_arr, np.array([x], dtype=np.int64)]))
                    best_size = len(current)
                    best_set = set(current)
                    no_improve = 0
                    added = True
                    break
            if added:
                continue
        
        # Strategy 2: Try swaps - remove an element, add a candidate
        if current and candidates:
            # Try multiple removals per iteration to find a good swap
            n_removals = min(10, len(current))
            removals = rng.sample(list(current), n_removals)
            found_swap = False
            
            for rem in removals:
                other_mask = current_arr != rem
                other_arr = current_arr[other_mask]
                
                # Calculate sums removed by taking out 'rem'
                removed_sums = np.concatenate([other_arr + rem, np.array([2 * rem], dtype=np.int64)])
                
                # Tentative counts (only modify the relevant entries)
                # Instead of copying the whole array, we'll check validity directly
                # For a candidate x to be valid after removing rem:
                #   counts_after[new_sums_x] = counts[new_sums_x] - overlap + 1 <= 2
                # where overlap = number of removed_sums that equal new_sums_x
                
                # Try more candidates
                n_swap_try = min(100, len(candidates))
                swap_candidates = rng.sample(candidates, n_swap_try)
                
                for x in swap_candidates:
                    if x == rem:
                        continue
                    new_sums = np.concatenate([other_arr + x, np.array([2 * x], dtype=np.int64)])
                    
                    # Check: counts[new_sums] - (removed_sums == new_sums).sum() + 1 <= 2
                    # i.e., counts[new_sums] - overlap <= 1
                    overlap = np.isin(new_sums, removed_sums)
                    effective_counts = counts[new_sums] - overlap.astype(np.int32)
                    
                    if effective_counts.max() <= 1:
                        # Perform swap
                        current.remove(rem)
                        current.add(x)
                        if rem in candidates:
                            candidates.remove(rem)
                        if x in candidates:
                            candidates.remove(x)
                        
                        # Update counts
                        counts[removed_sums] -= 1
                        counts[new_sums] += 1
                        
                        # Update current_arr
                        current_arr = np.sort(np.asarray(list(current), dtype=np.int64))
                        
                        found_swap = True
                        no_improve = 0
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
    
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

    # Phase 1: Multiple scored greedy runs with different seeds
    # The current program does 20 seeds. We keep this but maybe slightly fewer if local search is slow.
    # Let's do 15 seeds to leave more time for local search.
    for seed in range(15):
        if _time_left() < 15.0:
            break
        res = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped, seed=seed)
        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: Local improvement on the best greedy solution
    # This is the key addition from the inspiration.
    if _time_left() > 20.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved
        
        # Try local improvement on a second-best greedy solution if available
        if _time_left() > 40.0:
            # Run another greedy with different seed
            res_alt = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped, seed=99)
            if len(res_alt) >= best_len - 2:  # Close to best
                improved_alt = _local_improve(res_alt, cap=cap, time_fn=time_left, stop_fn=stopped)
                if len(improved_alt) > best_len:
                    best_len = len(improved_alt)
                    best = improved_alt

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
                # If a pattern is better, we could try to improve it too, but let's stick to the plan.

    # Phase 4: Additional greedy + local improve cycles if time permits
    if _time_left() > 50.0:
        for extra_seed in [200, 300]:
            if _time_left() < 25.0:
                break
            res_extra = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped, seed=extra_seed)
            if len(res_extra) >= best_len - 3:
                improved_extra = _local_improve(res_extra, cap=cap, time_fn=time_left, stop_fn=stopped)
                if len(improved_extra) > best_len:
                    best_len = len(improved_extra)
                    best = improved_extra

    # Final validation
    if best and not _valid(best):
        fallback = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped, seed=0)
        if _valid(fallback):
            best = fallback

    return best
