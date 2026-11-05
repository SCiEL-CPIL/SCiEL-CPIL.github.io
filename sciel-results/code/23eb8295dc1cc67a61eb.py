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


def _build_counts(chosen):
    """Build the sum-count array for a given chosen set."""
    cap = 1000
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    if not chosen:
        return counts
    c = np.asarray(chosen, dtype=np.int64)
    ii, jj = np.triu_indices(len(c))
    sums = c[ii] + c[jj]
    np.add.at(counts, sums, 1)
    return counts


def _greedy_build(cap=1000, time_fn=None, stop_fn=None, seed=42):
    """Greedy build with scored candidate selection."""
    rng = random.Random(seed)
    pool = set(range(1, cap + 1))
    chosen = []
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    
    while pool:
        if time_fn is not None and time_fn() < 3.0:
            break
        if stop_fn is not None and stop_fn():
            break
        pool_list = list(pool)
        if len(pool_list) <= 200:
            batch = pool_list
        else:
            batch = rng.sample(pool_list, 200)
        arr = np.array(batch, dtype=np.int64)
        if not chosen:
            new_sums = 2 * arr[:, None]
        else:
            chosen_arr = np.asarray(chosen, dtype=np.int64)
            cross = arr[:, None] + chosen_arr[None, :]
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
        if not chosen[:-1]:
            new_sums_x = np.array([2 * x], dtype=np.int64)
        else:
            chosen_arr = np.asarray(chosen[:-1], dtype=np.int64)
            new_sums_x = np.concatenate([chosen_arr + x, np.array([2 * x], dtype=np.int64)])
        counts[new_sums_x] += 1
    return chosen


def _local_improve(chosen, cap=1000, time_fn=None, stop_fn=None, max_iter=20000):
    """Local search with vectorized validity checks."""
    if not chosen:
        return chosen
    
    current_list = list(chosen)
    current_set = set(current_list)
    current_arr = np.array(current_list, dtype=np.int64)
    counts = _build_counts(current_list)
    best_size = len(current_list)
    best_set = set(current_list)
    
    rng = random.Random(123)
    candidates = [x for x in range(1, cap + 1) if x not in current_set]
    
    no_improve = 0
    max_no_improve = 5000
    
    for it in range(max_iter):
        if time_fn is not None and time_fn() < 5.0:
            break
        if stop_fn is not None and stop_fn():
            break
        
        # 1. Try to add a random candidate (growth by 1)
        if candidates and rng.random() < 0.2:
            x = rng.choice(candidates)
            new_sums = np.concatenate([current_arr + x, [2*x]])
            if (counts[new_sums] < 2).all():
                current_list.append(x)
                current_set.add(x)
                current_arr = np.append(current_arr, x)
                counts[new_sums] += 1
                candidates.remove(x)
                best_size = len(current_list)
                best_set = set(current_list)
                no_improve = 0
                continue

        # 2. Try 1-for-1 swap
        if current_list:
            rem = rng.choice(current_list)
            found = False
            for _ in range(5):
                if not candidates:
                    break
                x = rng.choice(candidates)
                if x == rem:
                    continue
                
                mask = current_arr != rem
                others = current_arr[mask]
                
                s_rem_vals = np.concatenate([rem + others, [2*rem]])
                s_x_vals = np.concatenate([x + others, [2*x]])
                
                in_rem = np.isin(s_x_vals, s_rem_vals)
                in_x = np.isin(s_rem_vals, s_x_vals)
                
                new_counts_x = counts[s_x_vals] + 1 - in_rem
                new_counts_rem = counts[s_rem_vals] - 1 + in_x
                
                if new_counts_x.max() <= 2 and new_counts_rem.max() <= 2:
                    counts[s_rem_vals] -= 1
                    counts[s_x_vals] += 1
                    
                    current_list.remove(rem)
                    current_set.remove(rem)
                    current_arr = np.delete(current_arr, np.where(current_arr == rem)[0][0])
                    
                    current_list.append(x)
                    current_set.add(x)
                    current_arr = np.append(current_arr, x)
                    
                    if rem in candidates:
                        candidates.remove(rem)
                    if x in candidates:
                        candidates.remove(x)
                        
                    best_size = len(current_list)
                    best_set = set(current_list)
                    no_improve = 0
                    found = True
                    break
            
            if not found:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

    if len(best_set) > len(current_list):
        return list(best_set)
    return list(current_list)


def _construct_blocks(cap=1000):
    """Structured pattern construction."""
    chosen = []
    for i in range(1, cap + 1):
        if i % 2 == 0 or i % 7 == 0:
            chosen.append(i)
    return chosen


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

    # Phase 1: Multiple greedy builds with different seeds
    for seed in range(15):
        if _time_left() < 15.0:
            break
        res = _greedy_build(cap=cap, time_fn=time_left, stop_fn=stopped, seed=seed)
        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: Local improvement on the best greedy solution
    if _time_left() > 20.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved

    # Phase 3: Try structured patterns
    patterns = [
        list(range(1, cap + 1, 2)),
        [i for i in range(1, cap + 1) if i % 3 != 0],
        [i for i in range(1, cap + 1) if i % 4 in (1, 2)],
    ]
    for pat in patterns:
        if _valid(pat):
            if len(pat) > best_len:
                best_len = len(pat)
                best = pat

    # Phase 4: More local search iterations with different starting points
    if _time_left() > 30.0:
        for seed in range(5):
            if _time_left() < 15.0:
                break
            res = _greedy_build(cap=cap, time_fn=time_left, stop_fn=stopped, seed=100 + seed)
            if len(res) > best_len:
                best_len = len(res)
                best = res
            # Local improve
            if _time_left() > 15.0:
                improved = _local_improve(res, cap=cap, time_fn=time_left, stop_fn=stopped)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best = improved

    # Final validation
    if best and not _valid(best):
        fallback = _greedy_build(cap=cap, time_fn=time_left, stop_fn=stopped, seed=42)
        if _valid(fallback):
            best = fallback

    return best
