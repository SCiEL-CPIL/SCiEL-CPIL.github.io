"""B2[2] set search for {1..1000} using conflict-aware greedy + local search."""
import time
import random
import numpy as np

DEADLINE = 140.0
START = time.time()


def _tl():
    return DEADLINE - (time.time() - START)


def _valid(chosen):
    if not chosen:
        return True
    c = np.asarray(chosen, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    max_s = int(c.max()) * 2
    counts = np.bincount(sums, minlength=max_s + 1)
    return bool(counts.max() <= 2)


def _greedy_fill(chosen, cap=1000, time_fn=None, stop_fn=None):
    """Add all feasible candidates in ascending order."""
    if not chosen:
        return []
    best = list(chosen)
    best_set = set(best)
    c = np.asarray(best, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    counts = np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)
    
    for v in range(1, cap + 1):
        if time_fn and time_fn() <= 0:
            break
        if stop_fn and stop_fn():
            break
        if v in best_set:
            continue
        ns = np.concatenate([c + v, np.array([2 * v], dtype=np.int64)])
        if (counts[ns] + 1).max() <= 2:
            best.append(v)
            best_set.add(v)
            counts[ns] += 1
            c = np.concatenate([c, np.array([v], dtype=np.int64)])
    return best


def _conflict_greedy(cap=1000, time_fn=None, stop_fn=None, rng=None,
                     batch_size=120, prefer_low=False):
    """Conflict-aware greedy: pick candidate that blocks fewest others."""
    if rng is None:
        rng = random.Random()
    pool = set(range(1, cap + 1))
    if prefer_low:
        pool_list = list(range(1, cap + 1))
    else:
        pool_list = list(range(cap, 0, -1))
    
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    
    while pool:
        if time_fn and time_fn() <= 0:
            break
        if stop_fn and stop_fn():
            break
        
        if len(pool) <= batch_size:
            batch_list = list(pool)
        else:
            batch_list = rng.sample(list(pool), batch_size)
        arr = np.array(batch_list, dtype=np.int64)
        
        if chosen_np.size == 0:
            new_sums = 2 * arr[:, None]
            gathered = counts[new_sums] + 1
            cand_max = gathered.max(axis=1)
        else:
            cross = arr[:, None] + chosen_np[None, :]
            new_sums = np.concatenate([cross, 2 * arr[:, None]], axis=1)
            gathered = counts[new_sums] + 1
            cand_max = gathered.max(axis=1)
        
        feasible_idx = np.where(cand_max <= 2)[0]
        if feasible_idx.size == 0:
            for b in batch_list:
                pool.discard(b)
            continue
        
        # Cost: number of sums that go from 1->2 (tightening)
        g = gathered[feasible_idx]
        tighten = np.sum((g == 2) & (g - 1 == 1), axis=1)
        create = np.sum(g == 1, axis=1)
        
        # Prefer candidates that tighten fewer sums and create fewer new sums
        scores = tighten * 8.0 + create * 0.5
        best_local = feasible_idx[int(np.argmin(scores))]
        x = int(arr[best_local])
        
        chosen.append(x)
        pool.discard(x)
        
        if chosen_np.size == 0:
            ns = np.array([2 * x], dtype=np.int64)
        else:
            ns = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
        counts[ns] += 1
    
    return chosen


def _random_greedy(cap=1000, time_fn=None, stop_fn=None, rng=None,
                   order_mode="random"):
    """Greedy with specified ordering."""
    if rng is None:
        rng = random.Random()
    
    if order_mode == "random":
        order = list(range(1, cap + 1))
        rng.shuffle(order)
    elif order_mode == "ascending":
        order = list(range(1, cap + 1))
    elif order_mode == "descending":
        order = list(range(cap, 0, -1))
    elif order_mode == "interleaved":
        lows = list(range(1, cap // 2 + 1))
        highs = list(range(cap // 2 + 1, cap + 1))
        rng.shuffle(lows)
        rng.shuffle(highs)
        order = []
        li = hi = 0
        while li < len(lows) or hi < len(highs):
            if li < len(lows):
                order.append(lows[li]); li += 1
            if hi < len(highs):
                order.append(highs[hi]); hi += 1
    else:
        order = list(range(1, cap + 1))
    
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    
    for x in order:
        if time_fn and time_fn() <= 0:
            break
        if stop_fn and stop_fn():
            break
        if chosen_np.size == 0:
            ns = np.array([2 * x], dtype=np.int64)
        else:
            ns = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        if (counts[ns] + 1).max() <= 2:
            chosen.append(x)
            chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
            counts[ns] += 1
    
    return chosen


def _local_search(chosen, cap=1000, time_fn=None, stop_fn=None,
                  max_rounds=40, rng=None):
    """Local search: remove elements and try to add more."""
    if rng is None:
        rng = random.Random()
    if not chosen:
        return chosen
    
    best = list(chosen)
    best_set = set(best)
    
    def _rebuild_counts(arr):
        n = len(arr)
        if n == 0:
            return np.zeros(2 * cap + 1, dtype=np.int32)
        ii, jj = np.triu_indices(n)
        sums = arr[ii] + arr[jj]
        return np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)
    
    for _ in range(max_rounds):
        if time_fn and time_fn() <= 0:
            break
        if stop_fn and stop_fn():
            break
        
        # Pick 1-2 elements to remove
        k = rng.choice([1, 1, 1, 2])  # mostly single removals
        if len(best) < k:
            break
        remove_idx = set(rng.sample(range(len(best)), k))
        trial = [best[i] for i in range(len(best)) if i not in remove_idx]
        trial_set = set(trial)
        tc = np.asarray(trial, dtype=np.int64)
        tcounts = _rebuild_counts(tc)
        
        # Try to add feasible candidates
        cand_list = [v for v in range(1, cap + 1) if v not in trial_set]
        if not cand_list:
            continue
        ca = np.array(cand_list, dtype=np.int64)
        if tc.size == 0:
            ns_all = 2 * ca[:, None]
        else:
            cross = ca[:, None] + tc[None, :]
            ns_all = np.concatenate([cross, 2 * ca[:, None]], axis=1)
        gathered = tcounts[ns_all] + 1
        feasible = np.where(gathered.max(axis=1) <= 2)[0]
        
        added = 0
        for fi in feasible:
            cand = int(ca[fi])
            if cand in trial_set:
                continue
            if tc.size == 0:
                ns = np.array([2 * cand], dtype=np.int64)
            else:
                ns = np.concatenate([tc + cand, np.array([2 * cand], dtype=np.int64)])
            if (tcounts[ns] + 1).max() <= 2:
                trial.append(cand)
                trial_set.add(cand)
                tcounts[ns] += 1
                tc = np.concatenate([tc, np.array([cand], dtype=np.int64)])
                added += 1
                if time_fn and time_fn() <= 0:
                    break
        
        if len(trial) > len(best):
            best = trial
            best_set = set(best)
    
    return best


def construct():
    global START, DEADLINE
    START = time.time()
    DEADLINE = 140.0
    
    cap = 1000
    best = []
    best_len = -1
    rng = random.Random(42)
    
    def tl():
        return _tl()
    
    def stopped():
        return _tl() < 8.0
    
    # Phase 1: Diverse greedy constructions
    iteration = 0
    while _tl() > 20.0:
        iteration += 1
        mode = iteration % 10
        
        if mode == 0:
            res = _conflict_greedy(cap=cap, time_fn=tl, stop_fn=stopped,
                                   rng=rng, batch_size=120, prefer_low=False)
        elif mode == 1:
            res = _conflict_greedy(cap=cap, time_fn=tl, stop_fn=stopped,
                                   rng=rng, batch_size=120, prefer_low=True)
        elif mode == 2:
            res = _random_greedy(cap=cap, time_fn=tl, stop_fn=stopped,
                                 rng=rng, order_mode="random")
        elif mode == 3:
            res = _random_greedy(cap=cap, time_fn=tl, stop_fn=stopped,
                                 rng=rng, order_mode="interleaved")
        elif mode == 4:
            res = _random_greedy(cap=cap, time_fn=tl, stop_fn=stopped,
                                 rng=rng, order_mode="ascending")
        elif mode == 5:
            res = _random_greedy(cap=cap, time_fn=tl, stop_fn=stopped,
                                 rng=rng, order_mode="descending")
        elif mode == 6:
            res = _conflict_greedy(cap=cap, time_fn=tl, stop_fn=stopped,
                                   rng=rng, batch_size=80, prefer_low=False)
        elif mode == 7:
            res = _random_greedy(cap=cap, time_fn=tl, stop_fn=stopped,
                                 rng=rng, order_mode="random")
        elif mode == 8:
            res = _conflict_greedy(cap=cap, time_fn=tl, stop_fn=stopped,
                                   rng=rng, batch_size=150, prefer_low=True)
        else:
            res = _random_greedy(cap=cap, time_fn=tl, stop_fn=stopped,
                                 rng=rng, order_mode="interleaved")
        
        if len(res) > best_len:
            best_len = len(res)
            best = res
    
    # Phase 2: Greedy fill on best
    if best and _tl() > 12.0:
        filled = _greedy_fill(best, cap=cap, time_fn=tl, stop_fn=stopped)
        if len(filled) > best_len:
            best = filled
            best_len = len(best)
    
    # Phase 3: Local search
    if best and _tl() > 10.0:
        improved = _local_search(best, cap=cap, time_fn=tl, stop_fn=stopped,
                                  max_rounds=50, rng=rng)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)
    
    # Phase 4: Another greedy fill + local search
    if best and _tl() > 8.0:
        filled2 = _greedy_fill(best, cap=cap, time_fn=tl, stop_fn=stopped)
        if len(filled2) > best_len:
            best = filled2
            best_len = len(best)
        improved2 = _local_search(best, cap=cap, time_fn=tl, stop_fn=stopped,
                                   max_rounds=30, rng=rng)
        if len(improved2) > best_len:
            best = improved2
            best_len = len(best)
    
    # Final validation
    if best and not _valid(best):
        simple = _random_greedy(cap=cap, order_mode="ascending")
        if _valid(simple) and len(simple) > 0:
            best = simple
    
    return best
