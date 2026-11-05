"""B2[2] set construction for {1, ..., 1000}.

Strategy:
  - Fast vectorized greedy construction with a scoring heuristic.
  - Portfolio: run many diverse randomized constructions within the time budget.
  - Light local improvement (swap-based) on the best result if time remains.
  - Explicit deadline guard with safety margin; return best found.
"""
import time
import random
import numpy as np

# ---- Configuration / deadline guard ----
# Evaluator kills at 150s wall-clock. Keep a safety margin so we never get killed.
DEADLINE_SECONDS = 120.0
START_TIME = time.time()


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen):
    """Validate that every sum has at most two representations a<=b. Returns bool."""
    n = len(chosen)
    if n == 0:
        return True
    c = np.asarray(chosen, dtype=np.int64)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    max_s = int(c.max()) * 2
    counts = np.bincount(sums, minlength=max_s + 1)
    return bool(counts.max() <= 2)


def _greedy_build_with_score(candidates, cap=1000, time_fn=None, stop_fn=None,
                             score_mode="minmax", batch=200, rng=None):
    """Score-driven greedy: at each step pick the best candidate (by score) among
    the remaining pool, using a vectorized batch check. Returns chosen list.

    score_mode:
      'minmax'  : minimize the resulting max count (then minimize new collisions)
      'rand'    : random pick (fallback)
    """
    if rng is None:
        rng = random
    pool = list(candidates)
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    
    while pool:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        
        # Take a batch of candidates to evaluate in parallel for speed.
        if len(pool) <= batch:
            batch_list = pool[:]
            evaluate_all = True
        else:
            batch_list = rng.sample(pool, batch)
            evaluate_all = False
        
        arr = np.array(batch_list, dtype=np.int64)
        
        # Compute for each candidate the vector of new sums and the resulting max
        # count. Use broadcasting: arr[:,None] + chosen_np[None,:] and 2*arr.
        if chosen_np.size == 0:
            new_sums = 2 * arr[:, None]  # shape (m,1)
        else:
            cross = arr[:, None] + chosen_np[None, :]  # (m, k)
            new_sums = np.concatenate([cross, 2 * arr[:, None]], axis=1)  # (m, k+1)
        
        # For each candidate m, we need max over its new sums of (counts[s]+1)
        # counts is (2cap+1). Gather counts[new_sums] -> (m, k+1)
        gathered = counts[new_sums] + 1  # (m, k+1)
        cand_max = gathered.max(axis=1)  # (m,)
        feasible = np.where(cand_max <= 2)[0]
        
        if feasible.size == 0:
            if evaluate_all:
                break  # no candidate in entire pool feasible -> stop
            else:
                # none in this batch feasible; remove them from pool and continue
                # (they are infeasible right now; they might become feasible later?
                #  Adding more elements only increases counts, so if infeasible now,
                #  it stays infeasible. So we can drop them.)
                for b in batch_list:
                    try:
                        pool.remove(b)
                    except ValueError:
                        pass
                continue
        
        # Among feasible, pick the best by secondary score: minimize the number of
        # new sums that hit an already-counted slot (i.e., that will have count>=2),
        # which keeps room. Use mean of gathered as a proxy for "how tight".
        tight = gathered[feasible].mean(axis=1)
        best_local = feasible[int(rng.randrange(len(feasible)) if score_mode == "rand" else np.argmin(tight))]
        x = int(arr[best_local])
        
        # Accept x
        chosen.append(x)
        pool.remove(x)
        
        if chosen_np.size == 0:
            new_sums_x = np.array([2 * x], dtype=np.int64)
        else:
            new_sums_x = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
        counts[new_sums_x] += 1
    
    return chosen


def _local_improve(chosen, time_fn=None, stop_fn=None, max_iters=200, rng=None):
    """Attempt to improve the set by swapping elements or adding new ones after removals.
    
    Strategy:
      1. Try to add any candidate not in set directly.
      2. If none, try single removals followed by greedy additions to see if net size increases.
    """
    if not chosen:
        return chosen
    if rng is None:
        rng = random
    
    chosen_set = set(chosen)
    best = list(chosen)
    cap = 1000
    candidates = [v for v in range(1, cap + 1) if v not in chosen_set]
    
    # Precompute counts for current set
    c = np.asarray(chosen, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    counts = np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)
    
    def rebuild_counts(ch):
        cc = np.asarray(ch, dtype=np.int64)
        nn = len(cc)
        if nn == 0:
            return np.zeros(2 * cap + 1, dtype=np.int32)
        i2, j2 = np.triu_indices(nn)
        s2 = cc[i2] + cc[j2]
        return np.bincount(s2, minlength=2 * cap + 1).astype(np.int32)
    
    # First pass: try to add any candidate directly
    for it in range(max_iters):
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        
        if not candidates:
            break
        
        # Try to find a candidate that can be added
        arr = np.array(candidates, dtype=np.int64)
        if len(arr) > 100:
            arr = np.array(rng.sample(candidates, 100), dtype=np.int64)
        
        if len(best) == 0:
            new_sums = 2 * arr[:, None]
        else:
            best_np = np.asarray(best, dtype=np.int64)
            cross = arr[:, None] + best_np[None, :]
            new_sums = np.concatenate([cross, 2 * arr[:, None]], axis=1)
        
        gathered = counts[new_sums] + 1
        cand_max = gathered.max(axis=1)
        feasible = np.where(cand_max <= 2)[0]
        
        if feasible.size > 0:
            best_local = feasible[0]
            x = int(arr[best_local])
            best.append(x)
            best_set = set(best)
            candidates = [v for v in candidates if v not in best_set]
            counts = rebuild_counts(best)
            continue
        
        # If no direct addition possible, try swap-based improvement
        # Pick a random element to try removing
        idx = rng.randrange(len(best))
        removed = best[idx]
        trial = [best[j] for j in range(len(best)) if j != idx]
        counts_t = rebuild_counts(trial)
        
        # Try to add multiple elements after removal
        added = []
        trial_set = set(trial)
        trial_np = np.asarray(trial, dtype=np.int64)
        
        for cand in candidates[:50]:
            if cand in trial_set:
                continue
            if len(trial) == 0:
                new_sums_c = np.array([2 * cand], dtype=np.int64)
            else:
                new_sums_c = np.concatenate([trial_np + cand, np.array([2 * cand], dtype=np.int64)])
            
            gathered_c = counts_t[new_sums_c] + 1
            if gathered_c.max() <= 2:
                trial.append(cand)
                trial_set.add(cand)
                trial_np = np.asarray(trial, dtype=np.int64)
                counts_t[new_sums_c] += 1
                added.append(cand)
        
        if len(trial) > len(best):
            best = trial
            best_set = set(best)
            candidates = [v for v in candidates if v not in best_set]
            counts = rebuild_counts(best)
        else:
            # Revert: do nothing, just continue
            pass
    
    return best


def construct():
    """Construct a B2[2] set in {1, ..., 1000} maximizing size."""
    best_result = []
    rng = random.Random(42)
    
    # Strategy: run multiple diverse greedy constructions with different orderings
    # and scoring modes, then take the best result.
    
    # First, try a few different orderings
    orderings = [
        list(range(1, 1001)),                    # ascending
        list(range(1000, 0, -1)),                # descending
        list(range(1, 1001, 2)),                 # odd first
        list(range(2, 1001, 2)),                 # even first
        list(rng.sample(range(1, 1001), 1000)),  # random
        list(rng.sample(range(1, 1001), 1000)),  # random
    ]
    
    # Add some structured orderings
    # Primes-ish: numbers with few divisors
    def num_divisors(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c
    
    few_div = sorted(range(1, 1001), key=lambda x: num_divisors(x))
    orderings.append(few_div)
    
    # Try each ordering with different scoring modes
    for order in orderings:
        if _time_left() < 5.0:
            break
        
        for score_mode in ["minmax", "rand"]:
            if _time_left() < 3.0:
                break
            
            result = _greedy_build_with_score(
                order, cap=1000, time_fn=_time_left, stop_fn=None,
                score_mode=score_mode, batch=200, rng=rng
            )
            
            if len(result) > len(best_result):
                best_result = result
    
    # Local improvement on the best result
    if _time_left() > 10.0 and len(best_result) > 0:
        improved = _local_improve(
            best_result, time_fn=_time_left, stop_fn=None,
            max_iters=200, rng=rng
        )
        if len(improved) > len(best_result):
            best_result = improved
    
    # Final validation
    if not _valid(best_result):
        # Fallback to a simple greedy if validation fails
        chosen, counts = [], {}
        for x in range(1, 1001):
            new = [x + y for y in chosen] + [2 * x]
            trial = dict(counts)
            for s in new:
                trial[s] = trial.get(s, 0) + 1
            if max(trial[s] for s in new) <= 2:
                chosen.append(x)
                counts = trial
        best_result = chosen
    
    return best_result
