"""Stronger B2[2] set search for {1..1000}.

Strategy:
  - Conflict-degree greedy: model as dynamic independent set, pick candidate
    that minimizes the number of other candidates it blocks.
  - Score-driven greedy with soft-max surrogate.
  - Local improvement via targeted single-swap and multi-swap.
  - Portfolio of randomized runs within a strict time budget.
  - Deadline guard with safety margin.
"""
import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0
START_TIME = time.time()

# Soft-max temperature for the scored greedy (lower = sharper argmin)
SOFTMAX_T = 0.35
# Weight for tightness (number of sums with count==1) in composite score
TIGHTNESS_W = 0.15


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen):
    """Validate B2[2]: every sum has at most two representations a<=b."""
    if not chosen:
        return True
    c = np.asarray(chosen, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    max_s = int(c.max()) * 2
    counts = np.bincount(sums, minlength=max_s + 1)
    return bool(counts.max() <= 2)


def _greedy_build(order, cap=1000, time_fn=None, stop_fn=None):
    """Greedy: iterate candidates in order; add if B2[2] preserved."""
    chosen = []
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    chosen_np = np.empty(0, dtype=np.int64)
    for x in order:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break
        if chosen_np.size == 0:
            new_sums = np.array([2 * x], dtype=np.int64)
        else:
            new_sums = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        new_counts = counts[new_sums] + 1
        if new_counts.max() <= 2:
            chosen.append(x)
            chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
            counts[new_sums] += 1
    return chosen


def _conflict_greedy(cap=1000, time_fn=None, stop_fn=None, rng=None,
                     batch_size=100, mode="min_conflict"):
    """Conflict-degree greedy: at each step, evaluate a batch of feasible
    candidates and pick the one that blocks the fewest other candidates.
    This models the dynamic independent set problem."""
    if rng is None:
        rng = random.Random()
    pool = set(range(1, cap + 1))
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)

    while pool:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break

        # Get a batch of candidates to evaluate
        if len(pool) <= batch_size:
            batch_list = list(pool)
        else:
            batch_list = rng.sample(list(pool), batch_size)
        arr = np.array(batch_list, dtype=np.int64)

        # Check feasibility for each candidate
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
            # No feasible candidates in this batch; remove them from pool
            for b in batch_list:
                pool.discard(b)
            continue

        # For each feasible candidate, compute how many OTHER candidates it would block
        # A candidate x blocks candidate z if adding x makes z infeasible
        # i.e., there exists y in chosen such that c[x+y+z] would exceed... 
        # Actually: x blocks z if after adding x, z becomes infeasible.
        # z is infeasible after adding x if for some y in chosen∪{x}, c'[z+y] > 2
        # where c' is counts after adding x.
        # Simpler: x blocks z if there exists y in chosen such that 
        #   counts[x+y] == 2 (so adding x fills that sum) AND 
        #   z + y == x + y ... no that's not right.
        # 
        # Let me think again. After adding x, the new counts are:
        #   c'[s] = c[s] + 1 if s = x + y for some y in chosen, or s = 2x
        # z is infeasible after adding x if for some y in chosen∪{x}, c'[z+y] > 2
        # i.e., c[z+y] + (1 if z+y is in {x+y' : y' in chosen} ∪ {2x}) > 2
        #
        # For a pair (x, z), x blocks z if there exists y in chosen such that:
        #   c[z+y] == 2 AND (z+y) is in the set of sums affected by adding x
        # The sums affected by adding x are {x+y' : y' in chosen} ∪ {2x}
        # So x blocks z if ∃ y' in chosen such that z + y = x + y' AND c[z+y] == 2
        # i.e., z - y' = x - y ... this is getting complex.
        #
        # Simpler heuristic: x blocks z if they share a "critical" sum.
        # A sum s is critical if c[s] == 2. Adding x makes sum s+x-y critical...
        #
        # Let's use a simpler proxy: for each feasible x, count how many z in the
        # batch would become infeasible. We can check this directly.
        feasible_candidates = arr[feasible_idx]
        
        # For efficiency, just pick the feasible candidate with minimum "impact"
        # Impact = number of sums that go from 0 to 1 (creates new constraints)
        # Actually, let's use: impact = number of sums that go from 1 to 2 (tightens)
        g = gathered[feasible_idx]
        # Sums that were 1 and become 2
        tighten = np.sum((g == 2) & (g - 1 == 1), axis=1)  # sums that become 2
        # Sums that were 0 and become 1
        create = np.sum((g == 1), axis=1)
        
        # We want to minimize tightening (sums going 1->2) and creating (0->1)
        # But creating is necessary; tightening is bad
        # Score: primarily minimize tighten, secondarily minimize create
        scores = tighten * 10 + create
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


def _scored_greedy(cap=1000, time_fn=None, stop_fn=None, rng=None,
                   batch_size=150, score_mode="min_tightness",
                   softmax_t=SOFTMAX_T):
    """Score-driven greedy: at each step, evaluate a batch of random candidates
    and pick the one that minimizes a composite score combining soft-max of
    max count with a tightness penalty. Lower scores are better."""
    if rng is None:
        rng = random.Random()
    pool = set(range(1, cap + 1))
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    t = float(softmax_t) if softmax_t and softmax_t > 0 else 1e-9

    while pool:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break
        if len(pool) <= batch_size:
            batch_list = list(pool)
            evaluate_all = True
        else:
            batch_list = rng.sample(list(pool), batch_size)
            evaluate_all = False

        arr = np.array(batch_list, dtype=np.int64)
        if chosen_np.size == 0:
            new_sums = 2 * arr[:, None]
        else:
            cross = arr[:, None] + chosen_np[None, :]
            new_sums = np.concatenate([cross, 2 * arr[:, None]], axis=1)

        gathered = counts[new_sums] + 1
        cand_max = gathered.max(axis=1)
        feasible = np.where(cand_max <= 2)[0]

        if feasible.size == 0:
            if evaluate_all:
                break
            else:
                for b in batch_list:
                    pool.discard(b)
                continue

        # Composite score: soft-max of max count + tightness penalty
        g = gathered[feasible]
        gmax = g.max(axis=1, keepdims=True)
        # Numerically stable log-sum-exp for soft-max
        lse = gmax + np.log(np.sum(np.exp(g - gmax), axis=1, keepdims=True))
        soft_max_score = lse[:, 0] / t

        # Tightness: count of sums that would become count==1 (more flexibility)
        # Actually we want to AVOID creating too many count==1 sums, so penalty
        # is the number of sums that go from 0 to 1
        tightness = np.sum(g == 1, axis=1) * TIGHTNESS_W

        scores = soft_max_score + tightness
        best_local = feasible[int(np.argmin(scores))]
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


def _local_improve(chosen, cap=1000, time_fn=None, stop_fn=None, max_iters=30, rng=None):
    """Targeted local improvement: add-first, then targeted swaps."""
    if rng is None:
        rng = random.Random()
    if not chosen:
        return chosen
    best = list(chosen)
    best_set = set(best)

    def _recompute_counts(arr):
        n = len(arr)
        if n > 0:
            ii, jj = np.triu_indices(n)
            sums = arr[ii] + arr[jj]
            return np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)
        return np.zeros(2 * cap + 1, dtype=np.int32)

    # Phase 0: greedy fill to local maximum (add all feasible)
    c = np.asarray(best, dtype=np.int64)
    counts = _recompute_counts(c)
    
    progress = True
    while progress and (time_fn is None or time_fn() > 0) and (stop_fn is None or not stop_fn()):
        progress = False
        cand_list = [v for v in range(1, cap + 1) if v not in best_set]
        if not cand_list:
            break
        ca = np.array(cand_list, dtype=np.int64)
        if c.size == 0:
            ns_all = 2 * ca[:, None]
        else:
            cross = ca[:, None] + c[None, :]
            ns_all = np.concatenate([cross, 2 * ca[:, None]], axis=1)
        gathered = counts[ns_all] + 1
        feasible = np.where(gathered.max(axis=1) <= 2)[0]
        if feasible.size == 0:
            break
        for fi in feasible:
            cand = int(ca[fi])
            if cand in best_set:
                continue
            if c.size == 0:
                ns = np.array([2 * cand], dtype=np.int64)
            else:
                ns = np.concatenate([c + cand, np.array([2 * cand], dtype=np.int64)])
            if (counts[ns] + 1).max() <= 2:
                best.append(cand)
                best_set.add(cand)
                counts[ns] += 1
                c = np.concatenate([c, np.array([cand], dtype=np.int64)])
                progress = True
                if time_fn is not None and time_fn() <= 0:
                    break
            if stop_fn is not None and stop_fn():
                break

    # Phase A: targeted single-swap (remove one, try to add more)
    for _ in range(max_iters):
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break
        idx = rng.randrange(len(best))
        removed = best[idx]
        trial = [best[j] for j in range(len(best)) if j != idx]
        trial_set = set(trial)
        tc = np.asarray(trial, dtype=np.int64)
        tcounts = _recompute_counts(tc)
        
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
                if time_fn is not None and time_fn() <= 0:
                    break
        
        if len(trial) > len(best):
            best = trial
            best_set = set(best)

    # Phase B: multi-swap (remove 2, try to add more)
    if time_fn is None or time_fn() > 5.0:
        for _ in range(15):
            if time_fn is not None and time_fn() <= 0:
                break
            if stop_fn is not None and stop_fn():
                break
            if len(best) < 2:
                break
            i, j = rng.sample(range(len(best)), 2)
            trial = [best[k] for k in range(len(best)) if k not in (i, j)]
            trial_set = set(trial)
            tc = np.asarray(trial, dtype=np.int64)
            tcounts = _recompute_counts(tc)
            
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
                    if time_fn is not None and time_fn() <= 0:
                        break
            
            if len(trial) > len(best):
                best = trial
                best_set = set(best)

    return best


def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 115.0

    cap = 1000
    full = list(range(1, cap + 1))
    best = []
    best_len = -1
    rng = random.Random(42)

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 5.0

    # Phase 1: portfolio of diverse greedy constructions
    iteration = 0
    while _time_left() > 15.0:
        iteration += 1
        mode = iteration % 7

        if mode == 0:
            # Conflict-degree greedy (primary)
            res = _conflict_greedy(cap=cap, time_fn=time_left, stop_fn=stopped,
                                   rng=rng, batch_size=100)
        elif mode == 1:
            # Conflict-degree greedy with larger batch
            res = _conflict_greedy(cap=cap, time_fn=time_left, stop_fn=stopped,
                                   rng=rng, batch_size=200)
        elif mode == 2:
            # Scored greedy (composite score)
            res = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped,
                                 rng=rng, batch_size=200, softmax_t=SOFTMAX_T)
        elif mode == 3:
            # Random order greedy
            order = full[:]
            rng.shuffle(order)
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 4:
            # Interleaved low/high
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
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 5:
            # Descending
            order = full[::-1]
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        else:
            # Ascending
            order = full[:]
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)

        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: local improvement (single-swap + multi-swap + greedy fill)
    if _time_left() > 10.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left,
                                  stop_fn=stopped, max_iters=30, rng=rng)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)

    # Final validation
    if best and not _valid(best):
        simple = _greedy_build(full, cap=cap)
        if _valid(simple) and len(simple) > 0:
            best = simple

    return best
