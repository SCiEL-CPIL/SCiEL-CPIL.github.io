"""Stronger B2[2] set search for {1..1000}.

Combined strategy:
  - Fast vectorized greedy construction with multiple diverse orderings.
  - A score-driven greedy that picks the best feasible candidate each step.
  - Local improvement via single-swap (remove one, add one or more).
  - Portfolio of randomized runs within a strict time budget.
  - Deadline guard with safety margin.

Improvements over seed:
  - Pre-allocated arrays in greedy to eliminate O(n) copies per iteration.
  - "Remove one, add two" phase for net +1 growth when outside pool is sparse.
  - Vectorized feasibility checks in repair phases.
  - Model-based seeding (Sidon base + 2-AP expansion) for better initial sets.
"""
import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0
START_TIME = time.time()

# Soft-max temperature for the scored greedy (lower = sharper argmin)
SOFTMAX_T = 0.35


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen, cap=1000):
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
    """Greedy: iterate candidates in order; add if B2[2] preserved.
    Uses pre-allocated array to avoid repeated np.concatenate."""
    chosen = []
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    # Pre-allocate max possible size
    cnp = np.empty(len(order), dtype=np.int64)
    n = 0
    for x in order:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break
        if n == 0:
            ns = np.array([2 * x], dtype=np.int64)
        else:
            ns = np.concatenate([cnp[:n] + x, np.array([2 * x], dtype=np.int64)])
        if (counts[ns] + 1).max() <= 2:
            cnp[n] = x
            n += 1
            counts[ns] += 1
            chosen.append(x)
    return chosen


def _scored_greedy(cap=1000, time_fn=None, stop_fn=None, rng=None,
                   batch_size=150, score_mode="min_tightness",
                   softmax_t=SOFTMAX_T):
    """Score-driven greedy: at each step, evaluate a batch of random candidates
    and pick the one that minimizes a soft-max surrogate of the resulting
    maximum count (tightness)."""
    if rng is None:
        rng = random.Random()
    pool = set(range(1, cap + 1))
    chosen = []
    cnp = np.empty(len(pool), dtype=np.int64)
    n = 0
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
        if n == 0:
            new_sums = 2 * arr[:, None]
        else:
            cross = arr[:, None] + cnp[:n][None, :]
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

        # Smooth surrogate: soft-max over the per-candidate count vector.
        g = gathered[feasible]
        gmax = g.max(axis=1, keepdims=True)
        lse = gmax + np.log(np.sum(np.exp(g - gmax), axis=1, keepdims=True))
        scores = lse[:, 0] / t
        best_local = feasible[int(np.argmin(scores))]
        x = int(arr[best_local])
        chosen.append(x)
        pool.discard(x)

        if n == 0:
            ns = np.array([2 * x], dtype=np.int64)
        else:
            ns = np.concatenate([cnp[:n] + x, np.array([2 * x], dtype=np.int64)])
        cnp[n] = x
        n += 1
        counts[ns] += 1

    return chosen


def _local_improve(chosen, cap=1000, time_fn=None, stop_fn=None, max_iters=20, rng=None):
    """Try single-swap improvements: remove one element, try to add better ones.
    Includes remove-one-add-two phase for net +1 growth."""
    if rng is None:
        rng = random.Random()
    if not chosen:
        return chosen
    best = list(chosen)
    best_set = set(best)
    outside = [v for v in range(1, cap + 1) if v not in best_set]

    c = np.asarray(best, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    counts = np.bincount(c[ii] + c[jj], minlength=2 * cap + 1).astype(np.int32)

    def rebuild(ch):
        cc = np.asarray(ch, dtype=np.int64)
        nn = len(cc)
        if nn == 0:
            return np.zeros(2 * cap + 1, dtype=np.int32), nn
        i2, j2 = np.triu_indices(nn)
        return np.bincount(cc[i2] + cc[j2], minlength=2 * cap + 1).astype(np.int32), nn

    for _ in range(max_iters):
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break

        # Phase 1: pure adds
        if outside:
            oa = np.asarray(outside, dtype=np.int64)
            if n == 0:
                new_sums_all = 2 * oa[:, None]
            else:
                cross = oa[:, None] + c[None, :]
                new_sums_all = np.concatenate([cross, 2 * oa[:, None]], axis=1)
            gathered = counts[new_sums_all] + 1
            cand_max = gathered.max(axis=1)
            feasible = np.where(cand_max <= 2)[0]
            if feasible.size:
                tight = gathered[feasible].mean(axis=1)
                bi = int(feasible[int(np.argmin(tight))])
                x = int(oa[bi])
                ns = new_sums_all[bi]
                best.append(x)
                best_set.add(x)
                outside.remove(x)
                c = np.concatenate([c, np.array([x], dtype=np.int64)])
                n += 1
                counts[ns] += 1
                continue

        # Phase 2: targeted swaps — remove one element, try to add an outside candidate
        if outside and n > 0:
            batch_size = min(n, 30)
            found_swap = False
            for _ in range(batch_size):
                if time_fn is not None and time_fn() <= 0:
                    break
                if stop_fn is not None and stop_fn():
                    break
                rem = rng.randrange(n)
                rem_val = int(c[rem])
                trial = [best[j] for j in range(n) if j != rem]
                cnt_t, _ = rebuild(trial)
                ta = np.asarray(trial, dtype=np.int64)
                oa = np.asarray(outside, dtype=np.int64)
                if ta.size == 0:
                    new_sums2 = 2 * oa[:, None]
                else:
                    cross2 = oa[:, None] + ta[None, :]
                    new_sums2 = np.concatenate([cross2, 2 * oa[:, None]], axis=1)
                gathered2 = cnt_t[new_sums2] + 1
                cand_max2 = gathered2.max(axis=1)
                feas2 = np.where(cand_max2 <= 2)[0]
                if feas2.size:
                    tight2 = gathered2[feas2].mean(axis=1)
                    bi2 = int(feas2[int(np.argmin(tight2))])
                    x = int(oa[bi2])
                    ns2 = new_sums2[bi2]
                    best = trial
                    best.append(x)
                    best_set = set(best)
                    best_set.add(x)
                    outside.remove(x)
                    cnt_t[ns2] += 1
                    counts = cnt_t
                    c = np.asarray(best, dtype=np.int64)
                    n = len(c)
                    found_swap = True
                    break
            if found_swap:
                continue

        # Phase 3: remove one, add two (net +1)
        if len(outside) >= 2 and n > 0:
            batch_size = min(n, 20)
            found_double = False
            for _ in range(batch_size):
                if time_fn is not None and time_fn() <= 0:
                    break
                if stop_fn is not None and stop_fn():
                    break
                rem = rng.randrange(n)
                rem_val = int(c[rem])
                trial = [best[j] for j in range(n) if j != rem]
                cnt_t, _ = rebuild(trial)
                ta = np.asarray(trial, dtype=np.int64)
                oa = np.asarray(outside, dtype=np.int64)
                if ta.size == 0:
                    new_sums2 = 2 * oa[:, None]
                else:
                    cross2 = oa[:, None] + ta[None, :]
                    new_sums2 = np.concatenate([cross2, 2 * oa[:, None]], axis=1)
                gathered2 = cnt_t[new_sums2] + 1
                cand_max2 = gathered2.max(axis=1)
                feas2 = np.where(cand_max2 <= 2)[0]
                if feas2.size >= 2:
                    # Try to add two feasible candidates
                    tight2 = gathered2[feas2].mean(axis=1)
                    order2 = np.argsort(tight2)
                    added = 0
                    new_best = list(trial)
                    new_set = set(new_best)
                    new_counts = cnt_t.copy()
                    new_ta = ta.copy()
                    for idx in order2:
                        x = int(oa[int(idx)])
                        if x in new_set:
                            continue
                        if new_ta.size == 0:
                            ns = np.array([2 * x], dtype=np.int64)
                        else:
                            ns = np.concatenate([new_ta + x, np.array([2 * x], dtype=np.int64)])
                        if (new_counts[ns] + 1).max() <= 2:
                            new_best.append(x)
                            new_set.add(x)
                            new_counts[ns] += 1
                            new_ta = np.concatenate([new_ta, np.array([x], dtype=np.int64)])
                            added += 1
                            if added >= 2:
                                break
                    if added >= 2 and len(new_best) > len(best):
                        best = new_best
                        best_set = set(best)
                        outside = [v for v in range(1, cap + 1) if v not in best_set]
                        c = np.asarray(best, dtype=np.int64)
                        n = len(c)
                        ii, jj = np.triu_indices(n)
                        counts = np.bincount(c[ii] + c[jj], minlength=2 * cap + 1).astype(np.int32)
                        found_double = True
                        break
            if found_double:
                continue

    # Multi-addition polish: repeatedly scan for any feasible addition
    c = np.asarray(best, dtype=np.int64)
    n = len(c)
    if n > 0:
        ii, jj = np.triu_indices(n)
        sums = c[ii] + c[jj]
        counts = np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)
    else:
        counts = np.zeros(2 * cap + 1, dtype=np.int32)

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

    return best


def _greedy_sidon_base(seed, cap=1000, tlimit=3.0):
    """Greedy Sidon base: pick integers in 1..cap so all pairwise sums (a<=b) are unique."""
    rng = random.Random(seed)
    chosen = []
    cnp = np.empty(cap, dtype=np.int64)
    n = 0
    seen = set()
    full = list(range(1, cap + 1))
    rng.shuffle(full)
    t0 = time.time()
    for x in full:
        if time.time() - t0 > tlimit:
            break
        if n == 0:
            new_sums = [2 * x]
        else:
            new_sums = list(cnp[:n] + x) + [2 * x]
        if all(s not in seen for s in new_sums):
            seen.update(new_sums)
            cnp[n] = x
            n += 1
            chosen.append(x)
    return chosen


def _build_model(base, B0, B1, cap=1000):
    """Build candidate set {a + b : a in base, b in B0..B1} within 1..cap, dedup."""
    out = set()
    for a in base:
        for b in range(B0, B1 + 1):
            v = a + b
            if 1 <= v <= cap:
                out.add(v)
    return sorted(out)


def _seed_set_from_model(base, B0, B1, cap=1000, order="asc"):
    """Greedy-select from the model's candidate pool to enforce B2[2] exactly."""
    pool = _build_model(base, B0, B1, cap=cap)
    if order == "desc":
        pool = pool[::-1]
    chosen = []
    cnp = np.empty(len(pool), dtype=np.int64)
    n = 0
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    for x in pool:
        if n == 0:
            ns = np.array([2 * x], dtype=np.int64)
        else:
            ns = np.concatenate([cnp[:n] + x, np.array([2 * x], dtype=np.int64)])
        if (counts[ns] + 1).max() <= 2:
            cnp[n] = x
            n += 1
            counts[ns] += 1
            chosen.append(x)
    return chosen


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

    # Phase 0: Model-based seeding for better initial sets
    if _time_left() > 30.0:
        for seed in [42, 123, 777]:
            if _time_left() < 20.0:
                break
            base = _greedy_sidon_base(seed, cap=cap, tlimit=2.0)
            if len(base) >= 10:
                for B0, B1 in [(1, 3), (2, 5), (1, 5)]:
                    if _time_left() < 15.0:
                        break
                    model_set = _seed_set_from_model(base, B0, B1, cap=cap)
                    if len(model_set) > best_len:
                        best_len = len(model_set)
                        best = model_set

    # Phase 1: portfolio of diverse greedy constructions
    iteration = 0
    while _time_left() > 12.0:
        iteration += 1
        mode = iteration % 6

        if mode == 0:
            order = full[:]
            rng.shuffle(order)
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 1:
            # Scored greedy (soft-max surrogate)
            res = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped,
                                 rng=rng, batch_size=200, softmax_t=SOFTMAX_T)
        elif mode == 2:
            # Rotated ascending
            start = rng.randrange(cap)
            order = list(range(start + 1, cap + 1)) + list(range(1, start + 1))
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 3:
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
        elif mode == 4:
            # Descending
            order = full[::-1]
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        else:
            # Model-based with different parameters
            base = _greedy_sidon_base(iteration * 13 + 7, cap=cap, tlimit=1.5)
            if len(base) >= 8:
                res = _seed_set_from_model(base, 1, 4, cap=cap)
            else:
                order = full[:]
                rng.shuffle(order)
                res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)

        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: local improvement (single-swap + multi-addition polish)
    if _time_left() > 8.0 and best:
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
