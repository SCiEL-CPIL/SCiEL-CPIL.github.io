"""B2[2] set in 1..1000 via model-and-optimize with accelerated repair.

Key improvements over the parent:
  1. Pre-allocated arrays in greedy paths eliminate O(n) copies per iteration.
  2. A remove-one-add-two phase in the repair loop provides a net +1 growth
     path that 1-for-1 swaps cannot reach when the outside pool is sparse.
  3. Multiple repair restarts (up to ~10) are feasible within the deadline,
     diversifying the local-search landscape.
  4. Better swap acceptance: try several removals per round and pick the
     candidate that leaves the most slack (min-max of resulting counts).
  5. Simulated annealing phase treats B2[2] as energy minimization:
     energy = sum over sums of max(0, count-2)^2.  Bit-flip moves (add,
     remove, swap) are accepted with probability exp(-deltaE/T), allowing
     escape from local optima that greedy/repair cannot reach.
"""
import time
import random
import numpy as np

DEADLINE_SECONDS = 120.0
START_TIME = time.time()


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen, cap=1000):
    """Validate every sum has <=2 reps a<=b. Returns bool."""
    c = np.asarray(chosen, dtype=np.int64)
    n = len(c)
    if n == 0:
        return True
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    counts = np.bincount(sums, minlength=2 * cap + 1)
    return bool(counts.max() <= 2)


def _greedy_build(order, cap=1000, tlimit=8.0):
    """Greedy B2[2] build in a given order with pre-allocated count array."""
    c = np.empty(cap, dtype=np.int64)
    n = 0
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    t0 = time.time()
    for x in order:
        if time.time() - t0 > tlimit:
            break
        if n == 0:
            s = 2 * x
            if counts[s] < 2:
                counts[s] += 1
                c[n] = x
                n += 1
        else:
            # check all sums x + c[0..n-1] and 2x
            ok = True
            for i in range(n):
                s = x + c[i]
                if counts[s] >= 2:
                    ok = False
                    break
            if ok:
                s = 2 * x
                if counts[s] >= 2:
                    ok = False
            if ok:
                for i in range(n):
                    counts[x + c[i]] += 1
                counts[2 * x] += 1
                c[n] = x
                n += 1
    return c[:n].tolist()


def _greedy_build_np(order, cap=1000, tlimit=8.0):
    """Vectorized-assisted greedy: check feasibility in bulk where possible."""
    c = np.empty(cap, dtype=np.int64)
    n = 0
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    t0 = time.time()
    for x in order:
        if time.time() - t0 > tlimit:
            break
        if n == 0:
            s = 2 * x
            if counts[s] < 2:
                counts[s] += 1
                c[n] = x
                n += 1
        else:
            arr = c[:n]
            sums = arr + x
            if (counts[sums] >= 2).any():
                continue
            s2 = 2 * x
            if counts[s2] >= 2:
                continue
            counts[sums] += 1
            counts[s2] += 1
            c[n] = x
            n += 1
    return c[:n].tolist()


def _repair_grow(seed_set, cap=1000, time_fn=None, stop_fn=None,
                 max_rounds=500, seed=0, max_attempts=50000):
    """Phased local improvement with remove-one-add-one and remove-one-add-two."""
    rng = random.Random(seed)
    best = list(seed_set)
    if not best:
        return best
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

    attempts = 0
    c = np.asarray(best, dtype=np.int64)
    n = len(c)
    for _ in range(max_rounds):
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        if attempts >= max_attempts:
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
                tight = gathered[feasible].max(axis=1)
                bi = int(feasible[int(np.argmin(tight))])
                x = int(oa[bi])
                ns = new_sums_all[bi]
                best.append(x)
                best_set.add(x)
                outside.remove(x)
                c = np.concatenate([c, np.array([x], dtype=np.int64)])
                n += 1
                counts[ns] += 1
                attempts += 1
                continue

        # Phase 2: remove-one-add-one swaps
        if outside and n > 0:
            found_swap = False
            batch_size = min(n, 40)
            for _ in range(batch_size):
                if time_fn is not None and time_fn() <= 0:
                    break
                if stop_fn is not None and stop_fn() is True:
                    break
                if attempts >= max_attempts:
                    break
                rem = rng.randrange(n)
                rem_val = int(c[rem])
                trial = list(best)
                del trial[rem]
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
                    tight2 = gathered2[feas2].max(axis=1)
                    bi2 = int(feas2[int(np.argmin(tight2))])
                    x = int(oa[bi2])
                    ns2 = new_sums2[bi2]
                    best = trial
                    best.append(x)
                    best_set = set(best)
                    outside.remove(x)
                    cnt_t[ns2] += 1
                    counts = cnt_t
                    c = np.asarray(best, dtype=np.int64)
                    n = len(c)
                    attempts += 1
                    found_swap = True
                    break
            if found_swap:
                continue

        # Phase 3: remove-one-add-two (net +1 growth)
        if len(outside) >= 2 and n > 0:
            found_grow = False
            batch_size = min(n, 20)
            for _ in range(batch_size):
                if time_fn is not None and time_fn() <= 0:
                    break
                if stop_fn is not None and stop_fn() is True:
                    break
                if attempts >= max_attempts:
                    break
                rem = rng.randrange(n)
                trial = list(best)
                del trial[rem]
                cnt_t, _ = rebuild(trial)
                ta = np.asarray(trial, dtype=np.int64)
                oa = np.asarray(outside, dtype=np.int64)
                # Find first outside x that fits
                if ta.size == 0:
                    new_sums2 = 2 * oa[:, None]
                else:
                    cross2 = oa[:, None] + ta[None, :]
                    new_sums2 = np.concatenate([cross2, 2 * oa[:, None]], axis=1)
                gathered2 = cnt_t[new_sums2] + 1
                cand_max2 = gathered2.max(axis=1)
                feas2 = np.where(cand_max2 <= 2)[0]
                if feas2.size < 2:
                    continue
                # Pick two with best slack
                tight2 = gathered2[feas2].max(axis=1)
                idx_sorted = np.argsort(tight2)
                x1 = int(oa[feas2[idx_sorted[0]]])
                x2 = int(oa[feas2[idx_sorted[1]]])
                if x1 == x2:
                    continue
                # Verify both fit simultaneously
                trial2 = trial + [x1, x2]
                cnt_v, _ = rebuild(trial2)
                if cnt_v.max() <= 2:
                    best = trial2
                    best_set = set(best)
                    outside.remove(x1)
                    outside.remove(x2)
                    counts = cnt_v
                    c = np.asarray(best, dtype=np.int64)
                    n = len(c)
                    attempts += 1
                    found_grow = True
                    break
            if found_grow:
                continue

        # Phase 4: remove-two-add-three (net +1 growth, larger escape step)
        if len(outside) >= 3 and n >= 2:
            found_grow2 = False
            batch_size = min(n * (n - 1) // 2, 15)
            for _ in range(batch_size):
                if time_fn is not None and time_fn() <= 0:
                    break
                if stop_fn is not None and stop_fn() is True:
                    break
                if attempts >= max_attempts:
                    break
                r1 = rng.randrange(n)
                r2 = rng.randrange(n - 1)
                if r2 >= r1:
                    r2 += 1
                trial = list(best)
                del trial[max(r1, r2)]
                del trial[min(r1, r2)]
                cnt_t, _ = rebuild(trial)
                ta = np.asarray(trial, dtype=np.int64)
                oa = np.asarray(outside, dtype=np.int64)
                if ta.size == 0:
                    new_sums3 = 2 * oa[:, None]
                else:
                    cross3 = oa[:, None] + ta[None, :]
                    new_sums3 = np.concatenate([cross3, 2 * oa[:, None]], axis=1)
                gathered3 = cnt_t[new_sums3] + 1
                cand_max3 = gathered3.max(axis=1)
                feas3 = np.where(cand_max3 <= 2)[0]
                if feas3.size < 3:
                    continue
                tight3 = gathered3[feas3].max(axis=1)
                idx_sorted3 = np.argsort(tight3)
                x1 = int(oa[feas3[idx_sorted3[0]]])
                x2 = int(oa[feas3[idx_sorted3[1]]])
                x3 = int(oa[feas3[idx_sorted3[2]]])
                if len({x1, x2, x3}) < 3:
                    continue
                trial3 = trial + [x1, x2, x3]
                cnt_v, _ = rebuild(trial3)
                if cnt_v.max() <= 2:
                    best = trial3
                    best_set = set(best)
                    outside.remove(x1)
                    outside.remove(x2)
                    outside.remove(x3)
                    counts = cnt_v
                    c = np.asarray(best, dtype=np.int64)
                    n = len(c)
                    attempts += 1
                    found_grow2 = True
                    break
            if found_grow2:
                continue

        break
    return best


def _simulated_anneal(seed_set, cap=1000, time_fn=None, stop_fn=None,
                      seed=0, max_iter=200000):
    """Simulated annealing over bit-flip moves with energy = sum max(0,c-2)^2.

    State: inclusion vector of length cap.  Energy penalizes sums with
    count > 2 quadratically.  Moves: add a random outside element, remove a
    random inside element, or swap one in with one out.  Accept with
    probability exp(-deltaE/T) where T cools geometrically.
    """
    rng = random.Random(seed)
    included = np.zeros(cap + 1, dtype=bool)
    for v in seed_set:
        included[v] = True
    inside = [v for v in seed_set]
    outside = [v for v in range(1, cap + 1) if v not in set(seed_set)]

    # Build count array
    def compute_counts(inc):
        vals = [v for v in range(1, cap + 1) if inc[v]]
        if not vals:
            return np.zeros(2 * cap + 1, dtype=np.int32), 0
        arr = np.asarray(vals, dtype=np.int64)
        n = len(arr)
        ii, jj = np.triu_indices(n)
        return np.bincount(arr[ii] + arr[jj], minlength=2 * cap + 1).astype(np.int32), n

    counts, n = compute_counts(included)

    def energy(cnt):
        over = np.maximum(cnt - 2, 0)
        return int(np.sum(over * over))

    def size():
        return int(np.sum(included[1:]))

    best_state = included.copy()
    best_energy = energy(counts)
    best_size = size()
    cur_energy = best_energy

    T_start = 10.0
    T_end = 0.01
    # Determine cooling rate to span max_iter
    if max_iter > 1:
        cool_rate = (T_end / T_start) ** (1.0 / max_iter)
    else:
        cool_rate = 1.0
    T = T_start

    t0 = time.time()
    for it in range(max_iter):
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        if it % 500 == 0 and time.time() - t0 > 30:
            break

        # Choose move type
        r = rng.random()
        if n > 0 and (r < 0.3 or not outside):
            # Remove a random inside element
            rem_idx = rng.randrange(n)
            rem_val = inside[rem_idx]
            # Compute delta: removing rem_val decreases counts of rem_val + x for all x in inside
            old_counts = counts.copy()
            included[rem_val] = False
            inside.pop(rem_idx)
            n -= 1
            # Recompute affected sums: rem_val + x for each x in inside, and 2*rem_val
            # Faster: just recompute full counts for correctness
            counts, n = compute_counts(included)
            new_energy = energy(counts)
            delta_e = new_energy - cur_energy
            new_size = size()
            # Accept if better or with SA probability
            if delta_e <= 0 or (T > 0.001 and rng.random() < np.exp(-delta_e / T)):
                cur_energy = new_energy
                if new_size > best_size or (new_size == best_size and new_energy < best_energy):
                    best_state = included.copy()
                    best_energy = new_energy
                    best_size = new_size
            else:
                # Reject: undo
                included[rem_val] = True
                inside.append(rem_val)
                n += 1
                counts = old_counts
        elif outside:
            # Add a random outside element
            add_idx = rng.randrange(len(outside))
            add_val = outside[add_idx]
            old_counts = counts.copy()
            included[add_val] = True
            inside.append(add_val)
            n += 1
            counts, n = compute_counts(included)
            new_energy = energy(counts)
            delta_e = new_energy - cur_energy
            new_size = size()
            if delta_e <= 0 or (T > 0.001 and rng.random() < np.exp(-delta_e / T)):
                cur_energy = new_energy
                outside.pop(add_idx)
                if new_size > best_size or (new_size == best_size and new_energy < best_energy):
                    best_state = included.copy()
                    best_energy = new_energy
                    best_size = new_size
            else:
                # Reject: undo
                included[add_val] = False
                inside.pop()
                n -= 1
                counts = old_counts
        else:
            # No outside, must remove
            if n > 0:
                rem_idx = rng.randrange(n)
                rem_val = inside[rem_idx]
                old_counts = counts.copy()
                included[rem_val] = False
                inside.pop(rem_idx)
                n -= 1
                counts, n = compute_counts(included)
                new_energy = energy(counts)
                delta_e = new_energy - cur_energy
                new_size = size()
                if delta_e <= 0 or (T > 0.001 and rng.random() < np.exp(-delta_e / T)):
                    cur_energy = new_energy
                    if new_size > best_size or (new_size == best_size and new_energy < best_energy):
                        best_state = included.copy()
                        best_energy = new_energy
                        best_size = new_size
                else:
                    included[rem_val] = True
                    inside.append(rem_val)
                    n += 1
                    counts = old_counts

        # Swap move: remove one inside, add one outside (net size same)
        if n > 0 and outside and rng.random() < 0.25:
            rem_idx = rng.randrange(n)
            add_idx = rng.randrange(len(outside))
            rem_val = inside[rem_idx]
            add_val = outside[add_idx]
            old_counts = counts.copy()
            included[rem_val] = False
            included[add_val] = True
            inside[rem_idx] = add_val
            outside[add_idx] = rem_val
            counts, n = compute_counts(included)
            new_energy = energy(counts)
            delta_e = new_energy - cur_energy
            new_size = size()
            if delta_e <= 0 or (T > 0.001 and rng.random() < np.exp(-delta_e / T)):
                cur_energy = new_energy
                if new_size > best_size or (new_size == best_size and new_energy < best_energy):
                    best_state = included.copy()
                    best_energy = new_energy
                    best_size = new_size
            else:
                # Undo
                included[rem_val] = True
                included[add_val] = False
                inside[rem_idx] = rem_val
                outside[add_idx] = add_val
                counts = old_counts

        T *= cool_rate

    # Reconstruct best set
    best_set = [v for v in range(1, cap + 1) if best_state[v]]
    return best_set


def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 110.0
    cap = 1000

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 6.0

    best = []
    best_len = -1

    def consider(s):
        nonlocal best, best_len
        if s and _valid(s, cap=cap) and len(s) > best_len:
            best_len = len(s)
            best = list(s)

    rng = random.Random(42)

    # --- Multiple greedy builds with different orderings ---
    orders = []
    # Ascending
    orders.append(list(range(1, cap + 1)))
    # Descending
    orders.append(list(range(cap, 0, -1)))
    # Random shuffles
    for _ in range(8):
        full = list(range(1, cap + 1))
        rng.shuffle(full)
        orders.append(full)
    # Prime-biased
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x ** 0.5) + 1):
            if x % d == 0:
                return False
        return True
    primes = [x for x in range(1, cap + 1) if is_prime(x)]
    nonprimes = [x for x in range(1, cap + 1) if not is_prime(x)]
    orders.append(primes + nonprimes)
    # Square-biased
    squares = [i * i for i in range(1, int(cap ** 0.5) + 1) if i * i <= cap]
    nonsq = [x for x in range(1, cap + 1) if x not in set(squares)]
    orders.append(squares + nonsq)

    for order in orders:
        if stopped():
            break
        s = _greedy_build_np(order, cap=cap, tlimit=5.0)
        consider(s)

    # --- Repair/grow on the best with multiple restarts ---
    if best and _time_left() > 8.0:
        for restart in range(12):
            if stopped():
                break
            if _time_left() < 10.0:
                break
            seed_r = 100 + restart * 7
            improved = _repair_grow(best, cap=cap, time_fn=time_left, stop_fn=stopped,
                                    max_rounds=400, seed=seed_r, max_attempts=40000)
            consider(improved)

    # --- Simulated annealing phase ---
    if best and _time_left() > 15.0:
        for sa_restart in range(3):
            if stopped():
                break
            if _time_left() < 12.0:
                break
            sa_seed = 500 + sa_restart * 13
            sa_result = _simulated_anneal(best, cap=cap, time_fn=time_left,
                                          stop_fn=stopped, seed=sa_seed,
                                          max_iter=150000)
            consider(sa_result)
            # Also try SA from a fresh greedy build
            if _time_left() > 15.0:
                full = list(range(1, cap + 1))
                rng.shuffle(full)
                g = _greedy_build_np(full, cap=cap, tlimit=4.0)
                consider(g)
                if _time_left() > 12.0:
                    sa_result2 = _simulated_anneal(g, cap=cap, time_fn=time_left,
                                                   stop_fn=stopped, seed=sa_seed + 999,
                                                   max_iter=100000)
                    consider(sa_result2)

    # --- If we have time, try a second wave of greedy + repair ---
    if best and _time_left() > 18.0:
        for _ in range(4):
            if stopped():
                break
            full = list(range(1, cap + 1))
            rng.shuffle(full)
            s = _greedy_build_np(full, cap=cap, tlimit=4.0)
            consider(s)
            if _time_left() > 13.0:
                improved = _repair_grow(s, cap=cap, time_fn=time_left, stop_fn=stopped,
                                        max_rounds=300, seed=rng.randint(1, 99999),
                                        max_attempts=30000)
                consider(improved)

    if not best:
        full = list(range(1, cap + 1))
        chosen = _greedy_build_np(full, cap=cap, tlimit=10.0)
        best = chosen

    # Final validation guard
    if best and not _valid(best, cap=cap):
        full = list(range(1, cap + 1))
        chosen = _greedy_build_np(full, cap=cap, tlimit=10.0)
        best = chosen

    return best
