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
            # Quick check: 2*x first (cheapest)
            s2 = 2 * x
            if counts[s2] >= 2:
                continue
            # Then check all sums x + c[i]
            arr = c[:n]
            sums = arr + x
            if (counts[sums] >= 2).any():
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
    """Fast simulated annealing with incremental count updates.

    Energy = sum over sums of max(0, count-2)^2.
    Moves: add, remove, swap. Uses incremental count updates for O(n) per move.
    """
    rng = random.Random(seed)
    inside = sorted(seed_set)
    inside_set = set(inside)
    outside = [v for v in range(1, cap + 1) if v not in inside_set]

    # Build initial counts
    def build_counts(vals):
        arr = np.asarray(vals, dtype=np.int64)
        nn = len(arr)
        if nn == 0:
            return np.zeros(2 * cap + 1, dtype=np.int32)
        ii, jj = np.triu_indices(nn)
        return np.bincount(arr[ii] + arr[jj], minlength=2 * cap + 1).astype(np.int32)

    counts = build_counts(inside)

    def energy_of(cnt):
        over = np.maximum(cnt - 2, 0)
        return int(np.sum(over * over))

    cur_energy = energy_of(counts)
    best_set = list(inside)
    best_energy = cur_energy
    best_size = len(inside)

    T_start = 8.0
    T_end = 0.005
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
        if it % 200 == 0 and time.time() - t0 > 40:
            break

        n = len(inside)
        r = rng.random()

        if n > 0 and (r < 0.25 or not outside):
            # Remove a random inside element
            rem_idx = rng.randrange(n)
            rem_val = inside[rem_idx]
            # Compute delta energy from removal
            # Sums affected: rem_val + x for x in inside (x != rem_val), and 2*rem_val
            # For each affected sum s, old count c, new count c-1
            # Energy change: (max(0,c-2))^2 - (max(0,c-1-2))^2 ... wait, new count is c-1
            # delta = (max(0,c-1-2))^2 - (max(0,c-2))^2 ... no
            # old contribution: max(0, c-2)^2
            # new contribution: max(0, c-1-2)^2 = max(0, c-3)^2
            # delta_e = new - old = max(0,c-3)^2 - max(0,c-2)^2
            # This is always <= 0 (removal can only decrease or keep energy same)
            delta_e = 0
            # Build list of affected sums
            affected = []
            for x in inside:
                if x != rem_val:
                    s = rem_val + x
                    affected.append(s)
            affected.append(2 * rem_val)

            # Apply removal
            for s in affected:
                old_c = counts[s]
                new_c = old_c - 1
                old_e = max(0, old_c - 2) ** 2
                new_e = max(0, new_c - 2) ** 2
                delta_e += new_e - old_e
                counts[s] = new_c

            # Remove from inside
            inside.pop(rem_idx)
            inside_set.discard(rem_val)
            outside.append(rem_val)

            new_size = n - 1
            if delta_e <= 0 or (T > 0.001 and rng.random() < np.exp(-delta_e / T)):
                cur_energy += delta_e
                if new_size > best_size or (new_size == best_size and cur_energy < best_energy):
                    best_set = list(inside)
                    best_energy = cur_energy
                    best_size = new_size
            else:
                # Undo
                for s in affected:
                    old_c = counts[s]
                    new_c = old_c + 1
                    counts[s] = new_c
                inside.insert(rem_idx, rem_val)
                inside_set.add(rem_val)
                outside.remove(rem_val)

        elif outside:
            # Add a random outside element
            add_idx = rng.randrange(len(outside))
            add_val = outside[add_idx]
            affected = []
            for x in inside:
                s = add_val + x
                affected.append(s)
            affected.append(2 * add_val)

            delta_e = 0
            for s in affected:
                old_c = counts[s]
                new_c = old_c + 1
                old_e = max(0, old_c - 2) ** 2
                new_e = max(0, new_c - 2) ** 2
                delta_e += new_e - old_e
                counts[s] = new_c

            inside.append(add_val)
            inside_set.add(add_val)
            outside.pop(add_idx)

            new_size = n + 1
            if delta_e <= 0 or (T > 0.001 and rng.random() < np.exp(-delta_e / T)):
                cur_energy += delta_e
                if new_size > best_size or (new_size == best_size and cur_energy < best_energy):
                    best_set = list(inside)
                    best_energy = cur_energy
                    best_size = new_size
            else:
                # Undo
                for s in affected:
                    old_c = counts[s]
                    new_c = old_c - 1
                    counts[s] = new_c
                inside.pop()
                inside_set.discard(add_val)
                outside.append(add_val)

        # Swap move with some probability
        elif n > 0 and outside and rng.random() < 0.4:
            rem_idx = rng.randrange(n)
            add_idx = rng.randrange(len(outside))
            rem_val = inside[rem_idx]
            add_val = outside[add_idx]

            # Compute delta for swap: remove rem_val, add add_val
            # First compute removal effects
            rem_affected = []
            for x in inside:
                if x != rem_val:
                    s = rem_val + x
                    rem_affected.append(s)
            rem_affected.append(2 * rem_val)

            # After removal, inside' = inside - {rem_val}
            # Then add add_val to inside'
            add_affected = []
            for x in inside:
                if x != rem_val:
                    s = add_val + x
                    add_affected.append(s)
            add_affected.append(2 * add_val)

            # Net effect on counts
            # We need to compute the combined delta
            # Let's just compute the new energy directly for affected sums
            all_affected = set(rem_affected + add_affected)
            delta_e = 0
            for s in all_affected:
                old_c = counts[s]
                # Compute how many of the old representations involve rem_val
                # and how many new ones involve add_val
                # This is complex; let's use a simpler approach:
                # temporarily apply and compute
                pass

            # Simpler: just apply both changes and compute energy diff
            # Apply removal
            for s in rem_affected:
                counts[s] -= 1
            # Apply addition
            for s in add_affected:
                counts[s] += 1

            new_energy = energy_of(counts)
            delta_e = new_energy - cur_energy

            inside[rem_idx] = add_val
            inside_set.discard(rem_val)
            inside_set.add(add_val)
            outside[add_idx] = rem_val

            new_size = n
            if delta_e <= 0 or (T > 0.001 and rng.random() < np.exp(-delta_e / T)):
                cur_energy = new_energy
                if new_size > best_size or (new_size == best_size and cur_energy < best_energy):
                    best_set = list(inside)
                    best_energy = cur_energy
                    best_size = new_size
            else:
                # Undo
                for s in add_affected:
                    counts[s] -= 1
                for s in rem_affected:
                    counts[s] += 1
                inside[rem_idx] = rem_val
                inside_set.add(rem_val)
                inside_set.discard(add_val)
                outside[add_idx] = add_val

        T *= cool_rate

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
    for _ in range(6):
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
        s = _greedy_build_np(order, cap=cap, tlimit=4.0)
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
    if best and _time_left() > 20.0:
        for sa_restart in range(2):
            if stopped():
                break
            if _time_left() < 15.0:
                break
            sa_seed = 500 + sa_restart * 13
            sa_result = _simulated_anneal(best, cap=cap, time_fn=time_left,
                                          stop_fn=stopped, seed=sa_seed,
                                          max_iter=100000)
            consider(sa_result)
            # Also try SA from a fresh greedy build
            if _time_left() > 20.0:
                full = list(range(1, cap + 1))
                rng.shuffle(full)
                g = _greedy_build_np(full, cap=cap, tlimit=3.0)
                consider(g)
                if _time_left() > 18.0:
                    sa_result2 = _simulated_anneal(g, cap=cap, time_fn=time_left,
                                                   stop_fn=stopped, seed=sa_seed + 999,
                                                   max_iter=80000)
                    consider(sa_result2)

    # --- Final polish: aggressive local search on the best set ---
    if best and _time_left() > 10.0:
        # Try to add any remaining outside elements that fit
        best_set = set(best)
        outside = [v for v in range(1, cap + 1) if v not in best_set]
        c = np.asarray(best, dtype=np.int64)
        n = len(c)
        ii, jj = np.triu_indices(n)
        counts = np.bincount(c[ii] + c[jj], minlength=2 * cap + 1).astype(np.int32)

        # Try adding each outside element
        added = []
        for x in outside:
            if _time_left() < 5.0:
                break
            sums = c + x
            s2 = 2 * x
            if (counts[sums] >= 2).any() or counts[s2] >= 2:
                continue
            counts[sums] += 1
            counts[s2] += 1
            best.append(x)
            best_set.add(x)
            c = np.concatenate([c, np.array([x], dtype=np.int64)])
            n += 1
            added.append(x)

        if added:
            consider(best)

        # Now try swaps to improve
        if _time_left() > 8.0:
            best_set = set(best)
            outside = [v for v in range(1, cap + 1) if v not in best_set]
            c = np.asarray(best, dtype=np.int64)
            n = len(c)
            ii, jj = np.triu_indices(n)
            counts = np.bincount(c[ii] + c[jj], minlength=2 * cap + 1).astype(np.int32)

            # Try removing one and adding two (net +1)
            for rem_idx in range(n):
                if _time_left() < 5.0:
                    break
                rem_val = int(c[rem_idx])
                # Compute counts without rem_val
                # Sums affected by removal
                rem_sums = []
                for i in range(n):
                    if i != rem_idx:
                        rem_sums.append(rem_val + int(c[i]))
                rem_sums.append(2 * rem_val)

                # Temporarily remove
                for s in rem_sums:
                    counts[s] -= 1

                # Try to add two outside elements
                best_pair = None
                for x in outside:
                    if _time_left() < 5.0:
                        break
                    sums_x = np.array([x + int(c[i]) for i in range(n) if i != rem_idx])
                    s2_x = 2 * x
                    if len(sums_x) > 0 and (counts[sums_x] >= 2).any():
                        continue
                    if counts[s2_x] >= 2:
                        continue
                    # Check if this x fits
                    # Now try to find a second element y that also fits with x
                    # This is expensive; just try one at a time for now
                    # Apply x temporarily
                    if len(sums_x) > 0:
                        counts[sums_x] += 1
                    counts[s2_x] += 1

                    # Find y
                    for y in outside:
                        if y == x:
                            continue
                        if _time_left() < 5.0:
                            break
                        # Inside now includes x (temporarily)
                        # Sums for y: y + each element in (best - {rem_val} + {x})
                        inside_vals = [int(c[i]) for i in range(n) if i != rem_idx] + [x]
                        sums_y = np.array([y + v for v in inside_vals])
                        s2_y = 2 * y
                        if (counts[sums_y] >= 2).any() or counts[s2_y] >= 2:
                            continue
                        # Both fit!
                        counts[sums_y] += 1
                        counts[s2_y] += 1
                        # Apply
                        best = [int(c[i]) for i in range(n) if i != rem_idx] + [x, y]
                        best_set = set(best)
                        outside.remove(x)
                        outside.remove(y)
                        c = np.asarray(best, dtype=np.int64)
                        n = len(c)
                        ii, jj = np.triu_indices(n)
                        counts = np.bincount(c[ii] + c[jj], minlength=2 * cap + 1).astype(np.int32)
                        best_pair = (x, y)
                        break
                    if best_pair:
                        break
                    # Undo x
                    if len(sums_x) > 0:
                        counts[sums_x] -= 1
                    counts[s2_x] -= 1

                if best_pair:
                    consider(best)
                    break
                else:
                    # Undo removal
                    for s in rem_sums:
                        counts[s] += 1

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
