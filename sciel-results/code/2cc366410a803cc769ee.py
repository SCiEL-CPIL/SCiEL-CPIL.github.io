import time
import random
import numpy as np

N = 1000
MAX_SUM = 2 * N          # 2000
K = 2                    # max representations allowed


def _can_add_vec(x, chosen_arr, counts):
    """True if adding x keeps every sum <= K. Vectorized."""
    if chosen_arr.size:
        s = chosen_arr + x
        if np.any(counts[s] >= K):
            return False
    if counts[2 * x] >= K:
        return False
    return True


def _add_vec(x, chosen_arr, counts):
    """Update counts in place after adding x."""
    if chosen_arr.size:
        s = chosen_arr + x
        counts[s] += 1
    counts[2 * x] += 1


def _remove_vec(x, chosen_arr, counts):
    """Update counts in place after removing x (x assumed present)."""
    if chosen_arr.size:
        s = chosen_arr + x
        counts[s] -= 1
    counts[2 * x] -= 1


def _can_add2_vec(x, y, chosen_arr, counts):
    """True if adding BOTH x and y keeps every sum <= K."""
    if chosen_arr.size:
        s = chosen_arr + x
        if np.any(counts[s] >= K):
            return False
        s = chosen_arr + y
        if np.any(counts[s] >= K):
            return False
    if counts[2 * x] >= K:
        return False
    if counts[2 * y] >= K:
        return False
    if counts[x + y] >= K:
        return False
    return True


def _add2_vec(x, y, chosen_arr, counts):
    """Update counts in place after adding both x and y."""
    if chosen_arr.size:
        counts[chosen_arr + x] += 1
        counts[chosen_arr + y] += 1
    counts[2 * x] += 1
    counts[2 * y] += 1
    counts[x + y] += 1


def _saturating_greedy(order, counts, noise=0.0, rng=None):
    """Greedy that adds ANY valid element (order decides which first).

    With noise>0, occasionally skip a valid element (to allow diversity later).
    """
    chosen = np.empty(0, dtype=np.int64)
    for x in order:
        if _can_add_vec(x, chosen, counts):
            if noise > 0.0 and rng is not None and rng.random() < noise:
                continue
            _add_vec(x, chosen, counts)
            chosen = np.concatenate([chosen, np.array([x], dtype=np.int64)])
    return chosen, counts


def _local_search(chosen, counts, rng, deadline, max_iter=2000000):
    """Grow / 2-opt / swap local search with fast vectorized validity checks.

    Moves (in priority order):
      (A) INSERT one new element           -> grows size by 1
      (B) 2-OPT  : remove 2, add 2         -> grows size by 1 (breaks deadlocks)
      (C) SWAP   : remove 1, add 1         -> same size, frees capacity
    """
    n = len(chosen)
    chosen_set = set(int(v) for v in chosen)
    best_size = n
    best_chosen = chosen.copy()

    avail = np.array([v for v in range(1, N + 1) if v not in chosen_set], dtype=np.int64)
    rng.shuffle(avail)

    start = time.time()
    for it in range(max_iter):
        if (it & 511) == 0:
            if time.time() > deadline:
                break
        if avail.size == 0:
            break
        if n == 0:
            break

        # pick a random available candidate
        x_add = int(avail[rng.randrange(avail.size)])
        if x_add in chosen_set:
            continue

        # ---- (A) direct INSERT (pure growth) ----
        if _can_add_vec(x_add, chosen, counts):
            _add_vec(x_add, chosen, counts)
            chosen = np.concatenate([chosen, np.array([x_add], dtype=np.int64)])
            chosen_set.add(x_add)
            avail = avail[avail != x_add]
            n += 1
            if n > best_size:
                best_size = n
                best_chosen = chosen.copy()
            continue

        # ---- (B) 2-OPT : remove two, add x_add + one more ----
        if n >= 2:
            i1 = rng.randrange(n)
            i2 = rng.randrange(n - 1)
            if i2 >= i1:
                i2 += 1
            r1 = int(chosen[i1])
            r2 = int(chosen[i2])

            # counts after removing r1 and r2
            c2 = counts.copy()
            if chosen.size:
                c2[chosen + r1] -= 1
                c2[chosen + r2] -= 1
            c2[2 * r1] -= 1
            c2[2 * r2] -= 1
            c2[r1 + r2] += 1  # the sum r1+r2 is no longer represented

            # pick a second add candidate
            x_add2 = int(avail[rng.randrange(avail.size)])
            if x_add2 in chosen_set or x_add2 == x_add:
                x_add2 = x_add
            # if x_add2 conflicts with x_add, just fall through to swap
            if x_add2 != x_add:
                if _can_add2_vec(x_add, x_add2, chosen, c2):
                    _add2_vec(x_add, x_add2, chosen, c2)
                    # remove r1, r2 from chosen
                    chosen = chosen[(chosen != r1) & (chosen != r2)]
                    chosen_set.discard(r1)
                    chosen_set.discard(r2)
                    chosen_set.add(x_add)
                    chosen_set.add(x_add2)
                    # update avail
                    m = avail != x_add
                    avail = avail[m]
                    if x_add2 != x_add:
                        m = avail != x_add2
                        avail = avail[m]
                    new_avail = np.array([v for v in (r1, r2) if v not in chosen_set],
                                         dtype=np.int64)
                    if new_avail.size:
                        avail = np.concatenate([avail, new_avail])
                    n = len(chosen)
                    if n > best_size:
                        best_size = n
                        best_chosen = chosen.copy()
                    continue

        # ---- (C) SWAP : remove one, add x_add (same size, frees capacity) ----
        idx = rng.randrange(n)
        x_rm = int(chosen[idx])

        c2 = counts.copy()
        if chosen.size:
            c2[chosen + x_rm] -= 1
        c2[2 * x_rm] -= 1

        s_add = chosen[chosen != x_rm] + x_add
        if (chosen.size and np.any(c2[s_add] >= K)) or c2[2 * x_add] >= K:
            continue

        # apply swap
        _add_vec(x_add, chosen[chosen != x_rm], c2)
        chosen = np.concatenate([chosen[chosen != x_rm], np.array([x_add], dtype=np.int64)])
        chosen_set.discard(x_rm)
        chosen_set.add(x_add)
        m = avail != x_add
        avail = avail[m]
        if x_rm not in chosen_set:
            avail = np.concatenate([avail, np.array([x_rm], dtype=np.int64)])
        # size unchanged

    return best_chosen


def construct():
    start_time = time.time()
    deadline = start_time + 135.0   # safety margin vs 150s kill

    rng = random.Random(42)

    best_chosen = np.empty(0, dtype=np.int64)
    best_size = 0

    # --- Phase 1: diverse greedy seeds (incl. skip-noise for diversity) ---
    seed_orders = [
        (list(range(1, N + 1)), 0.0, None),
        (list(range(N, 0, -1)), 0.0, None),
    ]
    for seed in (123, 7, 999, 2024, 555, 31337):
        o = list(range(1, N + 1))
        r = random.Random(seed)
        r.shuffle(o)
        seed_orders.append((o, 0.0, None))
    # a couple of skip-noise greedy seeds
    for seed in (4242, 8181):
        o = list(range(1, N + 1))
        r = random.Random(seed)
        r.shuffle(o)
        seed_orders.append((o, 0.10, r))
        seed_orders.append((o, 0.25, r))

    for o, noise, rn in seed_orders:
        ch, ct = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64),
                                    noise=noise, rng=rn)
        if len(ch) > best_size:
            best_size = len(ch)
            best_chosen = ch

    # --- Phase 2: local search from the best seed ---
    counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    # rebuild counts from best_chosen
    a = best_chosen
    if a.size:
        iu, ju = np.triu_indices(a.size)
        np.add.at(counts, a[iu] + a[ju], 1)
    ch = _local_search(best_chosen, counts, rng, deadline, max_iter=3000000)
    if len(ch) > best_size:
        best_size = len(ch)
        best_chosen = ch

    # --- Phase 3: restart loop with fresh seeds + short local search ---
    for trial in range(40):
        if time.time() > deadline:
            break
        o = list(range(1, N + 1))
        rt = random.Random(5000 + trial * 13)
        rt.shuffle(o)
        noise = 0.15 if trial % 3 == 0 else 0.0
        ch2, ct2 = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64),
                                      noise=noise, rng=rt if noise else None)
        if len(ch2) <= best_size:
            continue
        ch2 = _local_search(ch2, ct2, rng, deadline, max_iter=150000)
        if len(ch2) > best_size:
            best_size = len(ch2)
            best_chosen = ch2

    return [int(v) for v in best_chosen]


# EVOLVE-BLOCK-END
