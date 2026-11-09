import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 78


def _build_seed_network():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _verify(comps):
    total = 1 << N
    masks = np.arange(total, dtype=np.uint32)
    for (i, j) in comps:
        bi = np.uint32(1 << i)
        bj = np.uint32(1 << j)
        cond = (masks & bi) & ~(masks & bj)
        if np.any(cond):
            swap_mask = np.uint32((1 << i) | (1 << j))
            masks = np.where(cond, masks ^ swap_mask, masks)
    return bool(np.all((masks & (masks + 1)) == 0))


def _depth_of_network(comps):
    if not comps:
        return 0
    depth = [0] * N
    for (i, j) in comps:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)


def _greedy_remove(comps, deadline):
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(comps)
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return comps
            new_comps = comps[:idx] + comps[idx + 1:]
            if _verify(new_comps):
                comps = new_comps
                improved = True
                break
    return comps


def _local_search(comps, rng, deadline):
    """Legacy thin wrapper kept for compatibility; delegates to annealing."""
    return _simulated_anneal(comps, rng, deadline)


def _simulated_anneal(comps, rng, deadline):
    """
    Simulated-annealing optimizer.

    Unlike pure greedy deletion, this can *increase* network size (insertions)
    and perform larger structural moves (multi-swap, local reverse) so it can
    escape local minima that trap deletion-only search. It accepts worse
    states with a temperature-scheduled probability to explore the space.
    """
    comps = list(comps)
    best_comps = list(comps)
    best_len = len(comps)

    # Temperature schedule: start hot, cool geometrically toward a floor.
    T = 6.0
    T_min = 0.05
    cool_rate = 0.995
    t_start = time.time()
    # Estimate total iterations to make cooling reach T_min by deadline.
    # We'll just cool per iteration and also stop on deadline.
    n_iter = 0
    stall = 0
    max_stall = 600

    while time.time() < deadline:
        n_iter += 1
        if time.time() > deadline:
            break

        n = len(comps)
        # Choose a move type with weights; favor deletions when possible.
        r = rng.random()
        if r < 0.35:
            # Delete a random comparator
            if n < 2:
                continue
            idx = int(rng.randint(0, n))
            cand = comps[:idx] + comps[idx + 1:]
        elif r < 0.55:
            # Swap two comparators
            if n < 2:
                continue
            i = int(rng.randint(0, n - 1))
            j = int(rng.randint(0, n - 1))
            if i == j:
                continue
            cand = comps[:]
            cand[i], cand[j] = cand[j], cand[i]
        elif r < 0.65:
            # Multi-swap: swap two pairs (structural rearrangement)
            if n < 4:
                continue
            i = int(rng.randint(0, n - 1))
            j = int(rng.randint(0, n - 1))
            if i == j:
                continue
            k = int(rng.randint(0, n - 1))
            m = int(rng.randint(0, n - 1))
            if k in (i, j) or m in (i, j) or k == m:
                continue
            cand = comps[:]
            cand[i], cand[j] = cand[j], cand[i]
            cand[k], cand[m] = cand[m], cand[k]
        elif r < 0.75:
            # Insert a random comparator (allows growth)
            if n >= MAX_COMPARATORS:
                continue
            a = int(rng.randint(0, N - 1))
            b = int(rng.randint(a + 1, N))
            pos = int(rng.randint(0, n + 1))
            cand = comps[:pos] + [(a, b)] + comps[pos:]
        elif r < 0.85:
            # Local reverse of a small window (structural)
            if n < 3:
                continue
            pos = int(rng.randint(0, max(1, n - 2)))
            wlen = int(rng.randint(2, min(5, n - pos)))
            seg = comps[pos:pos + wlen]
            cand = comps[:pos] + seg[::-1] + comps[pos + wlen:]
        else:
            # Move a comparator to a new position
            if n < 2:
                continue
            i = int(rng.randint(0, n))
            j = int(rng.randint(0, n))
            if i == j:
                continue
            cand = comps[:]
            item = cand.pop(i)
            cand.insert(j, item)

        # Evaluate candidate
        if len(cand) >= best_len:
            # Only accept non-regressions cheaply; regressions need full check
            if _verify(cand):
                delta = len(cand) - len(comps)
                if delta <= 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                    comps = cand
                    if len(comps) < best_len:
                        best_len = len(comps)
                        best_comps = list(comps)
                        stall = 0
                    else:
                        stall += 1
                else:
                    stall += 1
            else:
                stall += 1
        else:
            # Candidate is shorter than current best; must verify correctness
            if _verify(cand):
                delta = len(cand) - len(comps)
                if delta <= 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                    comps = cand
                    if len(comps) < best_len:
                        best_len = len(comps)
                        best_comps = list(comps)
                        stall = 0
                    else:
                        stall += 1
                else:
                    stall += 1
            else:
                stall += 1

        # Cool down
        if T > T_min:
            T *= cool_rate
        else:
            # Keep a small temperature floor to allow occasional escapes
            T = T_min

        # Restart temperature if stuck for a while (reheat)
        if stall > max_stall:
            T = max(T, 3.0)
            stall = 0

        # Occasional greedy cleanup to remove redundancies
        if n_iter % 500 == 0 and time.time() < deadline - 1:
            cleaned = _greedy_remove(comps, deadline)
            if len(cleaned) < len(comps):
                comps = cleaned
                if len(comps) < best_len:
                    best_len = len(comps)
                    best_comps = list(comps)
                    stall = 0

    # Final greedy cleanup
    if time.time() < deadline - 0.5:
        final = _greedy_remove(best_comps, deadline)
        if len(final) < best_len:
            best_comps = final
            best_len = len(final)

    return best_comps


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    comps = _build_seed_network()
    if not _verify(comps):
        return comps

    best_comps = list(comps)
    best_len = len(comps)

    # Initial aggressive greedy deletion
    comps = _greedy_remove(comps, deadline)
    if len(comps) < best_len:
        best_len = len(comps)
        best_comps = list(comps)

    rng = np.random.default_rng(42)

    # Run simulated annealing in multiple restarts with different seeds
    # to explore different regions of the search space.
    restarts = 0
    max_restarts = 8
    while time.time() < deadline - 2 and restarts < max_restarts:
        restarts += 1
        # Give each restart its own sub-deadline
        sub_deadline = min(deadline - 1, start_time + DEADLINE_SECONDS * (restarts / max_restarts))
        if time.time() > sub_deadline:
            break
        seed = 42 + restarts * 1000
        rng_r = np.random.default_rng(seed)
        cand = _simulated_anneal(best_comps, rng_r, sub_deadline)
        if len(cand) < best_len:
            best_len = len(cand)
            best_comps = list(cand)

        # If we've hit a good size, do a final polish
        if best_len <= 80 and time.time() > start_time + DEADLINE_SECONDS * 0.7:
            break

    # Final polish with greedy deletion on the best found
    if time.time() < deadline - 0.5:
        final = _greedy_remove(best_comps, deadline)
        if len(final) < best_len:
            best_comps = final
            best_len = len(final)

    return best_comps
