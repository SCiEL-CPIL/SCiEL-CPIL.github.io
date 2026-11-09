import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096
# The evaluator kills at 90s; use a hard safety margin.
DEADLINE_SECONDS = 85


def _build_seed_network():
    """Odd-even transposition network (136 comparators)."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _verify(comps):
    """
    Fast zero-one principle verification using numpy.
    Returns True if the network sorts all 2^N boolean inputs.
    """
    total = 1 << N
    masks = np.arange(total, dtype=np.uint32)
    for (i, j) in comps:
        bi = np.uint32(1 << i)
        bj = np.uint32(1 << j)
        # If bit i is 1 and bit j is 0, swap them
        cond = (masks & bi) & ~(masks & bj)
        if np.any(cond):
            swap_mask = np.uint32((1 << i) | (1 << j))
            masks = np.where(cond, masks ^ swap_mask, masks)
    # Sorted means all 1s are above all 0s: masks & (masks+1) == 0
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
    """
    Repeatedly try to remove each comparator (from end to start) and check
    if the network still sorts. This is a local minimum finder for size.
    """
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
    """
    Local search with multiple move types:
    - Delete a comparator
    - Swap two comparators
    - Move a comparator
    - Insert a comparator (rarely)
    - Reverse a small window
    
    Uses simulated annealing to escape local minima.
    """
    comps = list(comps)
    best_comps = list(comps)
    best_len = len(comps)

    T = 4.0
    T_min = 0.05
    cool_rate = 0.997
    stall = 0
    max_stall = 400
    t_start = time.time()

    while time.time() < deadline:
        n = len(comps)
        r = rng.random()
        
        if r < 0.30:
            # Delete
            if n < 2:
                continue
            idx = int(rng.randint(0, n))
            cand = comps[:idx] + comps[idx + 1:]
        elif r < 0.50:
            # Swap two
            if n < 2:
                continue
            i = int(rng.randint(0, n - 1))
            j = int(rng.randint(0, n - 1))
            if i == j:
                continue
            cand = comps[:]
            cand[i], cand[j] = cand[j], cand[i]
        elif r < 0.60:
            # Move
            if n < 2:
                continue
            i = int(rng.randint(0, n))
            j = int(rng.randint(0, n))
            if i == j:
                continue
            cand = comps[:]
            item = cand.pop(i)
            cand.insert(j, item)
        elif r < 0.70:
            # Reverse window
            if n < 3:
                continue
            pos = int(rng.randint(0, max(1, n - 2)))
            wlen = int(rng.randint(2, min(5, n - pos)))
            seg = comps[pos:pos + wlen]
            cand = comps[:pos] + seg[::-1] + comps[pos + wlen:]
        elif r < 0.80:
            # Insert
            if n >= MAX_COMPARATORS:
                continue
            a = int(rng.randint(0, N - 1))
            b = int(rng.randint(a + 1, N))
            pos = int(rng.randint(0, n + 1))
            cand = comps[:pos] + [(a, b)] + comps[pos:]
        else:
            # Multi-swap
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

        # Evaluate
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

        # Cool
        if T > T_min:
            T *= cool_rate
        else:
            T = T_min

        # Reheat if stuck
        if stall > max_stall:
            T = max(T, 2.0)
            stall = 0

        # Periodic greedy cleanup
        if stall > 200 and time.time() < deadline - 1:
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


def _hierarchical_optimize(comps, deadline):
    """
    Try to optimize by treating the network as two halves (wires 0-8 and 9-16)
    with a crossover in the middle. This is a heuristic that may find
    redundancies that global search misses.
    """
    # This is a placeholder for a more sophisticated approach.
    # For now, just do a greedy pass.
    return _greedy_remove(comps, deadline)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    comps = _build_seed_network()
    if not _verify(comps):
        return comps

    best_comps = list(comps)
    best_len = len(comps)

    # Phase 1: Aggressive greedy deletion
    comps = _greedy_remove(comps, deadline)
    if len(comps) < best_len:
        best_len = len(comps)
        best_comps = list(comps)

    # Phase 2: Local search with multiple restarts
    rng = np.random.default_rng(42)
    max_restarts = 12
    restarts = 0

    while time.time() < deadline - 3 and restarts < max_restarts:
        restarts += 1
        # Allocate remaining time among restarts
        remaining = deadline - time.time()
        sub_deadline = time.time() + remaining / (max_restarts - restarts + 1)
        sub_deadline = min(sub_deadline, deadline - 1)

        if time.time() > sub_deadline:
            break

        seed = 42 + restarts * 1000
        rng_r = np.random.default_rng(seed)
        cand = _local_search(best_comps, rng_r, sub_deadline)
        if len(cand) < best_len:
            best_len = len(cand)
            best_comps = list(cand)

        # If we've made good progress, do a final polish
        if best_len <= 85 and time.time() > start_time + DEADLINE_SECONDS * 0.8:
            break

    # Phase 3: Final greedy polish
    if time.time() < deadline - 0.5:
        final = _greedy_remove(best_comps, deadline)
        if len(final) < best_len:
            best_comps = final
            best_len = len(final)

    return best_comps
