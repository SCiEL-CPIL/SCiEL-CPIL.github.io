"""Optimized two-colouring of [1,256] minimizing monochromatic 4-APs.

Combines:
  - Fast incremental single-flip local search (precomputed neighbor lists).
  - Vectorized batch simulated annealing (Ising-spin style) for global escape.
  - Multiple Thue-Morse restarts with short refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256
MAX_D = N // 3  # 85


def _thue_morse_fast(n):
    """Generate Thue-Morse sequence: t(i) = popcount(i) % 2."""
    c = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        c[i] = c[i >> 1] ^ (i & 1)
    return c


def _precompute_ap_neighbors():
    """neighbors[pos] = list of (other1, other2, other3) for each 4-AP containing pos."""
    neighbors = [[] for _ in range(N)]
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            p0 = a - 1
            p1 = p0 + d
            p2 = p0 + 2 * d
            p3 = p0 + 3 * d
            neighbors[p0].append((p1, p2, p3))
            neighbors[p1].append((p0, p2, p3))
            neighbors[p2].append((p0, p1, p3))
            neighbors[p3].append((p0, p1, p2))
    return neighbors


_NEIGHBORS = _precompute_ap_neighbors()


def _count_mono4_vectorized(c):
    """Vectorized count of monochromatic 4-APs."""
    count = 0
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            continue
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += int(np.sum(mask))
    return count


def _delta_flip(c, pos):
    """Change in mono-4-AP count when flipping position pos."""
    old_val = c[pos]
    new_val = 1 - old_val
    delta = 0
    for n1, n2, n3 in _NEIGHBORS[pos]:
        v1 = c[n1]
        v2 = c[n2]
        v3 = c[n3]
        mono_before = (old_val == v1) and (v1 == v2) and (v2 == v3)
        mono_after = (new_val == v1) and (v1 == v2) and (v2 == v3)
        if mono_before and not mono_after:
            delta -= 1
        elif not mono_before and mono_after:
            delta += 1
    return delta


def _local_search_incremental(c, time_limit, rng):
    """Local search using incremental delta computation."""
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)

    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        order = list(range(N))
        rng.shuffle(order)
        for pos in order:
            if (time.time() - start) >= time_limit:
                break
            delta = _delta_flip(best_c, pos)
            if delta < 0:
                best_c[pos] = 1 - best_c[pos]
                best_count += delta
                improved = True
    return best_c, best_count


def _batch_simulated_annealing(c, time_limit, rng):
    """Vectorized batch SA: flip many positions at once, recompute count once.

    This models the coloring as a spin system where we propose a batch of flips
    and accept the whole batch with a Metropolis criterion. Because the count is
    recomputed in a single vectorized pass per batch, we can try many more
    configurations than sequential single flips, helping escape local minima.
    """
    start = time.time()
    cur_c = c.copy()
    cur_count = _count_mono4_vectorized(cur_c)
    best_c = cur_c.copy()
    best_count = cur_count

    T = 4.0
    T_min = 0.05
    cooling = 0.998
    batch_size = 8

    while (time.time() - start) < time_limit and T > T_min:
        # Propose a batch of random flips
        idx = rng.sample(range(N), batch_size)
        for p in idx:
            cur_c[p] = 1 - cur_c[p]
        new_count = _count_mono4_vectorized(cur_c)
        delta = new_count - cur_count

        if delta < 0 or rng.random() < np.exp(-delta / T):
            cur_count = new_count
            if cur_count < best_count:
                best_count = cur_count
                best_c = cur_c.copy()
        else:
            # Reject: revert
            for p in idx:
                cur_c[p] = 1 - cur_c[p]

        T *= cooling

    return best_c, best_count


def _two_flip_search(c, time_limit, rng):
    """Try flipping pairs of positions simultaneously."""
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)

    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for _ in range(400):
            if (time.time() - start) >= time_limit:
                break
            i = rng.randint(0, N - 1)
            j = rng.randint(0, N - 1)
            if i == j:
                continue
            best_c[i] = 1 - best_c[i]
            best_c[j] = 1 - best_c[j]
            new_count = _count_mono4_vectorized(best_c)
            if new_count < best_count:
                best_count = new_count
                improved = True
            else:
                best_c[i] = 1 - best_c[i]
                best_c[j] = 1 - best_c[j]
    return best_c, best_count


def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    rng = random.Random(12345)

    # Start with Thue-Morse
    c = _thue_morse_fast(N)
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)

    # Phase 1: local search on initial Thue-Morse
    remaining = deadline - time.time()
    if remaining > 15:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 15, 25), rng)

    # Phase 2: batch SA to escape local minima
    remaining = deadline - time.time()
    if remaining > 20:
        sa_c, sa_count = _batch_simulated_annealing(best_c, min(remaining - 20, 18), rng)
        if sa_count < best_count:
            best_c = sa_c
            best_count = sa_count

    # Phase 3: refine after SA
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 10, 15), rng)

    # Phase 4: two-flip search
    remaining = deadline - time.time()
    if remaining > 15:
        best_c, best_count = _two_flip_search(best_c, min(remaining - 15, 12), rng)

    # Phase 5: multiple Thue-Morse restarts with short refinement + occasional batch SA
    remaining = deadline - time.time()
    if remaining > 20:
        for offset in range(0, 32):
            if time.time() >= deadline - 10:
                break
            c2 = _thue_morse_fast(N)
            c2 = np.roll(c2, offset)
            # short local search
            c2, c2_count = _local_search_incremental(c2, min(3.0, deadline - time.time() - 5), rng)
            # occasional batch SA for diversity
            if offset % 4 == 0 and (deadline - time.time()) > 12:
                c2, c2_count = _batch_simulated_annealing(c2, min(4.0, deadline - time.time() - 5), rng)
                c2, c2_count = _local_search_incremental(c2, min(2.0, deadline - time.time() - 5), rng)
            if c2_count < best_count:
                best_c = c2
                best_count = c2_count

    # Final local search
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 5, 8), rng)

    result = [int(x) for x in best_c]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
# EVOLVE-BLOCK-END
