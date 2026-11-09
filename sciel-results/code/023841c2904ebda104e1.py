"""Erdos minimum overlap: find equal-width step heights h in [0,1], sum N/2,
minimizing the maximum overlap of h with shifted (1-h)."""
import numpy as np
import time

N = 100


def _overlap_max(h):
    """Return the maximum over all nonzero shifts of sum_i h[i]*(1-h[(i+k)%N])."""
    n = len(h)
    # For each shift k, overlap = sum_i h[i] - sum_i h[i]*h[(i+k)%n]
    # = sum(h) - autocorr(k)
    # autocorr(k) = sum_i h[i]*h[(i+k)%n]
    best = 0.0
    sh = sum(h)
    for k in range(1, n):
        ac = np.dot(h, np.roll(h, -k))
        ov = sh - ac
        if ov > best:
            best = ov
    return best


def _try_binary_sequence(ones_positions, n):
    """Given a set of 'on' indices (binary h), return max overlap."""
    h = np.zeros(n)
    h[list(ones_positions)] = 1.0
    return _overlap_max(h)


def _local_search_binary(h, max_iters, deadline):
    """Local search via swaps on a binary sequence. Returns best h found."""
    n = len(h)
    ones = set(np.where(h == 1.0)[0].tolist())
    zeros = set(np.where(h == 0.0)[0].tolist())
    best_h = h.copy()
    best_score = _overlap_max(best_h)
    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break
        # pick a random one and a random zero, try swap
        a = next(iter(ones))
        b = next(iter(zeros))
        h[a] = 0.0
        h[b] = 1.0
        ones.discard(a); ones.add(b)
        zeros.add(a); zeros.discard(b)
        s = _overlap_max(h)
        if s < best_score - 1e-12:
            best_score = s
            best_h = h.copy()
        else:
            # revert
            h[a] = 1.0
            h[b] = 0.0
            ones.add(a); ones.discard(b)
            zeros.discard(a); zeros.add(b)
        it += 1
    return best_h, best_score


def _greedy_even_spread(n):
    """Construct a binary sequence with n/2 ones as evenly spread as possible."""
    m = n // 2
    h = np.zeros(n)
    # place ones at indices i * n // m for i in range(m) -> even spread
    idx = (np.arange(m) * n // m)
    h[idx] = 1.0
    return h


def construct():
    deadline = time.time() + 100.0  # safety margin before 150s kill (and 120s eval)
    n = N
    m = n // 2

    # Start from an evenly-spread binary sequence
    h = _greedy_even_spread(n)
    best_h = h.copy()
    best_score = _overlap_max(best_h)

    # Multiple restarts with random permutations of the even-spread pattern
    rng = np.random.default_rng(0)
    iters_per_restart = 4000
    restarts = 0
    while time.time() < deadline:
        # new random binary sequence with exactly m ones
        perm = rng.permutation(n)
        h = np.zeros(n)
        h[perm[:m]] = 1.0
        s = _overlap_max(h)
        if s < best_score - 1e-12:
            best_score = s
            best_h = h.copy()
        # local search from this start
        h2, s2 = _local_search_binary(h, iters_per_restart, deadline)
        if s2 < best_score - 1e-12:
            best_score = s2
            best_h = h2.copy()
        restarts += 1
        if time.time() > deadline:
            break

    # Ensure exact sum = n/2 (binary already has exactly m ones => sum = m = n/2)
    # Final verification
    assert abs(best_h.sum() - n / 2.0) < 1e-9, f"sum {best_h.sum()} != {n/2}"
    assert np.all(best_h >= 0.0) and np.all(best_h <= 1.0)
    return best_h


# EVOLVE-BLOCK-START
# (kept for compatibility; the real logic is above)
# EVOLVE-BLOCK-END
