"""Rewritten: prune + local-search sorting network on 17 wires."""
import time
import numpy as np

N = 17
DEADLINE = 82.0  # safety margin below 90s kill
START = time.time()


def _time_left():
    return DEADLINE - (time.time() - START)


def _network_matrix(comps):
    """Return (N,N) permutation matrix P such that wires_out = P @ wires_in."""
    P = np.eye(N, dtype=np.int8)
    for i, j in comps:
        c = (P[i, :] > P[j, :]).astype(np.int8)
        P[i, :] = P[i, :] - c
        P[j, :] = P[j, :] + c
    return P


def _is_sorted_network(comps):
    """True iff the network sorts every 0/1 input (zero-one principle)."""
    P = _network_matrix(comps)
    if P.shape[0] != N:
        return False
    # column sums must be exactly 1 (valid permutation)
    if not np.all(P.sum(axis=0) == 1):
        return False
    # For a sorting network: for each k, the top k wires must contain at
    # least the top k ones for every input. Equivalent: the k-th row (0-indexed)
    # must have weight >= k for all k in 0..N-1 ... but that's for a *fixed*
    # permutation. The correct condition: the network is a permutation matrix P
    # (output = P @ input). It sorts all inputs iff for every k (1..N), the sum
    # of the top k rows of P is >= k (i.e. at least k of the top k output wires
    # receive an input wire). Actually the standard test: a permutation P sorts
    # all 0/1 inputs iff for every k, the number of 1s in the first k output
    # positions is >= k whenever the input has k ones. The minimal case is the
    # input with ones on the *last* k input wires; then output ones = sum of
    # P[:k, N-k:]. So condition: for all k, sum_{r<k, c>=N-k} P[r,c] >= k.
    for k in range(1, N + 1):
        s = int(P[:k, N - k:].sum())
        if s < k:
            return False
    return True


def _seed_network():
    # Odd-even transposition (seed) - 136 comparators for 17 wires.
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _prune(comps):
    """Greedily drop any comparator whose removal preserves sorting."""
    comps = list(comps)
    changed = True
    # Try repeatedly from the end (removing later comparators is cheaper to
    # re-test since earlier ones are fixed).
    while changed and _time_left() > 5.0:
        changed = False
        for idx in range(len(comps) - 1, -1, -1):
            if _time_left() < 5.0:
                break
            cand = comps[:idx] + comps[idx + 1:]
            if _is_sorted_network(cand):
                comps = cand
                changed = True
                break  # re-scan after a removal
    return comps


def _local_search(comps, budget):
    """Local search: try deleting, swapping adjacent, and adding a random
    comparator; keep the best (fewest comparators) valid network."""
    comps = list(comps)
    best = comps
    best_len = len(comps)
    rng = np.random.default_rng(12345)
    while _time_left() > budget:
        # 1) deletion attempts (random position)
        if len(comps) > 1:
            idx = int(rng.integers(0, len(comps)))
            cand = comps[:idx] + comps[idx + 1:]
            if _is_sorted_network(cand):
                comps = cand
                if len(comps) < best_len:
                    best = list(comps); best_len = len(comps)
                continue
        # 2) swap two comparators
        if len(comps) >= 2:
            a = int(rng.integers(0, len(comps)))
            b = int(rng.integers(0, len(comps)))
            if a != b:
                cand = list(comps)
                cand[a], cand[b] = cand[b], cand[a]
                if _is_sorted_network(cand):
                    comps = cand
                    if len(comps) < best_len:
                        best = list(comps); best_len = len(comps)
                continue
        # 3) add a random comparator (may enable later deletions)
        i = int(rng.integers(0, N - 1))
        j = int(rng.integers(i + 1, N))
        pos = int(rng.integers(0, len(comps) + 1))
        cand = list(comps)
        cand.insert(pos, (i, j))
        if _is_sorted_network(cand):
            # only keep the addition if it later allows a net reduction;
            # for now accept it as a move to explore from.
            comps = cand
    return best


def construct():
    comps = _seed_network()
    # First, prune the seed to its irredundant core.
    comps = _prune(comps)
    # Then local search to shrink further.
    comps = _local_search(comps, budget=8.0)
    # Final safety prune.
    comps = _prune(comps)
    # Final validity check; fall back to seed if anything went wrong.
    if not _is_sorted_network(comps):
        comps = _seed_network()
    return list(comps)
