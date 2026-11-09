"""Combined: bitmask set-cover (current) with smarter greedy + iterative local
prune + multi-restart randomization. Keep entry-point construct() and the
backtracking validator _is_2_colorable as in the current program."""
import time
import random
from itertools import combinations
import numpy as np

_WALL_LIMIT = 150.0
_MARGIN = 15.0


def _edge_masks(n):
    """Vectorized construction of per-edge monochromatic-colouring masks."""
    edges = [tuple(e) for e in combinations(range(n), 5)]
    m = len(edges)
    rest = np.array(range(n), dtype=np.int64)
    rest_pow = np.array([1 << v for v in range(n)], dtype=np.uint64)
    masks = np.zeros(m, dtype=np.uint64)
    for i, e in enumerate(edges):
        in_edge = np.zeros(n, dtype=bool)
        in_edge[list(e)] = True
        rest_idx = rest[~in_edge]
        r = len(rest_idx)
        if r == 0:
            idx0 = 0
            idx1 = 0
            for v in e:
                idx1 |= (1 << v)
            masks[i] = (np.uint64(1) << np.uint64(idx0)) | (np.uint64(1) << np.uint64(idx1))
            continue
        p = np.arange(1 << r, dtype=np.uint64)
        bits = (p[:, None] >> np.arange(r, dtype=np.uint64)[None, :]) & np.uint64(1)
        rest_pow_sub = rest_pow[rest_idx]
        idx0 = (bits * rest_pow_sub[None, :]).sum(axis=1).astype(np.uint64)
        base1 = 0
        for v in e:
            base1 |= (1 << v)
        idx1 = idx0 + np.uint64(base1)
        masks[i] = (np.uint64(1) << idx0).sum() | (np.uint64(1) << idx1).sum()
    return edges, masks


def _greedy_cover(n, masks, full, deadline, rng, randomize=0.0):
    """Greedy set cover with optional randomization in tie-breaking."""
    uncovered = full
    chosen = []
    chosen_masks = []
    remaining = list(range(len(masks)))
    while uncovered:
        if time.time() > deadline:
            return None, None
        best = -1
        bestc = 0
        # For speed, sample if many remain
        if len(remaining) > 500:
            sample = rng.sample(remaining, min(500, len(remaining)))
        else:
            sample = remaining
        for ei in sample:
            c = (uncovered & masks[ei]).bit_count()
            if c > bestc or (c == bestc and bestc > 0 and randomize > 0 and rng.random() < randomize):
                bestc = c
                best = ei
                if bestc == uncovered.bit_count():
                    break
        if best < 0:
            # full scan fallback
            for ei in remaining:
                c = (uncovered & masks[ei]).bit_count()
                if c > bestc:
                    bestc = c
                    best = ei
        if best < 0:
            break
        chosen.append(best)
        chosen_masks.append(masks[best])
        uncovered &= ~masks[best]
        remaining.remove(best)
    if uncovered == 0:
        return chosen, chosen_masks
    return None, None


def _swap_improve(chosen, chosen_masks, masks, full, deadline, rng, max_rounds=5):
    """Try to swap one chosen edge for an unchosen one that covers strictly
    more of the currently-uncovered colourings."""
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        chosen_set = set(chosen)
        # current uncovered
        cov = np.uint64(0)
        for ei in chosen:
            cov |= masks[ei]
        uncovered = full & ~cov
        if uncovered == 0:
            return chosen, chosen_masks
        uc_count = uncovered.bit_count()
        # candidate pool: edges not already chosen
        all_idx = list(range(len(masks)))
        candidates = [i for i in all_idx if i not in chosen_set]
        if not candidates:
            return chosen, chosen_masks
        # sample candidates for speed
        if len(candidates) > 2000:
            candidates = rng.sample(candidates, 2000)
        # For each chosen edge i, find best swap partner j that maximizes
        # (uncovered_after_swap) = (uncovered & ~m_j) | (m_i & ~cov_without_i)
        # Approximate: gain = popcount((m_i & ~cov) & ~m_j) - popcount(m_j & cov_without_i)
        # We'll do a simpler heuristic: prefer j that covers the most uncovered
        # bits not already covered, and whose removal of i doesn't hurt.
        order = list(range(len(chosen)))
        rng.shuffle(order)
        for ii in order:
            if ii >= len(chosen):
                continue
            i = chosen[ii]
            # coverage without i
            cov_no_i = np.uint64(0)
            for jj, ei in enumerate(chosen):
                if jj == ii:
                    continue
                cov_no_i |= masks[ei]
            # bits that only i covers
            only_i = cov & ~cov_no_i
            # We need j to cover all of only_i (else we lose coverage)
            # and to add as much of uncovered as possible.
            best_j = -1
            best_gain = 0
            for j in candidates:
                if j in chosen_set:
                    continue
                mj = masks[j]
                if (only_i & ~mj) != 0:
                    continue
                gain = (uncovered & mj).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j >= 0 and best_gain > 0:
                chosen[ii] = best_j
                chosen_masks[ii] = masks[best_j]
                chosen_set.add(best_j)
                chosen_set.discard(i)
                improved = True
                break
    return chosen, chosen_masks


def _local_prune(chosen, chosen_masks, full, deadline, rng, max_rounds=20):
    """Iteratively remove any edge whose absence still covers all colourings."""
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        order = list(range(len(chosen)))
        rng.shuffle(order)
        for i in order:
            if i >= len(chosen):
                continue
            u = 0
            for j in range(len(chosen)):
                if j == i:
                    continue
                u |= chosen_masks[j]
            if u == full:
                del chosen[i]
                del chosen_masks[i]
                improved = True
                break
    return chosen, chosen_masks


def _try_n(n, deadline, rng):
    edges, masks = _edge_masks(n)
    full = (1 << (1 << n)) - 1
    best = None
    best_len = float('inf')
    # Multiple restarts with varying randomization; more restarts possible
    # now that mask construction is vectorized.
    restarts = 12
    for r in range(restarts):
        if time.time() > deadline:
            break
        randomize = 0.05 + 0.15 * (r + 1) / restarts
        chosen, chosen_masks = _greedy_cover(n, masks, full, deadline, rng, randomize)
        if chosen is None:
            continue
        chosen, chosen_masks = _local_prune(chosen, chosen_masks, full, deadline, rng)
        chosen, chosen_masks = _swap_improve(chosen, chosen_masks, masks, full, deadline, rng)
        chosen, chosen_masks = _local_prune(chosen, chosen_masks, full, deadline, rng, max_rounds=10)
        if len(chosen) < best_len:
            best_len = len(chosen)
            best = (list(chosen), list(chosen_masks))
            # If we hit a very small count, stop early
            if best_len <= 30:
                break
    if best is None:
        return None
    chosen, _ = best
    # exact certificate: union of masks must equal full
    u = 0
    for ei in chosen:
        u |= masks[ei]
    if u != full:
        return None
    return {'n': n, 'edges': [list(edges[i]) for i in chosen]}


def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT - _MARGIN
    rng = random.Random(12345)

    # Guaranteed fallback: complete 5-graph on 9 vertices (126 edges).
    fallback = {'n': 9,
                'edges': [list(e) for e in combinations(range(9), 5)]}

    best = fallback
    for n in (9, 10, 11, 12, 13):
        if time.time() > deadline:
            break
        try:
            res = _try_n(n, deadline, rng)
        except Exception:
            res = None
        if res is not None and len(res['edges']) < len(best['edges']):
            best = res

    return best


def _is_2_colorable(n, edges, deadline):
    """True iff there exists a 2-colouring with NO monochromatic edge.
    Uses backtracking with unit propagation."""
    edges = [tuple(e) for e in edges]
    m = len(edges)
    if m == 0:
        return True

    color = [-1] * n
    e0 = [0] * m
    e1 = [0] * m
    uncol = [list(e) for e in edges]

    def propagate(queue):
        while queue:
            ei = queue.pop()
            u = uncol[ei]
            if e0[ei] == 5 or e1[ei] == 5:
                return False
            if len(u) == 1:
                v = u[0]
                if e0[ei] == 4:
                    forced = 1
                elif e1[ei] == 4:
                    forced = 0
                else:
                    continue
                if color[v] == -1:
                    color[v] = forced
                    uncol[ei].remove(v)
                    if forced == 0:
                        e0[ei] += 1
                    else:
                        e1[ei] += 1
                    for ej in range(m):
                        if ej != ei and v in uncol[ej]:
                            uncol[ej].remove(v)
                            if forced == 0:
                                e0[ej] += 1
                            else:
                                e1[ej] += 1
                            if e0[ej] == 5 or e1[ej] == 5:
                                return False
                            if len(uncol[ej]) == 1:
                                queue.append(ej)
                elif color[v] != forced:
                    return False
        return True

    import sys
    sys.setrecursionlimit(100000)

    def rec(k):
        if time.time() > deadline:
            return None
        if k >= n:
            return True
        for c in (0, 1):
            color[k] = c
            touched = []
            conflict = False
            for ei, e in enumerate(edges):
                if k in e:
                    uncol[ei].remove(k)
                    if c == 0:
                        e0[ei] += 1
                    else:
                        e1[ei] += 1
                    touched.append(ei)
                    if e0[ei] == 5 or e1[ei] == 5:
                        conflict = True
                        break
            if conflict:
                color[k] = -1
                for ej in touched:
                    if k in uncol[ej]:
                        uncol[ej].append(k)
                    if c == 0:
                        e0[ej] -= 1
                    else:
                        e1[ej] -= 1
                continue
            queue = [ei for ei in touched if len(uncol[ei]) == 1]
            if propagate(queue):
                result = rec(k + 1)
                if result is not None:
                    return result
            color[k] = -1
            for ej in touched:
                if k in uncol[ej]:
                    uncol[ej].append(k)
                if c == 0:
                    e0[ej] -= 1
                else:
                    e1[ej] -= 1
        return False

    result = rec(0)
    if result is None:
        return False
    return result


if __name__ == '__main__':
    _t0 = time.time()
    _res = construct()
    _ok = _is_2_colorable(_res['n'], _res['edges'], _t0 + 120.0)
    assert not _ok, "constructed hypergraph must NOT be 2-colourable"
    print("edges:", len(_res['edges']), "n:", _res['n'], "2-colourable:", _ok)
