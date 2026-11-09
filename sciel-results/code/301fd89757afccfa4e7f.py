"""Combined: bitmask set-cover with faster mask construction, smarter greedy,
swap-improve local search + multi-restart randomization. Keep entry-point
construct() and the backtracking validator _is_2_colorable."""
import time
import random
from itertools import combinations

_WALL_LIMIT = 150.0
_MARGIN = 15.0


def _build_masks_fast(n):
    """Build edges list and monochromatic masks.
    For each 5-subset E of {0..n-1}, the mask has bits set at all colourings
    where E is monochromatic (all 0 or all 1)."""
    edges = [tuple(e) for e in combinations(range(n), 5)]
    bit = [1 << v for v in range(n)]
    if n <= 12:
        # Precompute subset_or for all subsets of {0..n-1}
        subset_or = [0] * (1 << n)
        for s in range(1, 1 << n):
            lb = s & (-s)
            v = lb.bit_length() - 1
            subset_or[s] = subset_or[s ^ lb] | bit[v]
        full_mask = (1 << n) - 1
        masks = []
        for e in edges:
            e_mask = 0
            for v in e:
                e_mask |= (1 << v)
            rest_mask = full_mask ^ e_mask
            base1 = e_mask
            m = 0
            sub = rest_mask
            while True:
                idx0 = subset_or[sub]
                idx1 = base1 | subset_or[sub]
                m |= (1 << idx0) | (1 << idx1)
                if sub == 0:
                    break
                sub = (sub - 1) & rest_mask
            masks.append(m)
    else:
        masks = []
        for e in edges:
            e_set = set(e)
            base1 = 0
            for v in e:
                base1 |= bit[v]
            rest = [v for v in range(n) if v not in e_set]
            r = len(rest)
            m = 0
            for p in range(1 << r):
                idx0 = 0
                idx1 = base1
                for i, v in enumerate(rest):
                    if (p >> i) & 1:
                        b = bit[v]
                        idx0 |= b
                        idx1 |= b
                m |= (1 << idx0) | (1 << idx1)
            masks.append(m)
    return edges, masks


def _greedy_cover(n, masks, full, deadline, rng, randomize=0.0, lookahead=0):
    """Greedy set cover with optional randomization and look-ahead scoring.
    When lookahead > 0, scores each candidate by (immediate coverage, -future cost)."""
    uncovered = full
    chosen = []
    chosen_masks = []
    remaining = list(range(len(masks)))
    while uncovered:
        if time.time() > deadline:
            return None, None
        best = -1
        bestc = 0
        best_score = -1
        use_lookahead = lookahead > 0 and len(remaining) > 100
        if len(remaining) > 800:
            sample = rng.sample(remaining, min(800, len(remaining)))
        else:
            sample = remaining
        for ei in sample:
            c = (uncovered & masks[ei]).bit_count()
            if c == 0:
                continue
            if use_lookahead:
                # Lookahead: after picking ei, how much would the next best pick cover?
                unc2 = uncovered & ~masks[ei]
                if unc2 == 0:
                    score = c * 1000  # huge bonus if this finishes it
                else:
                    # Estimate next best coverage
                    next_best = 0
                    for ej in sample[:50]:
                        if ej == ei:
                            continue
                        nc = (unc2 & masks[ej]).bit_count()
                        if nc > next_best:
                            next_best = nc
                    score = c * 100 - next_best
            else:
                score = c
            if score > best_score or (score == best_score and bestc > 0 and randomize > 0 and rng.random() < randomize):
                best_score = score
                bestc = c
                best = ei
                if bestc == uncovered.bit_count():
                    break
        if best < 0:
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


def _local_prune(chosen, chosen_masks, full, deadline, rng, max_rounds=20):
    """Iteratively remove any edge whose absence still covers all colourings.
    Uses incremental coverage tracking for speed."""
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
            # Check if removing chosen[i] still covers full
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


def _pair_prune(chosen, chosen_masks, masks, full, deadline, rng, max_rounds=10):
    """Try to replace pairs of chosen edges with a single edge that covers their union.
    Also try to remove pairs where both are redundant."""
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        n_ch = len(chosen)
        if n_ch < 2:
            break
        # Build a set of chosen indices for O(1) lookup
        chosen_set = set(chosen)
        order = list(range(n_ch))
        rng.shuffle(order)
        for i in order:
            if i >= len(chosen):
                continue
            for k in range(i + 1, len(chosen)):
                if time.time() > deadline:
                    return chosen, chosen_masks
                # Compute what i and k uniquely cover
                others = 0
                for j in range(len(chosen)):
                    if j != i and j != k:
                        others |= chosen_masks[j]
                need = (chosen_masks[i] | chosen_masks[k]) & ~others
                if need == 0:
                    # Both redundant
                    del chosen[max(i, k)]
                    del chosen[min(i, k)]
                    del chosen_masks[max(i, k)]
                    del chosen_masks[min(i, k)]
                    improved = True
                    break
                # Try to find a single edge covering need
                if need.bit_count() > 0 and len(chosen) <= 50:
                    for j in range(len(masks)):
                        if j in chosen_set:
                            continue
                        if (masks[j] & need) == need:
                            # Replace i,k with j
                            del chosen[max(i, k)]
                            del chosen[min(i, k)]
                            del chosen_masks[max(i, k)]
                            del chosen_masks[min(i, k)]
                            chosen.append(j)
                            chosen_masks.append(masks[j])
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return chosen, chosen_masks


def _swap_improve(chosen, chosen_masks, masks, full, deadline, rng, max_rounds=15):
    """Try single-edge replacements and pair-to-single replacements."""
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        n_ch = len(chosen)
        if n_ch < 2:
            break
        order = list(range(n_ch))
        rng.shuffle(order)
        for i in order:
            if i >= len(chosen):
                continue
            mi = chosen_masks[i]
            others_cover = 0
            for j in range(len(chosen)):
                if j != i:
                    others_cover |= chosen_masks[j]
            unique_i = mi & ~others_cover
            if unique_i == 0:
                del chosen[i]
                del chosen_masks[i]
                improved = True
                break
            if len(chosen) <= 60:
                chosen_set = set(chosen)
                for j in range(len(masks)):
                    if j in chosen_set:
                        continue
                    mj = masks[j]
                    if (others_cover | mj) == full:
                        chosen[i] = j
                        chosen_masks[i] = mj
                        improved = True
                        break
        if improved:
            continue
        # Pair replacement
        if len(chosen) <= 45 and time.time() < deadline:
            for i in range(len(chosen)):
                if time.time() > deadline:
                    break
                for k in range(i + 1, len(chosen)):
                    others = 0
                    for j in range(len(chosen)):
                        if j != i and j != k:
                            others |= chosen_masks[j]
                    need = (chosen_masks[i] | chosen_masks[k]) & ~others
                    if need == 0:
                        del chosen[max(i, k)]
                        del chosen[min(i, k)]
                        del chosen_masks[max(i, k)]
                        del chosen_masks[min(i, k)]
                        improved = True
                        break
                    if need.bit_count() > 0:
                        chosen_set = set(chosen)
                        for j in range(len(masks)):
                            if j in chosen_set:
                                continue
                            if (masks[j] & need) == need:
                                del chosen[max(i, k)]
                                del chosen[min(i, k)]
                                del chosen_masks[max(i, k)]
                                del chosen_masks[min(i, k)]
                                chosen.append(j)
                                chosen_masks.append(masks[j])
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    return chosen, chosen_masks


def _try_n(n, deadline, rng):
    edges, masks = _build_masks_fast(n)
    full = (1 << (1 << n)) - 1
    best = None
    best_len = float('inf')
    # More restarts with varying strategies
    restarts = 18
    for r in range(restarts):
        if time.time() > deadline:
            break
        randomize = 0.1 + 0.2 * r / restarts
        lookahead = 1 if r % 3 == 0 else 0
        chosen, chosen_masks = _greedy_cover(n, masks, full, deadline, rng, randomize, lookahead)
        if chosen is None:
            continue
        chosen, chosen_masks = _local_prune(chosen, chosen_masks, full, deadline, rng)
        chosen, chosen_masks = _swap_improve(chosen, chosen_masks, masks, full, deadline, rng)
        chosen, chosen_masks = _pair_prune(chosen, chosen_masks, masks, full, deadline, rng)
        chosen, chosen_masks = _local_prune(chosen, chosen_masks, full, deadline, rng, max_rounds=10)
        if time.time() < deadline:
            chosen, chosen_masks = _swap_improve(chosen, chosen_masks, masks, full, deadline, rng, max_rounds=8)
            chosen, chosen_masks = _pair_prune(chosen, chosen_masks, masks, full, deadline, rng, max_rounds=5)
            chosen, chosen_masks = _local_prune(chosen, chosen_masks, full, deadline, rng, max_rounds=8)
        if len(chosen) < best_len:
            best_len = len(chosen)
            best = (list(chosen), list(chosen_masks))
            if best_len <= 30:
                break
    if best is None:
        return None
    chosen, _ = best
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

    fallback = {'n': 9,
                'edges': [list(e) for e in combinations(range(9), 5)]}

    best = fallback
    for n in (9, 10, 11, 12):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
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
