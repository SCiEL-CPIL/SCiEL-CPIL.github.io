"""Combined: bitmask set-cover with faster mask construction, smarter greedy,
swap-improvement local search, and multi-restart randomization."""
import time
import random
from itertools import combinations

_WALL_LIMIT = 150.0
_MARGIN = 15.0


def _build_masks_fast(n):
    """Build edge masks using precomputed vertex bitmasks for speed."""
    edges = [tuple(sorted(e)) for e in combinations(range(n), 5)]
    # Precompute: for each vertex v, the bitmask contribution
    # For an edge with vertex set S, the monochromatic colorings are those
    # where all vertices in S have the same color.
    # Total colorings: 2^n. For edge S (|S|=5), mono colorings = 2^(n-5) * 2 = 2^(n-4)
    # We need to build a 2^n-bit integer where bit c is 1 iff coloring c makes S monochromatic.
    
    # Faster approach: for each edge, iterate over all 2^n colorings is too slow.
    # Instead: a coloring is monochromatic on S iff all bits in S are equal.
    # Number of such colorings: 2^(n-5) * 2.
    # We can enumerate: fix S to all-0 or all-1, vary the rest.
    
    masks = []
    for e in edges:
        vs = list(e)
        vs_set = set(vs)
        rest = [v for v in range(n) if v not in vs_set]
        r = len(rest)
        m = 0
        # All-0 on S: rest varies freely
        for p in range(1 << r):
            idx0 = 0
            for i, v in enumerate(rest):
                if (p >> i) & 1:
                    idx0 |= (1 << v)
            m |= (1 << idx0)
        # All-1 on S: rest varies freely
        base1 = 0
        for v in vs:
            base1 |= (1 << v)
        for p in range(1 << r):
            idx1 = base1
            for i, v in enumerate(rest):
                if (p >> i) & 1:
                    idx1 |= (1 << v)
            m |= (1 << idx1)
        masks.append(m)
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


def _swap_improve(chosen, chosen_masks, masks, full, deadline, rng, max_rounds=30):
    """Try to reduce edge count by swapping: remove one chosen, add one not chosen."""
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        chosen_set = set(chosen)
        order = list(range(len(chosen)))
        rng.shuffle(order)
        for i in order:
            if i >= len(chosen):
                continue
            # Try removing chosen[i]
            u_without = 0
            for j in range(len(chosen)):
                if j != i:
                    u_without |= chosen_masks[j]
            # Find a single edge not in chosen that covers the gap
            gap = full & ~u_without
            if gap == 0:
                # Can just remove
                del chosen[i]
                del chosen_masks[i]
                improved = True
                break
            # Look for an edge that covers all of gap
            best_swap = -1
            best_extra = float('inf')
            for ei in range(len(masks)):
                if ei in chosen_set:
                    continue
                covers = (gap & masks[ei]) == gap
                if covers:
                    # Count how many of gap this edge covers (should be all)
                    extra = (masks[ei] & ~u_without).bit_count()
                    if extra < best_extra:
                        best_extra = extra
                        best_swap = ei
            if best_swap >= 0:
                del chosen[i]
                del chosen_masks[i]
                chosen.append(best_swap)
                chosen_masks.append(masks[best_swap])
                chosen_set = set(chosen)
                improved = True
                break
    return chosen, chosen_masks


def _local_prune(chosen, chosen_masks, full, deadline, rng, max_rounds=30):
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
    edges, masks = _build_masks_fast(n)
    full = (1 << (1 << n)) - 1
    best = None
    best_len = float('inf')
    # Multiple restarts with varying randomization
    restarts = 10
    for r in range(restarts):
        if time.time() > deadline:
            break
        randomize = 0.05 + 0.15 * (r + 1) / restarts
        chosen, chosen_masks = _greedy_cover(n, masks, full, deadline, rng, randomize)
        if chosen is None:
            continue
        # Local prune to remove redundant edges
        chosen, chosen_masks = _local_prune(chosen, chosen_masks, full, deadline, rng)
        # Swap improvement: try to replace edges to reduce count
        chosen, chosen_masks = _swap_improve(chosen, chosen_masks, masks, full, deadline, rng)
        # Another prune pass after swaps
        chosen, chosen_masks = _local_prune(chosen, chosen_masks, full, deadline, rng)
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
    # Try n=11 first (most likely to find good solutions), then others
    for n in (11, 10, 12, 9):
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
