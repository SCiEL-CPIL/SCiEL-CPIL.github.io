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
    For each 5-subset E, mask bits are set at all colourings where E is monochromatic.
    Uses precomputed rest-bit arrays for speed."""
    edges = [tuple(e) for e in combinations(range(n), 5)]
    bit = [1 << v for v in range(n)]
    # Precompute: for each vertex v, its bit value
    masks = []
    for e in edges:
        e_set = set(e)
        base1 = 0
        for v in e:
            base1 |= bit[v]
        rest = [v for v in range(n) if v not in e_set]
        r = len(rest)
        # Precompute the list of bits for rest vertices
        rest_bits = [bit[v] for v in rest]
        m = 0
        # For each subset p of rest, idx0 = OR of bits in p, idx1 = base1 | idx0
        # Build incrementally using Gray code or just iterate
        for p in range(1 << r):
            idx0 = 0
            # Use bit manipulation: extract set bits of p
            pp = p
            i = 0
            while pp:
                if pp & 1:
                    idx0 |= rest_bits[i]
                pp >>= 1
                i += 1
            # Handle remaining bits (i through r-1) that are 0 in p
            # Actually the while loop handles all set bits; unset bits contribute 0
            idx1 = base1 | idx0
            m |= (1 << idx0) | (1 << idx1)
        masks.append(m)
    return edges, masks


def _greedy_cover(n, masks, full, deadline, rng, randomize=0.0):
    """Greedy set cover with coverage-based scoring and randomization."""
    uncovered = full
    chosen = []
    chosen_masks = []
    chosen_set = set()
    remaining = list(range(len(masks)))
    full_bits = full.bit_count()
    while uncovered:
        if time.time() > deadline:
            return None, None
        best = -1
        bestc = 0
        # Use larger sample for better quality
        if len(remaining) > 1000:
            sample = rng.sample(remaining, min(1000, len(remaining)))
        elif len(remaining) > 300:
            sample = rng.sample(remaining, min(300, len(remaining)))
        else:
            sample = remaining
        unc_count = uncovered.bit_count()
        for ei in sample:
            c = (uncovered & masks[ei]).bit_count()
            if c > bestc or (c == bestc and bestc > 0 and randomize > 0 and rng.random() < randomize):
                bestc = c
                best = ei
                if bestc == unc_count:
                    break
        if best < 0:
            # Fallback: scan all
            for ei in remaining:
                c = (uncovered & masks[ei]).bit_count()
                if c > bestc:
                    bestc = c
                    best = ei
        if best < 0 or bestc == 0:
            break
        chosen.append(best)
        chosen_masks.append(masks[best])
        chosen_set.add(best)
        uncovered &= ~masks[best]
        remaining.remove(best)
    if uncovered == 0:
        return chosen, chosen_masks
    return None, None


def _local_prune(chosen, chosen_masks, full, deadline, rng, max_rounds=20):
    """Iteratively remove any edge whose absence still covers all colourings.
    Uses prefix/suffix OR for O(k) per round."""
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        k = len(chosen)
        if k < 2:
            break
        # Compute prefix OR: pre[i] = OR of chosen_masks[0..i-1]
        pre = [0] * (k + 1)
        for i in range(k):
            pre[i + 1] = pre[i] | chosen_masks[i]
        # Compute suffix OR: suf[i] = OR of chosen_masks[i+1..k-1]
        suf = [0] * (k + 1)
        for i in range(k - 1, -1, -1):
            suf[i] = suf[i + 1] | chosen_masks[i]
        # Shuffle order for diversity
        order = list(range(k))
        rng.shuffle(order)
        for i in order:
            if i >= len(chosen):
                continue
            # Union of all except i
            u = pre[i] | suf[i + 1]
            if u == full:
                del chosen[i]
                del chosen_masks[i]
                improved = True
                break
    return chosen, chosen_masks


def _swap_improve(chosen, chosen_masks, masks, full, deadline, rng, max_rounds=10):
    """Try single-edge removals and pair-to-single replacements to reduce count."""
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        n_ch = len(chosen)
        if n_ch < 2:
            break
        # Compute prefix/suffix OR for quick "others" computation
        pre = [0] * (n_ch + 1)
        for i in range(n_ch):
            pre[i + 1] = pre[i] | chosen_masks[i]
        suf = [0] * (n_ch + 1)
        for i in range(n_ch - 1, -1, -1):
            suf[i] = suf[i + 1] | chosen_masks[i]

        # Pass 1: single-edge removal or replacement
        order = list(range(n_ch))
        rng.shuffle(order)
        for i in order:
            if i >= len(chosen) or time.time() > deadline:
                break
            others_cover = pre[i] | suf[i + 1]
            unique_i = chosen_masks[i] & ~others_cover
            if unique_i == 0:
                del chosen[i]
                del chosen_masks[i]
                improved = True
                break
            # Try to find replacement edge
            if len(chosen) <= 50:
                chosen_set = set(chosen)
                for j in range(len(masks)):
                    if j in chosen_set:
                        continue
                    if (others_cover | masks[j]) == full:
                        chosen[i] = j
                        chosen_masks[i] = masks[j]
                        improved = True
                        break
                if improved:
                    break

        if improved:
            continue

        # Pass 2: pair-to-single replacement (only for small n_ch)
        if len(chosen) <= 30 and time.time() < deadline:
            n_ch2 = len(chosen)
            # Recompute prefix/suffix
            pre2 = [0] * (n_ch2 + 1)
            for i in range(n_ch2):
                pre2[i + 1] = pre2[i] | chosen_masks[i]
            suf2 = [0] * (n_ch2 + 1)
            for i in range(n_ch2 - 1, -1, -1):
                suf2[i] = suf2[i + 1] | chosen_masks[i]
            for i in range(n_ch2):
                if time.time() > deadline:
                    break
                for k in range(i + 1, n_ch2):
                    others = pre2[i] | suf2[k + 1]
                    # Also need to exclude chosen[i] and chosen[k] from "others"
                    # pre2[i] covers 0..i-1, suf2[k+1] covers k+1..end
                    # But we also need to exclude i and k themselves
                    # Actually pre2[i] = OR of 0..i-1, suf2[k+1] = OR of k+1..end
                    # We need OR of all except i and k:
                    # = (OR of 0..i-1) | (OR of i+1..k-1) | (OR of k+1..end)
                    # pre2[i] gives 0..i-1, suf2[k+1] gives k+1..end
                    # Middle: OR of i+1..k-1
                    if i + 1 < k:
                        mid = 0
                        for j in range(i + 1, k):
                            mid |= chosen_masks[j]
                        others = pre2[i] | mid | suf2[k + 1]
                    else:
                        others = pre2[i] | suf2[k + 1]
                    need = (chosen_masks[i] | chosen_masks[k]) & ~others
                    if need == 0:
                        # Both removable
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
    # Multiple restarts with varying randomization
    restarts = 12
    for r in range(restarts):
        if time.time() > deadline:
            break
        randomize = 0.1 + 0.2 * (r + 1) / restarts
        chosen, chosen_masks = _greedy_cover(n, masks, full, deadline, rng, randomize)
        if chosen is None:
            continue
        chosen, chosen_masks = _local_prune(chosen, chosen_masks, full, deadline, rng)
        chosen, chosen_masks = _swap_improve(chosen, chosen_masks, masks, full, deadline, rng)
        # Another prune round after swaps
        chosen, chosen_masks = _local_prune(chosen, chosen_masks, full, deadline, rng, max_rounds=15)
        if len(chosen) < best_len:
            best_len = len(chosen)
            best = (list(chosen), list(chosen_masks))
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
    # Try n values; n=11 and n=12 are most promising for few edges
    # Allocate time: n=11 gets most, then n=12, then n=10, n=9
    for n in (11, 12, 10, 9):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 8:
            break
        try:
            res = _try_n(n, deadline, rng)
        except Exception:
            res = None
        if res is not None and len(res['edges']) < len(best['edges']):
            best = res
            # If we found a very good solution, try to improve further
            if len(best['edges']) <= 55:
                # Use remaining time for additional restarts on same n
                if time.time() < deadline - 10:
                    try:
                        res2 = _try_n(n, deadline, rng)
                    except Exception:
                        res2 = None
                    if res2 is not None and len(res2['edges']) < len(best['edges']):
                        best = res2

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
