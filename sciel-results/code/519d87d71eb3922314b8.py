"""Reciprocal-subsum-free set search: exact scaled-integer validity + greedy + beam + local polish."""
import time
import itertools
from fractions import Fraction

# ---- Precompute scaled reciprocal integers (exact) ----
# L = lcm(1..24). 1/x is a positive integer scaled by L.
_LCM = 1
for _i in range(1, 25):
    _g = _LCM
    _a, _b = _LCM, _i
    while _b:
        _a, _b = _b, _a % _b
    _g = _a
    _LCM = _LCM // _g * _i

# scaled[i] = L / i  for i in 1..24 (1-indexed)
_scaled = [0] + [_LCM // i for i in range(1, 25)]
_UNIVERSE = list(range(1, 25))
_UNIV_SET = frozenset(_UNIVERSE)


def _valid(selected):
    """Exact check: no reciprocal of a selected element equals sum of reciprocals
    of a subset of the OTHER selected elements."""
    vals = [_scaled[x] for x in selected]
    n = len(vals)
    # For each target index t, check if vals[t] is a subset sum of the other n-1 vals.
    for t in range(n):
        target = vals[t]
        others = vals[:t] + vals[t + 1:]
        # Meet-in-the-middle subset sum: does any subset of others sum to target?
        m = len(others)
        # quick prune: if sum(others) < target -> impossible
        if sum(others) < target:
            continue
        # split
        half = m // 2
        left = others[:half]
        right = others[half:]
        left_sums = {0}
        for v in left:
            left_sums |= {s + v for s in list(left_sums)}
        right_sums = {0}
        for v in right:
            right_sums |= {s + v for s in list(right_sums)}
        # check intersection
        for s in left_sums:
            if (target - s) in right_sums:
                return False
    return True


def _valid_add(selected_set, x):
    """Check whether adding x to selected_set keeps validity.
    Only need to check: (a) x's reciprocal vs subset of selected, and
    (b) each selected element's reciprocal vs subset of (selected + x) excluding itself.
    For (b), only subsets that include x can be new, i.e. 1/sel == 1/x + sum(subset of selected\{sel}).
    """
    if x in selected_set:
        return False
    sx = _scaled[x]
    sel = list(selected_set)
    sv = [_scaled[s] for s in sel]
    # (a) sx must not be a subset sum of sv
    if _is_subset_sum(sv, sx):
        return False
    # (b) for each sel element, check 1/sel == 1/x + sum(subset of sel\{sel})
    for i, s in enumerate(sel):
        target = sv[i] - sx  # need sum(subset of others) == target
        if target < 0:
            continue
        others = sv[:i] + sv[i + 1:]
        if _is_subset_sum(others, target):
            return False
    return True


def _is_subset_sum(vals, target):
    """Exact: is target a subset sum of vals (positive ints)?"""
    if target == 0:
        return True
    if sum(vals) < target:
        return False
    m = len(vals)
    half = m // 2
    left = vals[:half]
    right = vals[half:]
    left_sums = {0}
    for v in left:
        left_sums |= {s + v for s in list(left_sums)}
    right_sums = {0}
    for v in right:
        right_sums |= {s + v for s in list(right_sums)}
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False


def _greedy_construct():
    """Monotone greedy: add elements in decreasing order if valid."""
    selected = set()
    for x in sorted(_UNIVERSE, reverse=True):
        if _valid_add(selected, x):
            selected.add(x)
    return selected


def _beam_search(deadline, margin=0.5):
    """Beam search over add/swap moves with exact validity."""
    start = time.time()
    best = _greedy_construct()
    beam = [best]
    # also seed a few alternative greedy orders
    for start_el in (24, 12, 8, 6, 4, 3, 2):
        sel = set()
        if _valid_add(sel, start_el):
            sel.add(start_el)
        for x in sorted(_UNIVERSE, reverse=True):
            if x != start_el and _valid_add(sel, x):
                sel.add(x)
        beam.append(sel)

    beam_width = 8
    max_iters = 400
    for it in range(max_iters):
        if time.time() - start > deadline - margin:
            break
        candidates = []
        for sel in beam:
            # add moves
            for x in _UNIVERSE:
                if x not in sel and _valid_add(sel, x):
                    candidates.add = None
                    candidates.append(sel | {x})
            # swap moves: remove one, try to add
            for a in sel:
                base = sel - {a}
                for x in _UNIVERSE:
                    if x not in base and x != a and _valid_add(base, x):
                        candidates.append(base | {x})
        if not candidates:
            break
        # dedupe and sort by size desc
        uniq = []
        seen = set()
        for c in candidates:
            key = frozenset(c)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        uniq.sort(key=lambda s: (-len(s), sum(s)))
        beam = uniq[:beam_width]
        if beam and len(beam[0]) > len(best):
            best = beam[0]
    return best


def _local_polish(selected, deadline, margin=0.5):
    """Try to add more elements; also try removing one and adding two."""
    start = time.time()
    sel = set(selected)
    improved = True
    while improved:
        if time.time() - start > deadline - margin:
            break
        improved = False
        # add
        for x in _UNIVERSE:
            if x not in sel and _valid_add(sel, x):
                sel.add(x)
                improved = True
                break
        # swap: remove a, add b, c
        for a in list(sel):
            base = sel - {a}
            # try adding two
            for b in _UNIVERSE:
                if b in base or b == a:
                    continue
                if not _valid_add(base, b):
                    continue
                mid = base | {b}
                for c in _UNIVERSE:
                    if c in mid or c == a:
                        continue
                    if _valid_add(mid, c):
                        sel = mid | {c}
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return sel


def construct():
    deadline = time.time() + 85.0  # safety margin under 90s
    best = _greedy_construct()
    # beam search
    bs = _beam_search(deadline)
    if len(bs) > len(best):
        best = bs
    # local polish on best
    polished = _local_polish(best, deadline)
    if len(polished) > len(best):
        best = polished
    # final safety: verify validity
    if not _valid(best):
        # fallback to known valid seed
        return list(range(13, 25))
    return sorted(best)
