"""Rewritten program: minimize the strict deficit 1 - sum(1/d_i) over distinct
denominators 2..48.  Since the total reciprocal sum of ALL 47 denominators is
~6.43, we choose a small exclusion set E whose reciprocal sum is just above
(TOTAL - 1); the included sum is then just below 1.  A pruned branch-and-bound
DFS finds the exclusion set, then local add/swap refinement reduces the deficit
further.  Exact rational arithmetic; wall-clock deadline guard throughout."""

import time
from fractions import Fraction

DENOMS = list(range(2, 49))
N = len(DENOMS)
ONE = Fraction(1, 1)
TOTAL = sum(Fraction(1, d) for d in DENOMS)
TARGET = TOTAL - ONE


def _find_exclusion(values, target, deadline):
    """Pruned DFS: find subset of values with sum STRICTLY > target, minimizing
    the sum.  Returns (sum, set_of_indices) or None."""
    n = len(values)
    best = [None]  # (sum, frozenset)

    # sort by value descending to prune earlier (large items first)
    order = sorted(range(n), key=lambda i: -values[i])

    def dfs(idx, partial, chosen):
        if time.time() > deadline:
            return
        if partial > target:
            if best[0] is None or partial < best[0][0]:
                best[0] = (partial, frozenset(chosen))
            return  # adding more only increases sum
        if idx == n:
            return
        # remaining max possible sum; if partial + rest <= target, cannot exceed
        # (we still might not reach target; that's fine, we just return)
        v = values[order[idx]]
        # branch include
        chosen.append(order[idx])
        dfs(idx + 1, partial + v, chosen)
        chosen.pop()
        # branch exclude
        dfs(idx + 1, partial, chosen)

    dfs(0, Fraction(0, 1), [])
    if best[0] is None:
        return None
    return best[0][0], set(best[0][1])


def _refine(selected, values, deadline):
    """Local search: add / remove / swap to minimize deficit while sum < 1."""
    cur = sum(values[i] for i in selected)
    deficit = ONE - cur
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # add an excluded denominator
        best_add = None
        for j in range(N):
            if j in selected:
                continue
            ns = cur + values[j]
            if ns < ONE:
                nd = ONE - ns
                if best_add is None or nd < best_add[0]:
                    best_add = (nd, j)
        if best_add is not None and best_add[0] < deficit:
            selected.add(best_add[1])
            cur += values[best_add[1]]
            deficit = best_add[0]
            improved = True
            continue
        # remove if over 1
        if cur >= ONE:
            best_rem = None
            for j in selected:
                ns = cur - values[j]
                if ns < ONE:
                    nd = ONE - ns
                    if best_rem is None or nd < best_rem[0]:
                        best_rem = (nd, j)
            if best_rem is not None:
                selected.discard(best_rem[1])
                cur -= values[best_rem[1]]
                deficit = best_rem[0]
                improved = True
                continue
        # swap
        best_swap = None
        for a in selected:
            base = cur - values[a]
            for b in range(N):
                if b in selected:
                    continue
                ns = base + values[b]
                if ns < ONE:
                    nd = ONE - ns
                    if best_swap is None or nd < best_swap[0]:
                        best_swap = (nd, a, b)
        if best_swap is not None and best_swap[0] < deficit:
            selected.discard(best_swap[1])
            selected.add(best_swap[2])
            cur = cur - values[best_swap[1]] + values[best_swap[2]]
            deficit = best_swap[0]
            improved = True
    return selected


def construct():
    deadline = time.time() + 75.0
    values = [Fraction(1, d) for d in DENOMS]

    best = None
    best_deficit = None

    r = _find_exclusion(values, TARGET, deadline - 20.0)
    if r is not None:
        exc_sum, exc_pos = r
        included = set(range(N)) - exc_pos
        s = sum(values[i] for i in included)
        if s >= ONE:
            for i in sorted(included, key=lambda k: -values[k]):
                if s < ONE:
                    break
                included.discard(i)
                s -= values[i]
        if included:
            included = _refine(included, values, deadline - 8.0)
            d = ONE - sum(values[i] for i in included)
            best = included
            best_deficit = d

    if best is None or not best:
        # greedy fallback
        selected = set()
        s = Fraction(0, 1)
        for i, v in enumerate(values):
            if s + v < ONE:
                selected.add(i)
                s += v
        return sorted(selected)

    return sorted(best)


if __name__ == "__main__":
    idx = construct()
    s = sum(Fraction(1, DENOMS[i]) for i in idx)
    print("indices:", idx)
    print("sum:", s)
    print("deficit:", ONE - s)
    print("score (1/deficit):", (ONE - s) ** -1)
