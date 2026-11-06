"""Factorial decomposition of 12! into distinct factors > 12, minimizing the largest factor.

Strategy: enumerate all divisors of 12! that are > 12 (and proper, i.e. < 12!/13 so they
can participate in a multi-factor product). Then do an ascending, strictly-increasing
backtracking search over these divisors (which guarantees distinctness and a canonical
order) to find, among ALL valid decompositions (1..12 factors), the one that minimizes
the largest factor. A wall-clock deadline guard returns the best found before the limit.
"""

import time

# ---- precompute the divisor universe (once, at import, cheap) ----
def _compute_universe():
    target = 479001600  # 12!
    # proper-divisor bound: a factor must leave room for >=1 other factor > 12
    max_factor = target // 13
    divs = []
    i = 1
    while i <= max_factor:
        if target % i == 0:
            j = target // i
            if i > 12:
                divs.append(i)
            if j != i and j > 12:
                divs.append(j)
        i += 1
    divs.sort()
    return target, divs, set(divs)

_TARGET, _DIVS, _DIVSET = _compute_universe()


def construct():
    deadline = time.time() + 80.0  # safety margin under the 90 s kill
    target = _TARGET
    divs = _DIVS
    divset = _DIVSET

    # best: (max_factor, count, tuple(sorted_factors))
    best = None

    def better(cand):
        nonlocal best
        if best is None:
            return True
        cm, cf = cand[0], cand[1]
        bm, bf = best[0], best[1]
        if cm < bm:
            return True
        if cm == bm:
            if cf < bf:
                return True
            if cf == bf and cand[2] < best[2]:
                return True
        return False

    def backtrack(rem, prev, acc, idx):
        nonlocal best
        # prune: if current running max already >= best max, stop
        if best is not None and acc and acc[-1] >= best[0]:
            return
        if rem == 1:
            if len(acc) >= 1:
                if better((acc[-1], len(acc), tuple(acc))):
                    best = (acc[-1], len(acc), tuple(acc))
            return
        # rem itself is a valid single remaining factor
        if rem > prev and rem in divset:
            if better((rem, len(acc) + 1, tuple(acc + [rem]))):
                best = (rem, len(acc) + 1, tuple(acc + [rem]))
            return
        # try adding a factor d from divs[idx:] with prev < d < rem
        for d in divs[idx:]:
            if d <= prev:
                continue
            if d >= rem:
                break
            if rem % d != 0:
                continue
            if best is not None and d >= best[0]:
                break  # ascending: all later d' >= d >= best max -> prune
            backtrack(rem // d, d, acc + [d], idx)
            if time.time() > deadline:
                return

    # seed with the trivial single-factor solution
    best = (target, 1, (target,))
    backtrack(target, 12, [], 0)

    if best is None:
        return [target]
    return list(best[2])


if __name__ == "__main__":
    print(construct())
