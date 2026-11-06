"""Rewritten: greedy with small-first ordering + fast feasibility + polish."""
import time

N = 256
DEADLINE = 75.0  # safety margin under 90s wall-clock


def _has_3ap_with(S, x):
    """Return True if adding x to set S creates a 3-term AP (with x involved)."""
    s = S
    n = len(s)
    for i in range(n):
        a = s[i]
        # x, a, c : a = (x+c)/2 -> c = 2a - x
        c = 2 * a - x
        if c != x and c in s:
            return True
        # a, x, c : x = (a+c)/2 -> c = 2x - a
        c = 2 * x - a
        if c != x and c in s:
            return True
    return False


def construct():
    start = time.time()

    # Candidate ordering: try small numbers first (high reciprocal value).
    # We build a set greedily. Use a set for O(1) membership.
    selected = set()
    selected_list = []

    # Primary greedy: ascending order (small first)
    for x in range(1, N + 1):
        if time.time() - start > DEADLINE:
            break
        if not _has_3ap_with(selected, x):
            selected.add(x)
            selected_list.append(x)

    # Polish pass: try to add any missed numbers (in ascending order of value)
    # that can now fit, in case ordering affected feasibility.
    for x in range(1, N + 1):
        if time.time() - start > DEADLINE:
            break
        if x not in selected and not _has_3ap_with(selected, x):
            selected.add(x)
            selected_list.append(x)

    # If we still have time, try a local-improvement: swap to increase weight.
    # For each excluded x, see if removing some included y (larger) lets us add x.
    # This is a heuristic polish; keep it bounded.
    improved = True
    while improved and (time.time() - start) < DEADLINE - 5.0:
        improved = False
        # value of adding x is 1/x; cost is 1/y for removed y. Only worthwhile if 1/x > 1/y
        # i.e. x < y. So try small excluded x, remove large included y.
        for x in range(1, N + 1):
            if time.time() - start > DEADLINE - 5.0:
                break
            if x in selected:
                continue
            # find a y in selected with y > x such that (selected - {y}) + {x} is 3AP-free
            for y in sorted((v for v in selected if v > x), reverse=True):
                # tentative removal
                selected.discard(y)
                if not _has_3ap_with(selected, x):
                    selected.add(x)
                    improved = True
                    break
                else:
                    selected.add(y)  # restore
        # single pass per outer loop

    result = sorted(selected)
    if not result:
        return [1]
    return result


if __name__ == "__main__":
    from fractions import Fraction
    s = construct()
    total = sum(Fraction(1, v) for v in s)
    print(len(s), float(total), total.numerator, total.denominator)
