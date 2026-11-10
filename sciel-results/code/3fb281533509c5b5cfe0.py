"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 45.0

    elements = list(range(24, 0, -1))
    n = len(elements)
    recips = {x: Fraction(1, x) for x in elements}

    best_set = []
    best_size = 0

    def check_time():
        return time.time() > deadline

    def branch_and_bound(idx, current_set, sums_set, reach_flags):
        nonlocal best_set, best_size

        if check_time():
            return

        if len(current_set) + (n - idx) <= best_size:
            return

        if idx == n:
            if len(current_set) > best_size:
                best_set = current_set[:]
                best_size = len(current_set)
            return

        x = elements[idx]
        rx = recips[x]

        if not reach_flags[x - 1]:
            new_sums = set(sums_set)
            new_flags = list(reach_flags)
            new_flags[x - 1] = True

            for s in sums_set:
                nv = s + rx
                new_sums.add(nv)
                if nv > 0:
                    k_cand = round(1.0 / float(nv))
                    if 1 <= k_cand <= 24:
                        if nv == Fraction(1, k_cand):
                            new_flags[k_cand - 1] = True

            current_set.append(x)
            branch_and_bound(idx + 1, current_set, new_sums, tuple(new_flags))
            current_set.pop()

        branch_and_bound(idx + 1, current_set, sums_set, reach_flags)

    branch_and_bound(0, [], {Fraction(0)}, (False,) * 24)

    if not best_set:
        return [24]
    return sorted(best_set)
# EVOLVE-BLOCK-END
