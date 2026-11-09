"""Reciprocal weight of three-progression-free sets in [1,256]."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85.0
    N = 256

    # Greedy: add smallest elements first that don't create a 3-AP
    in_set = [False] * (N + 1)
    selected = []

    def can_add_fast(x):
        # Check all 3-APs involving x
        # x as first: (x, x+d, x+2d)
        d = 1
        while x + 2*d <= N:
            if in_set[x+d] and in_set[x+2*d]:
                return False
            d += 1
        # x as middle: (x-d, x, x+d)
        d = 1
        while x - d >= 1 and x + d <= N:
            if in_set[x-d] and in_set[x+d]:
                return False
            d += 1
        # x as last: (x-2d, x-d, x)
        d = 1
        while x - 2*d >= 1:
            if in_set[x-2*d] and in_set[x-d]:
                return False
            d += 1
        return True

    for x in range(1, N + 1):
        if can_add_fast(x):
            in_set[x] = True
            selected.append(x)

    best_selected = list(selected)
    best_sum = sum(Fraction(1, x) for x in selected)

    # Local search with swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        curr_set = set(selected)

        # Try to improve by removing one element and adding another
        for i in range(len(selected) - 1, -1, -1):
            if time.time() > deadline:
                break
            rem = selected[i]
            in_set[rem] = False
            rem_set = curr_set - {rem}

            # Try adding smaller elements first
            for j in range(1, rem):
                if j in rem_set:
                    continue
                # Check if j can be added to rem_set
                ok = True
                # j as first
                d = 1
                while j + 2*d <= N:
                    if in_set[j+d] and in_set[j+2*d]:
                        ok = False
                        break
                    d += 1
                if not ok:
                    continue
                # j as middle
                d = 1
                while j - d >= 1 and j + d <= N:
                    if in_set[j-d] and in_set[j+d]:
                        ok = False
                        break
                    d += 1
                if not ok:
                    continue
                # j as last
                d = 1
                while j - 2*d >= 1:
                    if in_set[j-2*d] and in_set[j-d]:
                        ok = False
                        break
                    d += 1
                if not ok:
                    continue

                # Valid swap
                in_set[j] = True
                new_selected = [x for x in selected if x != rem]
                new_selected.append(j)
                new_sum = best_sum - Fraction(1, rem) + Fraction(1, j)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_selected = new_selected
                    selected = new_selected
                    curr_set = set(selected)
                    improved = True
                    break
                else:
                    in_set[j] = False

            in_set[rem] = True

        # Try adding an element by removing one
        for j in range(1, N + 1):
            if time.time() > deadline:
                break
            if in_set[j]:
                continue
            for i in range(len(selected)):
                rem = selected[i]
                in_set[rem] = False
                # Check if j can be added
                ok = True
                d = 1
                while j + 2*d <= N:
                    if in_set[j+d] and in_set[j+2*d]:
                        ok = False
                        break
                    d += 1
                if not ok:
                    in_set[rem] = True
                    continue
                d = 1
                while j - d >= 1 and j + d <= N:
                    if in_set[j-d] and in_set[j+d]:
                        ok = False
                        break
                    d += 1
                if not ok:
                    in_set[rem] = True
                    continue
                d = 1
                while j - 2*d >= 1:
                    if in_set[j-2*d] and in_set[j-d]:
                        ok = False
                        break
                    d += 1
                if not ok:
                    in_set[rem] = True
                    continue

                in_set[j] = True
                new_selected = [x for x in selected if x != rem]
                new_selected.append(j)
                new_sum = best_sum - Fraction(1, rem) + Fraction(1, j)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_selected = new_selected
                    selected = new_selected
                    improved = True
                    break
                else:
                    in_set[j] = False
                in_set[rem] = True

    return best_selected

# EVOLVE-BLOCK-END
