"""Combined program: fast greedy + enhanced local search with swaps."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_DEADLINE = 82.0  # safety margin before 90s kill


def _can_add(x, selected):
    """Check if x can be added to selected without creating a violation."""
    n = len(selected)
    # Check 1: x divides sum of any 3 distinct elements in selected
    if n >= 3:
        for i in range(n):
            si = selected[i]
            for j in range(i + 1, n):
                sj = selected[j]
                s_ij = si + sj
                for k in range(j + 1, n):
                    if (s_ij + selected[k]) % x == 0:
                        return False

    # Check 2: For each a in selected, a | (x + b + c) for some b,c in selected\{a}
    for a in selected:
        for i in range(n):
            if selected[i] == a:
                continue
            bi = selected[i]
            for j in range(n):
                if j == i or selected[j] == a:
                    continue
                cj = selected[j]
                if (x + bi + cj) % a == 0:
                    return False
    return True


def _check_validity(selected):
    """Check if the set is valid."""
    n = len(selected)
    if n < 4:
        return True
    for a in selected:
        others = [v for v in selected if v != a]
        for triple in combinations(others, 3):
            if (triple[0] + triple[1] + triple[2]) % a == 0:
                return False
    return True


def _greedy_largest_first():
    """Greedy that picks largest elements first."""
    selected = []
    for x in range(_N, 0, -1):
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected


def _greedy_smallest_first():
    """Greedy that picks smallest elements first."""
    selected = []
    for x in range(1, _N + 1):
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected


def _greedy_random(seed):
    """Greedy with random ordering."""
    rng = random.Random(seed)
    selected = []
    candidates = list(range(1, _N + 1))
    rng.shuffle(candidates)
    for x in candidates:
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected


def _local_search(selected, time_limit=15.0):
    """Enhanced local search: add, 1-for-1 swap, 2-for-2 swap."""
    if not _check_validity(selected):
        return selected

    start = time.time()
    best = list(selected)
    best_set = set(best)

    while time.time() - start < time_limit:
        improved = False
        candidates = [x for x in range(1, _N + 1) if x not in best_set]
        if not candidates:
            break

        # Try adding
        for x in candidates:
            if _can_add(x, best):
                best.append(x)
                best_set.add(x)
                improved = True
                break

        if not improved and len(best) >= 1:
            # Try 1-for-1 swap
            swap_done = False
            for i in range(len(best)):
                if time.time() - start > time_limit:
                    break
                removed = best[i]
                temp_list = best[:i] + best[i + 1:]

                for x in candidates:
                    if x in best_set:
                        continue
                    if _can_add(x, temp_list):
                        best = temp_list + [x]
                        best_set = best_set - {removed} | {x}
                        swap_done = True
                        break

                if swap_done:
                    break

            improved = swap_done

        if not improved and len(best) >= 2:
            # Try 2-for-2 swap
            found = False
            for i in range(len(best)):
                if time.time() - start > time_limit:
                    break
                for j in range(i + 1, len(best)):
                    r1, r2 = best[i], best[j]
                    temp_list = [v for k, v in enumerate(best) if k != i and k != j]

                    added_count = 0
                    for x in candidates:
                        if x in best_set:
                            continue
                        if _can_add(x, temp_list):
                            temp_list.append(x)
                            added_count += 1
                            if added_count == 2:
                                break

                    if added_count > 0:
                        best = temp_list
                        best_set = set(best)
                        found = True
                        break
                if found:
                    break

            if not found:
                break

    return best


def construct():
    start_time = time.time()
    deadline = _DEADLINE

    # Generate diverse initial solutions
    initial_solutions = []

    # Largest first
    if time.time() - start_time < deadline - 30:
        sol = _greedy_largest_first()
        if sol:
            initial_solutions.append(sol)

    # Smallest first
    if time.time() - start_time < deadline - 30:
        sol = _greedy_smallest_first()
        if sol:
            initial_solutions.append(sol)

    # Random greedy with different seeds
    for seed in range(10):
        if time.time() - start_time < deadline - 25:
            sol = _greedy_random(seed)
            if sol:
                initial_solutions.append(sol)

    # Pick the best initial solution
    if not initial_solutions:
        return list(range(1, _MAX + 1))

    best_initial = max(initial_solutions, key=len)
    best = list(best_initial)

    # Run local search on the best initial
    if time.time() - start_time < deadline - 5:
        remaining = deadline - (time.time() - start_time) - 3
        if remaining > 2:
            improved = _local_search(best, time_limit=min(remaining, 20))
            if len(improved) > len(best):
                best = improved

    # If we still have time, try local search on other initial solutions
    if time.time() - start_time < deadline - 5:
        for sol in initial_solutions:
            if time.time() - start_time > deadline - 5:
                break
            if len(sol) >= len(best):
                continue
            remaining = deadline - (time.time() - start_time) - 3
            if remaining > 1:
                improved = _local_search(sol, time_limit=min(remaining, 5))
                if len(improved) > len(best):
                    best = improved

    # Final check and return
    if not _check_validity(best):
        # Fallback: rebuild from scratch with largest-first
        best = _greedy_largest_first()

    return best
# EVOLVE-BLOCK-END
