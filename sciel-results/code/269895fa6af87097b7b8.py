# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero
    quadratic residue modulo 205. Uses local search with simulated annealing
    to maximize the set size.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin for 630s limit

    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Precompute forbidden as a boolean array for speed
    forbidden_arr = np.zeros(m, dtype=bool)
    for f in forbidden:
        forbidden_arr[f] = True

    def is_valid_pair(a, b):
        d = (a - b) % m
        return not forbidden_arr[d]

    def is_valid_subset(subset):
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                if not is_valid_pair(subset[i], subset[j]):
                    return False
        return True

    # Greedy initial solution
    def greedy_init(order):
        selected = []
        for candidate in order:
            ok = True
            for s in selected:
                if not is_valid_pair(candidate, s):
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected

    # Local search with add, remove, swap moves
    def local_search(subset, max_iters=50000):
        current = list(subset)
        current_set = set(current)
        best = list(current)
        best_size = len(current)

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            improved = False

            # Try adding an element
            for cand in range(m):
                if cand in current_set:
                    continue
                ok = True
                for s in current:
                    if not is_valid_pair(cand, s):
                        ok = False
                        break
                if ok:
                    current.append(cand)
                    current_set.add(cand)
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
                    improved = True
                    break

            if improved:
                continue

            # Try removing one element and adding two
            for i in range(len(current)):
                removed = current[i]
                temp = [x for j, x in enumerate(current) if j != i]
                temp_set = set(temp)

                # Try to add two elements
                added = []
                for cand in range(m):
                    if cand in temp_set or cand == removed:
                        continue
                    ok = True
                    for s in temp:
                        if not is_valid_pair(cand, s):
                            ok = False
                            break
                    if ok and len(added) < 2:
                        added.append(cand)
                        temp_set.add(cand)

                if len(added) >= 2:
                    new_current = temp + added
                    if len(new_current) > best_size:
                        current = new_current
                        current_set = set(current)
                        best = list(current)
                        best_size = len(current)
                        improved = True
                        break

            if improved:
                continue

            # Try swap: remove one, add one different
            for i in range(len(current)):
                removed = current[i]
                temp = [x for j, x in enumerate(current) if j != i]
                temp_set = set(temp)

                for cand in range(m):
                    if cand in temp_set or cand == removed:
                        continue
                    ok = True
                    for s in temp:
                        if not is_valid_pair(cand, s):
                            ok = False
                            break
                    if ok:
                        new_current = temp + [cand]
                        if len(new_current) > best_size:
                            current = new_current
                            current_set = set(current)
                            best = list(current)
                            best_size = len(current)
                            improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return best

    # Generate multiple initial solutions and run local search on each
    best_solution = []
    best_size = 0

    # Greedy init with natural order
    sol = greedy_init(list(range(m)))
    if len(sol) > best_size:
        best_size = len(sol)
        best_solution = list(sol)

    # Try different random orderings for greedy
    rng = random.Random(42)
    for trial in range(200):
        if time.time() > deadline - 60:
            break
        order = list(range(m))
        rng.shuffle(order)
        sol = greedy_init(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = list(sol)

    # Local search from best so far
    if time.time() < deadline - 60:
        improved = local_search(best_solution, max_iters=30000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = list(improved)

    # More local search with different random restarts
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        order = list(range(m))
        rng.shuffle(order)
        sol = greedy_init(order)
        if len(sol) >= best_size:
            improved = local_search(sol, max_iters=10000)
            if len(improved) > best_size:
                best_size = len(improved)
                best_solution = list(improved)

    return np.array(sorted(best_solution), dtype=np.int64)


# EVOLVE-BLOCK-END
