"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 256  # numbers 1..256, indexed 0..255

    # Precompute conflict: (a*b) % (a+b) == 0 for distinct a, b
    # comp_mask[v] = bitmask of vertices that conflict with v (v is index 0..255, number = v+1)
    comp_mask = [0] * N
    for a_idx in range(N):
        a = a_idx + 1
        mask = 0
        for b_idx in range(N):
            if a_idx == b_idx:
                continue
            b = b_idx + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << b_idx)
        comp_mask[a_idx] = mask

    # Degree of each vertex
    deg = [comp_mask[i].bit_count() for i in range(N)]

    def greedy_max_independent_set(order):
        """Greedy MIS: process vertices in given order, add if no conflict with already-added."""
        selected = set()
        for v in order:
            can_add = True
            for x in selected:
                if comp_mask[v] & (1 << x):
                    can_add = False
                    break
            if can_add:
                selected.add(v)
        return selected

    # Try multiple greedy orderings
    best_set = set()
    best_size = 0

    # Ordering 1: minimum degree first
    order_min_deg = sorted(range(N), key=lambda v: (deg[v], v))
    s = greedy_max_independent_set(order_min_deg)
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)

    # Ordering 2: maximum degree first
    order_max_deg = sorted(range(N), key=lambda v: (-deg[v], v))
    s = greedy_max_independent_set(order_max_deg)
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)

    # Ordering 3: random shuffles
    for trial in range(50):
        if time.time() > deadline - 15:
            break
        order = list(range(N))
        random.shuffle(order)
        s = greedy_max_independent_set(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)

    # Local search: try to improve by swapping
    current = set(best_set)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try removing each element and see if we can add more
        elems = list(current)
        random.shuffle(elems)
        for i in range(len(elems)):
            if time.time() > deadline - 10:
                break
            a = elems[i]
            trial_set = current - {a}
            # Try to add all possible candidates
            candidates = [x for x in range(N) if x not in trial_set]
            random.shuffle(candidates)
            added = 0
            for c in candidates:
                if time.time() > deadline - 10:
                    break
                ok = True
                for x in trial_set:
                    if comp_mask[c] & (1 << x):
                        ok = False
                        break
                if ok:
                    trial_set.add(c)
                    added += 1
            if len(trial_set) > len(current):
                current = trial_set
                improved = True
                break

    best_set = current
    best_size = len(best_set)

    # Try a simple branch-and-bound with bitsets for small subproblems
    # Actually, let's try a more sophisticated local search with 2-swap

    current = set(best_set)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        elems = list(current)
        random.shuffle(elems)
        for i in range(min(len(elems), 30)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, min(len(elems), 30)):
                if time.time() > deadline - 5:
                    break
                a, b = elems[i], elems[j]
                trial_set = current - {a, b}
                candidates = [x for x in range(N) if x not in trial_set]
                random.shuffle(candidates)
                added = 0
                for c in candidates:
                    if time.time() > deadline - 5:
                        break
                    ok = True
                    for x in trial_set:
                        if comp_mask[c] & (1 << x):
                            ok = False
                            break
                    if ok:
                        trial_set.add(c)
                        added += 1
                        if added >= 3:
                            break
                if len(trial_set) > len(current):
                    current = trial_set
                    improved = True
                    break
            if improved:
                break

    best_set = current
    best_size = len(best_set)

    # Convert to list of numbers (1-indexed)
    result = sorted(v + 1 for v in best_set)

    if not result:
        return [1]

    return result
# EVOLVE-BLOCK-END
