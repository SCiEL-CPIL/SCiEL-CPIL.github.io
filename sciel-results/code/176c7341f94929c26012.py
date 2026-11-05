"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before the 90s kill
    n = 128

    # Enumerate every equal-LCM triple (a < b < c) in [1,128].
    triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    triples.append((a - 1, b - 1, c - 1))  # 0-indexed
    m = len(triples)

    # Build conflict structure: for each pair, list of elements that form a bad triple with them
    pair_conflicts = {}
    for a, b, c in triples:
        ab = (a, b) if a < b else (b, a)
        ac = (a, c) if a < c else (c, a)
        bc = (b, c) if b < c else (c, b)
        pair_conflicts.setdefault(ab, []).append(c)
        pair_conflicts.setdefault(ac, []).append(b)
        pair_conflicts.setdefault(bc, []).append(a)

    def is_valid(S):
        for a, b, c in combinations(S, 3):
            if lcm(a, b) == lcm(a, c) == lcm(b, c):
                return False
        return True

    def greedy():
        selected = []
        for x in range(1, n + 1):
            ok = True
            for a, b in combinations(selected, 2):
                if lcm(a, b) == lcm(a, x) == lcm(b, x):
                    ok = False
                    break
            if ok:
                selected.append(x)
        return selected

    # Greedy incumbent (always valid, nonempty).
    best = greedy()

    if m == 0:
        return list(range(1, n + 1))

    # Exact maximum independent set via MILP (HiGHS).
    if time.time() - start < DEADLINE:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            c = -np.ones(n)  # minimize -sum x  <=>  maximize |S|
            A = np.zeros((m, n))
            for i, (a, b, cc) in enumerate(triples):
                A[i, a] = 1.0
                A[i, b] = 1.0
                A[i, cc] = 1.0
            cons = LinearConstraint(A, ub=2.0)   # no hyperedge fully chosen
            bounds = Bounds(0.0, 1.0)
            res = milp(c, constraints=cons, bounds=bounds, integrality=np.ones(n))
            if res is not None and res.x is not None:
                sol = [i + 1 for i in range(n) if res.x[i] > 0.5]
                if len(sol) > 0 and is_valid(sol) and len(sol) >= len(best):
                    best = sol
        except Exception:
            pass

    # Local search polish: try to add elements one at a time, or swap out one for two
    if time.time() - start < DEADLINE - 5:
        best_set = set(best)
        improved = True
        while improved and time.time() - start < DEADLINE - 5:
            improved = False
            # Try adding each missing element
            for x in range(1, n + 1):
                if x in best_set:
                    continue
                # Check if adding x creates any bad triple
                can_add = True
                for a, b in combinations(best_set, 2):
                    if lcm(a, b) == lcm(a, x) == lcm(b, x):
                        can_add = False
                        break
                if can_add:
                    best_set.add(x)
                    improved = True
                    break
            if not improved:
                # Try swapping: remove one element, add two others
                for x in list(best_set):
                    temp_set = best_set - {x}
                    # Find two elements that can both be added
                    candidates = []
                    for y in range(1, n + 1):
                        if y in best_set:
                            continue
                        can_add_y = True
                        for a, b in combinations(temp_set, 2):
                            if lcm(a, b) == lcm(a, y) == lcm(b, y):
                                can_add_y = False
                                break
                        if can_add_y:
                            candidates.append(y)
                    if len(candidates) >= 2:
                        # Try pairs of candidates
                        for y, z in combinations(candidates, 2):
                            new_set = temp_set | {y, z}
                            if is_valid(new_set):
                                best_set = new_set
                                improved = True
                                break
                if improved:
                    break

        if len(best_set) > len(best):
            best = list(best_set)

    return best
# EVOLVE-BLOCK-END
