import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 145  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_milp(elements):
        n = len(elements)
        if n == 0:
            return []

        elem_to_idx = {x: i for i, x in enumerate(elements)}

        # Collect constraints
        constraints_rows = []
        constraints_bounds = []

        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    if i == j:
                        row = [0] * n
                        row[i] = 1
                        row[k] = 1
                        constraints_rows.append(row)
                        constraints_bounds.append(1)
                    else:
                        row = [0] * n
                        row[i] = 1
                        row[j] = 1
                        row[k] = 1
                        constraints_rows.append(row)
                        constraints_bounds.append(2)

        if not constraints_rows:
            return list(elements)

        A = np.array(constraints_rows, dtype=float)
        b = np.array(constraints_bounds, dtype=float)

        c = -np.ones(n)

        lb = np.zeros(n)
        ub = np.ones(n)

        integrality = np.ones(n)

        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, 0, b),
                integrality=integrality,
                bounds=Bounds(lb, ub)
            )
            if res.status == 0:
                selected = [elements[i] for i in range(n) if res.x[i] > 0.5]
                return selected
        except Exception:
            pass

        return heuristic_solve(elements)

    def heuristic_solve(elements):
        n = len(elements)
        elem_to_idx = {x: i for i, x in enumerate(elements)}

        # Start with sign partition
        pos = [x for x in elements if x > 0]
        neg = [x for x in elements if x < 0]
        best_set = set(pos) if len(pos) >= len(neg) else set(neg)

        # Greedy add
        current = set(best_set)
        improved = True
        while improved:
            improved = False
            for val in elements:
                if val in current:
                    continue
                can_add = True
                for a in current:
                    if a + val in current:
                        can_add = False
                        break
                if can_add:
                    current.add(val)
                    improved = True

        if len(current) > len(best_set):
            best_set = current

        return list(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = deadline - time.time()
        if remaining < 10:
            elems = instance(size, seed)
            # Fallback: take all positives
            results.append([x for x in elems if x > 0])
            continue

        elems = instance(size, seed)
        res = solve_milp(elems)
        results.append(res)

    return results
