"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import numpy as np

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    # Safety margin: 80s for calculation, 10s buffer (evaluator kills at 90s)
    deadline = 80.0

    # Range [2, 256]
    nums = list(range(2, 257))
    n = len(nums)
    weights = [1.0 / x for x in nums]

    # Identify conflicting pairs where lcm(a, b) <= 256
    # We want to maximize sum(1/x) for a set with no such pairs.
    # This is a Maximum Weight Independent Set.

    conflict_pairs = []
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            aj = nums[j]
            if _lcm(ai, aj) <= 256:
                conflict_pairs.append((i, j))

    if time.time() - start_time > deadline:
        return [2]

    # Try MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix

        if not conflict_pairs:
            return nums

        c = np.array([-w for w in weights])  # Minimize negative weight

        # Build sparse constraint matrix
        m = len(conflict_pairs)
        rows = np.empty(2 * m, dtype=int)
        cols = np.empty(2 * m, dtype=int)
        vals = np.ones(2 * m)

        for k, (i, j) in enumerate(conflict_pairs):
            rows[2*k] = k
            cols[2*k] = i
            rows[2*k+1] = k
            cols[2*k+1] = j

        A = csr_matrix((vals, (rows, cols)), shape=(m, n))

        # Constraints: x_i + x_j <= 1 for all conflicts
        constraints = LinearConstraint(A, ub=1)
        bounds = Bounds(0, 1)
        integrality = np.ones(n)

        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

        if res.status == 0:  # Optimal solution found
            x = res.x
            selected = [nums[i] for i in range(n) if x[i] > 0.5]
            if selected:
                return selected
        elif res.status == 1:  # Infeasible (should not happen, [256] is valid)
            pass
    except Exception:
        pass

    # Fallback: Greedy + Local Search
    # Sort by weight descending (i.e., value ascending)
    # We want large 1/x, so small x.
    order = sorted(range(n), key=lambda i: -weights[i])

    selected_indices = set()
    for idx in order:
        # Check compatibility with all selected
        compatible = True
        for s_idx in selected_indices:
            if _lcm(nums[idx], nums[s_idx]) <= 256:
                compatible = False
                break
        if compatible:
            selected_indices.add(idx)

    # Local Search: Try to add any remaining node
    improved = True
    while improved:
        improved = False
        for idx in range(n):
            if idx in selected_indices:
                continue
            compatible = True
            for s_idx in selected_indices:
                if _lcm(nums[idx], nums[s_idx]) <= 256:
                    compatible = False
                    break
            if compatible:
                selected_indices.add(idx)
                improved = True
                if time.time() - start_time > deadline:
                    break
        if time.time() - start_time > deadline:
            break

    if not selected_indices:
        return [256]  # Any single node is valid

    return [nums[i] for i in selected_indices]

# EVOLVE-BLOCK-END
