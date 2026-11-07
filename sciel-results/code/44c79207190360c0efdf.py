"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
from functools import lru_cache

_N = 256
_DEADLINE = 85.0  # safety margin before 90s kill

def _build_conflict_pairs():
    """Build list of (i, j) pairs where gcd(i,j) > 1, for i < j."""
    pairs = []
    for i in range(1, _N + 1):
        for j in range(i + 1, _N + 1):
            if gcd(i, j) > 1:
                pairs.append((i, j))
    return pairs

def _solve_milp(conflict_pairs, deadline):
    """Solve the MWIS via scipy MILP."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        import scipy.sparse as sp
    except ImportError:
        return None

    n = _N
    # Objective: maximize sum(i * x_i) => minimize -sum(i * x_i)
    c = -np.arange(1, n + 1, dtype=float)

    # Constraints: for each conflict pair (i, j), x_i + x_j <= 1
    # (indices are 0-based in the array, but our numbers are 1-based)
    if not conflict_pairs:
        # No conflicts: take all
        return list(range(1, n + 1))

    # Build sparse constraint matrix
    rows = []
    cols = []
    vals = []
    for (i, j) in conflict_pairs:
        # i, j are 1-based; convert to 0-based for scipy
        rows.append(len(rows)); cols.append(i - 1); vals.append(1.0)
        rows.append(len(rows)); cols.append(j - 1); vals.append(1.0)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(conflict_pairs), n))
    lb = np.zeros(len(conflict_pairs))
    ub = np.ones(len(conflict_pairs))
    constraints = LinearConstraint(A, lb, ub)

    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality = np.ones(n)  # all binary

    start = time.time()
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

    if time.time() > deadline:
        return None

    if res.status != 0:  # 0 = optimal
        return None

    # Extract solution
    selected = []
    for idx in range(n):
        if res.x[idx] is not None and abs(res.x[idx] - 1.0) < 1e-6:
            selected.append(idx + 1)
    return selected

def _greedy_fallback(deadline):
    """Greedy: pick numbers in descending order, adding if coprime to all selected."""
    selected = []
    for num in range(_N, 0, -1):
        if time.time() > deadline:
            break
        ok = True
        for s in selected:
            if gcd(num, s) > 1:
                ok = False
                break
        if ok:
            selected.append(num)
    return selected

def _local_search_improve(selected, deadline):
    """Try to improve by swapping or adding."""
    selected_set = set(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each unselected number
        for num in range(1, _N + 1):
            if num in selected_set:
                continue
            if all(gcd(num, s) == 1 for s in selected_set):
                selected_set.add(num)
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one selected, add one unselected if net gain
        for remove in list(selected_set):
            if time.time() > deadline:
                break
            for add in range(1, _N + 1):
                if add in selected_set:
                    continue
                if add <= remove:
                    continue
                # Check if add is coprime to all selected except remove
                if all(gcd(add, s) == 1 for s in selected_set if s != remove):
                    selected_set.discard(remove)
                    selected_set.add(add)
                    improved = True
                    break
            if improved:
                break
    return list(selected_set)

def construct():
    start = time.time()
    deadline = start + _DEADLINE

    # Build conflict pairs (this is fast: 256*255/2 ~ 32640 gcd checks)
    conflict_pairs = _build_conflict_pairs()

    # Try MILP first
    result = None
    if time.time() < deadline:
        result = _solve_milp(conflict_pairs, deadline)

    # Fallback to greedy + local search if MILP fails or times out
    if result is None or len(result) == 0:
        result = _greedy_fallback(deadline)

    # Polish with local search
    if time.time() < deadline - 1.0:
        polished = _local_search_improve(result, deadline)
        if sum(polished) >= sum(result):
            result = polished

    # Ensure non-empty
    if not result:
        result = [1]

    # Ensure Python ints, list type
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
