"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

# EVOLVE-BLOCK-START
_N = 256
_DEADLINE_FRAC = 0.85  # use 85% of 90s = 76.5s

def _build_constraints():
    """Build the constraint matrix for 3-AP-free condition.
    
    For each (a, b) with a < b, the third term c = 2b - a must not be in the set
    if both a and b are. So x_a + x_b + x_c <= 2 for all valid triples.
    We also need to consider c = 2a - b (when b < a < c), but since we iterate
    a < b, c = 2b - a > b, so we cover all triples where the middle is b.
    Actually, a 3-term AP (x, y, z) with x < y < z has y = (x+z)/2, so z = 2y - x.
    So for each pair (x, z) with x < z and x+z even, y = (x+z)/2, we need x_x + x_y + x_z <= 2.
    Equivalently, for each (a, b) with a < b, c = 2b - a; if c <= N, then x_a + x_b + x_c <= 2.
    """
    constraints = []
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            c = 2 * b - a
            if c > _N:
                break
            # constraint: x_a + x_b + x_c <= 2
            row = np.zeros(_N)
            row[a - 1] = 1
            row[b - 1] = 1
            row[c - 1] = 1
            constraints.append(row)
    return np.array(constraints)

def _milp_solve(cst_matrix, time_limit):
    """Solve the MILP. Returns the solution array or None."""
    # Maximize sum of 1/x_i * x_i
    c = -1.0 / np.arange(1, _N + 1, dtype=float)  # negative for minimization
    
    # 0 <= x_i <= 1, integer
    integrality = np.ones(_N)
    lb = np.zeros(_N)
    ub = np.ones(_N)
    bounds = Bounds(lb, ub)
    
    # x_a + x_b + x_c <= 2 for all AP triples
    constraint = LinearConstraint(cst_matrix, ub=2.0)
    
    result = milp(
        c=c,
        constraints=constraint,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit}
    )
    
    if result.x is None:
        return None
    return result.x

def _validate(selected):
    """Check that selected has no 3-term AP."""
    s = set(selected)
    for a in selected:
        for b in selected:
            if a < b:
                c = 2 * b - a
                if c <= _N and c in s:
                    return False
    return True

def _local_search_improve(selected, time_limit):
    """Try to improve the solution by adding/removing/swapping elements."""
    selected = set(selected)
    best_score = sum(1.0 / x for x in selected)
    
    while True:
        t0 = time.time()
        if t0 > time_limit:
            break
        
        improved = False
        # Try adding elements
        for x in range(1, _N + 1):
            if x in selected:
                continue
            # Check if adding x creates a 3-AP
            valid = True
            for a in selected:
                # x could be the first term: x, y, z where y = (x+z)/2
                # x could be the middle: a, x, c where c = 2x - a
                # x could be the last: a, b, x where b = (a+x)/2
                # Check: for each pair (a, b) in selected, does adding x create AP?
                # Actually, check if x with any two existing elements forms AP
                for b in selected:
                    if a == b:
                        continue
                    # Check if {a, b, x} forms a 3-AP
                    triple = sorted([a, b, x])
                    if triple[0] + triple[2] == 2 * triple[1]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                new_score = best_score + 1.0 / x
                if new_score > best_score + 1e-15:
                    selected.add(x)
                    best_score = new_score
                    improved = True
        
        if not improved:
            break
    
    return sorted(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 90 * _DEADLINE_FRAC
    
    # Build constraint matrix
    t0 = time.time()
    cst_matrix = _build_constraints()
    build_time = time.time() - t0
    
    # Solve MILP with time limit
    remaining = deadline - time.time()
    if remaining < 10:
        # Not enough time for MILP, use greedy
        return _greedy_construct()
    
    # Leave time for local search
    milp_time = max(5, remaining - 15)
    
    selected = None
    try:
        x_sol = _milp_solve(cst_matrix, milp_time)
        if x_sol is not None:
            selected = [i + 1 for i in range(_N) if x_sol[i] > 0.5]
    except Exception:
        pass
    
    if not selected or len(selected) == 0:
        selected = _greedy_construct()
    
    # Validate
    if not _validate(selected):
        # Fallback to greedy
        selected = _greedy_construct()
    
    # Local search improvement
    remaining = deadline - time.time()
    if remaining > 2:
        selected = _local_search_improve(selected, time.time() + remaining - 1)
    
    # Final validation
    if not _validate(selected):
        selected = _greedy_construct()
    
    return selected

def _greedy_construct():
    """Greedy: add elements in order of decreasing 1/x (i.e., increasing x) if valid."""
    selected = []
    for x in range(1, _N + 1):
        if all(2 * b - a != x for a in selected for b in selected if a < b):
            selected.append(x)
    return selected
# EVOLVE-BLOCK-END
