"""Linear triple system on 16 points minimizing independence number."""
import time
import random
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
from numpy import ones, zeros, array

DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150s)


def _independence_number_fast(triples, n=16):
    """Largest subset of points containing no complete triple.
    Uses branch-and-bound with bitsets for speed."""
    if not triples:
        return n
    
    # Build adjacency: for each pair of points, check if they're in a triple together
    # A set S is independent iff for every triple {a,b,c}, S does not contain all of a,b,c
    # Equivalently: the hypergraph with edges = triples has independence number = max S with no edge fully contained
    
    # Precompute: for each point, which triples contain it
    point_to_triples = [[] for _ in range(n)]
    triples_bits = []
    for ti, t in enumerate(triples):
        b = 0
        for x in t:
            b |= (1 << x)
            point_to_triples[x].append(ti)
        triples_bits.append(b)
    
    # Branch and bound: try to build largest independent set
    # Start with all points, remove conflicts
    # Use recursive search with pruning
    
    best_found = [0]
    
    def can_extend(mask, size):
        """Check if we can extend current partial independent set (mask) to size >= best_found[0]"""
        if size >= best_found[0]:
            return True
        # Upper bound: size + number of remaining points that could be added
        # Quick bound: if size + (n - mask.bit_count()) <= best_found[0], prune
        return True  # simplified; real pruning below
    
    # Simpler approach: enumerate subsets by size from high to low, but with early termination
    # For n=16, worst case C(16,8)=12870 which is manageable if we prune well
    
    # Better: use the fact that if we have k triples, each triple "kills" 8 subsets of size 3
    # Let's do a smarter search: try all subsets of size s from n down to 1
    
    # Precompute for fast checking: a subset is NOT independent iff it contains some triple
    # For a given mask, check if any triple is fully contained
    
    def is_independent(mask):
        for b in triples_bits:
            if (mask & b) == b:
                return False
        return True
    
    # Search from largest size down
    for size in range(n, 0, -1):
        # Use combinations but with early exit
        for combo in combinations(range(n), size):
            mask = 0
            for x in combo:
                mask |= (1 << x)
            if is_independent(mask):
                return size
    return 0


def _independence_number(triples, n=16):
    """Wrapper that calls the fast version."""
    return _independence_number_fast(triples, n)


def _is_linear(triples):
    """Any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True


def _solve_milp(n=16, time_limit=30.0):
    """Use MILP to find a linear triple system minimizing independence number.
    
    Variables: x_t for each possible triple t (binary: selected or not)
    Constraints: for each pair (a,b), at most one selected triple contains both a and b
    Objective: maximize number of selected triples (proxy for covering pairs well)
    
    Then we evaluate the actual independence number of the result.
    """
    start = time.time()
    
    # All possible triples on n points
    all_triples = list(combinations(range(n), 3))
    m = len(all_triples)  # 560 triples
    
    # Pair -> list of triple indices containing that pair
    pair_to_triples = {}
    for ti, t in enumerate(all_triples):
        for a, b in combinations(t, 2):
            pair = (min(a, b), max(a, b))
            if pair not in pair_to_triples:
                pair_to_triples[pair] = []
            pair_to_triples[pair].append(ti)
    
    pairs = list(pair_to_triples.keys())
    p = len(pairs)  # 120 pairs
    
    # MILP: maximize sum(x_t) subject to for each pair, sum of x_t over triples containing that pair <= 1
    # This ensures linearity (no two triples share a pair)
    
    c = -1.0 * ones(m)  # minimize -sum = maximize sum
    
    # Constraints: for each pair, sum of x_t for t containing that pair <= 1
    A_rows = []
    for pi, pair in enumerate(pairs):
        row = zeros(m)
        for ti in pair_to_triples[pair]:
            row[ti] = 1.0
        A_rows.append(row)
    
    A = array(A_rows)
    lb = zeros(p)
    ub = ones(p)
    
    constraints = LinearConstraint(A, lb, ub)
    bounds = Bounds(0, 1)
    
    # Solve with time limit
    try:
        res = milp(c, constraints=constraints, integrality=ones(m), bounds=bounds,
                   options={"time_limit": time_limit, "presolve": True})
    except Exception:
        return None
    
    if res.status != 0:
        return None
    
    selected = []
    for ti in range(m):
        if res.x[ti] > 0.5:
            selected.append(list(all_triples[ti]))
    
    return selected


def _local_search(triples, n=16, time_limit=30.0):
    """Improve a linear triple system via local search."""
    start = time.time()
    
    def time_ok():
        return (time.time() - start) < time_limit
    
    current = [list(t) for t in triples]
    cur_alpha = _independence_number(current, n)
    
    # Build pair map for fast linearity check
    def build_pair_map(triples):
        pm = {}
        for t in triples:
            for a, b in combinations(sorted(t), 2):
                pm[(a, b)] = True
        return pm
    
    def can_add(pm, t):
        for a, b in combinations(sorted(t), 2):
            if (a, b) in pm:
                return False
        return True
    
    # All possible triples
    all_triples = list(combinations(range(n), 3))
    
    random.seed(42)
    
    for iteration in range(5000):
        if not time_ok():
            break
        
        pm = build_pair_map(current)
        
        # Try adding a random triple
        if random.random() < 0.7:
            t = list(random.choice(all_triples))
            if can_add(pm, t):
                new_alpha = _independence_number(current + [t], n)
                if new_alpha <= cur_alpha:
                    current.append(t)
                    cur_alpha = new_alpha
        else:
            # Try swapping: remove one triple, add another
            if len(current) > 0:
                ri = random.randint(0, len(current) - 1)
                removed = current.pop(ri)
                pm = build_pair_map(current)
                t = list(random.choice(all_triples))
                if can_add(pm, t):
                    new_alpha = _independence_number(current + [t], n)
                    if new_alpha <= cur_alpha:
                        current.append(t)
                        cur_alpha = new_alpha
                    else:
                        current.append(removed)
    
    return current, cur_alpha


def construct():
    start = time.time()
    n = 16
    
    def time_ok():
        return (time.time() - start) < DEADLINE
    
    # Phase 1: Solve MILP to get a good starting point
    milp_result = _solve_milp(n, time_limit=40.0)
    
    if milp_result and len(milp_result) > 0:
        # Verify linearity
        if _is_linear(milp_result):
            best = milp_result
            best_alpha = _independence_number(best, n)
        else:
            best = None
            best_alpha = n
    else:
        best = None
        best_alpha = n
    
    # If MILP didn't give good result, use a fallback construction
    if best is None or best_alpha > 10:
        # Fallback: construct a simple linear system
        # Use disjoint triples where possible, plus some overlapping
        fallback = []
        # 5 disjoint triples covering 15 points
        for i in range(5):
            fallback.append([3*i, 3*i+1, 3*i+2])
        # Add triples involving point 15
        # {15, 0, 3} - shares no pair with existing (0,1,2), (3,4,5), etc.
        # Actually (0,3) is not in any existing triple, so {15,0,3} is linear
        fallback.append([15, 0, 3])
        fallback.append([15, 1, 4])
        fallback.append([15, 2, 5])
        
        if _is_linear(fallback):
            if best is None:
                best = fallback
                best_alpha = _independence_number(best, n)
            else:
                fa = _independence_number(fallback, n)
                if fa < best_alpha:
                    best = fallback
                    best_alpha = fa
    
    # Phase 2: Local search to improve
    if time_ok() and best is not None:
        remaining_time = DEADLINE - (time.time() - start) - 5.0
        if remaining_time > 10:
            improved, imp_alpha = _local_search(best, n, time_limit=remaining_time)
            if imp_alpha < best_alpha:
                best = improved
                best_alpha = imp_alpha
    
    # Final verification
    if best is None:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    if not _is_linear(best):
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    return best


if __name__ == "__main__":
    res = construct()
    print("triples:", len(res), "alpha:", _independence_number(res))
