"""
Strategy: Formulate the problem as a Mixed Integer Linear Programming (MILP) problem.
Variables: x_i in {0, 1} for i in 1..180.
Objective: Maximize sum(x_i).
Constraints: For each forbidden triple (a, b, c), x_a + x_b + x_c <= 2.
Solver: scipy.optimize.milp (HiGHS backend).
"""
import time
import sys
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    N = 180
    
    # 1. Precompute all forbidden triples (a, b, c) with 1/a = 1/b + 1/c, a,b,c distinct in [1, N]
    # Condition: (b-a)(c-a) = a^2
    triples = []
    seen_triples = set()
    
    for a in range(1, N + 1):
        a_sq = a * a
        # d = b - a, e = c - a => d * e = a^2
        # b = a + d, c = a + e
        # We need b, c <= N and b != c.
        # Iterate divisors d of a^2
        for d in range(1, a + 1):
            if a_sq % d == 0:
                e = a_sq // d
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    # Normalize the triple to avoid duplicates (order of b,c doesn't matter for the set, 
                    # but for the constraint x_a+x_b+x_c <= 2, the variables are just a sum.
                    # However, we just need the set of indices {a, b, c}.
                    # Since a < b and a < c is implied by the equation 1/a = 1/b + 1/c (a is smallest),
                    # the triple is uniquely identified by (a, b, c) with b < c or just the set.
                    # Let's store sorted tuple to be safe, though a is always smallest.
                    t = tuple(sorted((a, b, c)))
                    if t not in seen_triples:
                        seen_triples.add(t)
                        triples.append(t)
    
    # 2. Setup MILP
    # Variables: x_1, ..., x_180 (index 0 to 179)
    n_vars = N
    c_obj = -1.0 * [1] * n_vars  # Minimize negative sum = Maximize sum
    
    # Constraints: For each triple (a, b, c), x_a + x_b + x_c <= 2
    # Indices are 0-based in scipy, so subtract 1 from values
    A_ub = []
    b_ub = []
    
    for a, b, c in triples:
        row = [0] * n_vars
        row[a - 1] = 1
        row[b - 1] = 1
        row[c - 1] = 1
        A_ub.append(row)
        b_ub.append(2)
    
    # If no triples, return all
    if not A_ub:
        return list(range(1, N + 1))
        
    # Convert to array format for efficiency
    import numpy as np
    A = np.array(A_ub, dtype=float)
    b = np.array(b_ub, dtype=float)
    
    # Constraints: A @ x <= b
    # Also x >= 0 and x <= 1 (bounds handle integrality with integer variables)
    cons = LinearConstraint(A, ub=b)
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integers
    integrality = np.ones(n_vars)
    
    # Solve
    try:
        res = milp(
            c=c_obj,
            constraints=cons,
            integrality=integrality,
            bounds=bounds
        )
        
        # Check if solved within time limit
        if res.status == 0:  # Optimal
            x = res.x
            # Construct set
            # x is numpy array, indices 0..179
            # Filter for x_i > 0.5
            S = [i + 1 for i in range(n_vars) if x[i] > 0.5]
            return S
        else:
            # If not optimal, try to get best feasible solution found so far?
            # milp might not return a feasible solution if it didn't find one or timed out.
            # Fallback to a simple greedy if MILP fails or times out.
            pass
    except Exception:
        pass

    # 3. Fallback: Greedy construction if MILP fails or times out
    # This ensures we always return a valid non-empty set.
    
    # Build forbidden pairs for fast checking
    forbidden_pairs = [set() for _ in range(N + 1)]
    for (a, b, c) in triples:
        forbidden_pairs[a].add((b, c))
        forbidden_pairs[b].add((a, c))
        forbidden_pairs[c].add((a, b))
    
    def greedy_construct(order):
        S = set()
        for x in order:
            valid = True
            for (y, z) in forbidden_pairs[x]:
                if y in S and z in S:
                    valid = False
                    break
            if valid:
                S.add(x)
        return S
        
    import random
    random.seed(42)
    best_S = set()
    best_size = 0
    
    # Run a few quick greedy passes
    remaining = deadline - time.time() - 5.0
    if remaining > 1.0:
        for i in range(10):
            order = list(range(1, N + 1))
            random.shuffle(order)
            S = greedy_construct(order)
            if len(S) > best_size:
                best_S = S
                best_size = len(S)
                
    if best_S:
        return sorted(best_S)
    
    # Absolute fallback: just return {1} which is always valid?
    # 1/1 = 1/1 + 1/1 -> 1 = 2 (False). But distinct elements required.
    # If S={1}, no 3 distinct elements. Valid.
    return [1]

# EVOLVE-BLOCK-END
