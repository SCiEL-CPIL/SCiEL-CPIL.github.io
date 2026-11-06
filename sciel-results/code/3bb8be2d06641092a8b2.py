import time
from itertools import combinations
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute LCM table for speed
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]
            
    # Find all bad triples
    bad_triples = []
    for a, b, c in combinations(nums, 3):
        lab = lcm_table[a][b]
        if lab != lcm_table[a][c] or lab != lcm_table[b][c]:
            continue
        bad_triples.append((a-1, b-1, c-1)) # 0-indexed for matrix
        
    if time.time() > deadline:
        return [1]
        
    M = len(bad_triples)
    
    # If no bad triples, return all
    if M == 0:
        return nums
        
    # Construct MILP
    # Variables x_0 ... x_127
    # Maximize sum(x)
    # Subject to: for each triple (i,j,k), x_i + x_j + x_k <= 2
    #             0 <= x_i <= 1, x_i integer
    
    c = np.ones(N)
    
    # Build sparse matrix A
    rows = []
    cols = []
    data = []
    for idx, (i, j, k) in enumerate(bad_triples):
        rows.extend([idx, idx, idx])
        cols.extend([i, j, k])
        data.extend([1, 1, 1])
        
    A = csr_matrix((data, (rows, cols)), shape=(M, N))
    
    # Constraints: A x <= 2
    # LinearConstraint(A, lb=-np.inf, ub=2)
    # Actually, milp takes constraints as (A, lb, ub)
    # We want A x <= 2, so lb=-inf, ub=2
    
    # Bounds: 0 <= x_i <= 1
    lb_bounds = np.zeros(N)
    ub_bounds = np.ones(N)
    
    integrality = np.ones(N) # All integer
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb=-np.inf, ub=2),
            integrality=integrality,
            bounds=Bounds(lb_bounds, ub_bounds)
        )
        
        if res.status == 0: # Optimal
            # res.x contains the solution
            # Filter out numerical noise
            solution = [i+1 for i in range(N) if res.x[i] > 0.5]
            if solution:
                return solution
        else:
            # Fallback if MILP fails or is infeasible (should not be, [1] is feasible)
            pass
            
    except Exception:
        pass
        
    # Fallback: Greedy + Local Search
    # This is the logic from the previous good solution, slightly optimized
    
    # Precompute conflicts for greedy
    # conflict[i] is a set of pairs (j, k) such that (i, j, k) is a bad triple?
    # No, for greedy, we check if adding v creates a bad triple with existing pairs.
    # pair_conflicts[(a,b)] = set of c such that (a,b,c) is bad.
    
    pair_conflicts = {}
    for i, j, k in bad_triples:
        # i, j, k are 0-indexed
        pair_conflicts.setdefault((i, j), set()).add(k)
        pair_conflicts.setdefault((i, k), set()).add(j)
        pair_conflicts.setdefault((j, k), set()).add(i)
        
    def can_add(v, selected_list):
        """Check if v (0-indexed) can be added to selected_list (list of 0-indexed)."""
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                a, b = selected_list[i], selected_list[j]
                key = (a, b) if a < b else (b, a)
                if v in pair_conflicts.get(key, ()):
                    return False
        return True
        
    def greedy(order):
        selected = []
        for x in order:
            if can_add(x, selected):
                selected.append(x)
        return selected
        
    candidates = []
    
    # Strategy 1: Natural order
    candidates.append(greedy(list(range(N))))
    
    # Strategy 2: Reverse order
    candidates.append(greedy(list(range(N-1, -1, -1))))
    
    # Strategy 3: Random restarts
    import random
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        order = list(range(N))
        random.shuffle(order)
        candidates.append(greedy(order))
        
    best = max(candidates, key=len)
    
    # Local search: Add
    def local_search_add(sel, deadline):
        sel_list = list(sel)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(N):
                if v in sel_list:
                    continue
                if can_add(v, sel_list):
                    sel_list.append(v)
                    improved = True
        return sel_list
        
    if time.time() < deadline - 2:
        best = local_search_add(best, deadline)
        
    # Convert back to 1-indexed
    return [x + 1 for x in best]
