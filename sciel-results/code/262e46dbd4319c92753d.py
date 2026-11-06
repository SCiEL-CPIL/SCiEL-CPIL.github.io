import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    N = 180
    universe = list(range(1, N + 1))
    
    # 1. Precompute all "bad" triples {a, b, c} such that 1/a = 1/b + 1/c
    bad_triples = set()
    
    for a in range(1, N + 1):
        a2 = a * a
        for x in range(1, int(np.sqrt(a2)) + 1):
            if a2 % x == 0:
                y = a2 // x
                if x == y:
                    continue
                b = a + x
                c = a + y
                if b <= N and c <= N:
                    bad_triples.add((a, b, c))
    
    if not bad_triples:
        return universe

    # Build conflict structure: for each element, which pairs conflict
    conflict_pairs_for = {i: set() for i in range(1, N+1)}
    for (a, b, c) in bad_triples:
        conflict_pairs_for[a].add((b, c))
        conflict_pairs_for[b].add((a, c))
        conflict_pairs_for[c].add((a, b))
    
    # Precompute adjacency: for each element x, the set of elements that can't be in S with x
    # Actually, we need: if x is in S, then for any (u,v) in conflict_pairs_for[x], not both u and v can be in S.
    
    def is_valid_set(S):
        """Check if S has no bad triple."""
        for (a, b, c) in bad_triples:
            if a in S and b in S and c in S:
                return False
        return True
    
    def greedy_solve(order):
        S = set()
        for x in order:
            is_conflict = False
            for (u, v) in conflict_pairs_for[x]:
                if u in S and v in S:
                    is_conflict = True
                    break
            if not is_conflict:
                S.add(x)
        return S
    
    def local_search(S, max_iters=5000):
        """Try to improve S by swapping out one element for another."""
        S = set(S)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 5:
                break
            # Try removing one element and adding another
            for remove_elem in list(S):
                S_new = S - {remove_elem}
                # Try adding each element not in S_new
                for add_elem in range(1, N+1):
                    if add_elem in S_new:
                        continue
                    S_trial = S_new | {add_elem}
                    if is_valid_set(S_trial) and len(S_trial) > len(S):
                        S = S_trial
                        improved = True
                        break
                if improved:
                    break
        return S
    
    best_set = []
    best_size = 0
    
    # Try MILP first with better options
    n_vars = N
    c = np.ones(n_vars)
    
    n_constraints = len(bad_triples)
    A_list = []
    for (a, b, c) in bad_triples:
        row = [0] * N
        row[a-1] = 1
        row[b-1] = 1
        row[c-1] = 1
        A_list.append(row)
        
    A = np.array(A_list)
    b_ub = np.full(n_constraints, 2)
    
    constraints = LinearConstraint(A, ub=b_ub)
    integrality = np.ones(n_vars)
    bounds = Bounds(0, 1)
    
    try:
        res = milp(c=-c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 30, "mip_rel_gap": 0.01})
        
        if res.status in [0, 2] and hasattr(res, 'x') and res.x is not None:
            x_int = np.round(res.x).astype(int)
            if np.all(A @ x_int <= 2 + 1e-5):
                candidate_set = [i+1 for i in range(N) if x_int[i] == 1]
                if len(candidate_set) > 0:
                    best_set = candidate_set
                    best_size = len(candidate_set)
    except Exception as e:
        pass
    
    # Greedy with multiple orders
    orders = [
        list(range(1, N+1)),
        list(range(N, 0, -1)),
    ]
    
    # Add sorted by degree (fewest conflicts first)
    degrees = {i: len(conflict_pairs_for[i]) for i in range(1, N+1)}
    orders.append(sorted(range(1, N+1), key=lambda x: degrees[x]))
    orders.append(sorted(range(1, N+1), key=lambda x: -degrees[x]))
    
    for order in orders:
        S = greedy_solve(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Random restarts for greedy
    start_time = time.time()
    while time.time() - start_time < 10:
        order = random.sample(range(1, N+1), N)
        S = greedy_solve(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Local search on best solution
    if best_set and time.time() < deadline - 10:
        improved_set = local_search(best_set, max_iters=20000)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = list(improved_set)
    
    # More local search iterations if time permits
    while time.time() < deadline - 5:
        improved_set = local_search(best_set, max_iters=5000)
        if len(improved_set) <= best_size:
            break
        best_size = len(improved_set)
        best_set = list(improved_set)
    
    return best_set if best_set else [1]
