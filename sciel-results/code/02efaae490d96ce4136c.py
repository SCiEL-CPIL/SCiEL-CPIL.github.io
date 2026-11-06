import time
import random
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
    
    def is_valid_set(S):
        """Check if S has no bad triple."""
        for (a, b, c) in bad_triples:
            if a in S and b in S and c in S:
                return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to S without creating a bad triple."""
        for (u, v) in conflict_pairs_for[x]:
            if u in S and v in S:
                return False
        return True
    
    def greedy_solve(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    
    def fast_local_search(S, max_iters=50000):
        """Fast local search: try to add elements, and do remove-one-add-one or remove-one-add-two moves."""
        S = set(S)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 5:
                break
            
            # Phase 1: Try to add any missing element
            for x in range(1, N+1):
                if x not in S and can_add(S, x):
                    S.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try remove-one-add-one moves
            for remove_elem in list(S):
                S_without = S - {remove_elem}
                # Find all elements that can now be added
                addable = []
                for x in range(1, N+1):
                    if x not in S_without and x != remove_elem:
                        if can_add(S_without, x):
                            addable.append(x)
                
                if len(addable) >= 1:
                    # Try adding the best one (or all if beneficial)
                    S_new = S_without | {addable[0]}
                    if len(S_new) > len(S):
                        S = S_new
                        improved = True
                        break
                
                if len(addable) >= 2:
                    # Try adding two elements
                    for i in range(min(len(addable), 10)):
                        for j in range(i+1, min(len(addable), 10)):
                            S_trial = S_without | {addable[i], addable[j]}
                            if is_valid_set(S_trial) and len(S_trial) > len(S):
                                S = S_trial
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        return S
    
    def hill_climbing(S, max_iters=100000):
        """More aggressive hill climbing with random restarts on local optima."""
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation: remove 1-3 random elements
            k = random.choice([1, 1, 2, 2, 3])
            if len(S) > k:
                to_remove = random.sample(list(S), k)
                S = S - set(to_remove)
            
            # Greedily add back
            changed = True
            while changed:
                changed = False
                for x in range(1, N+1):
                    if x not in S and can_add(S, x):
                        S.add(x)
                        changed = True
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
        
        return list(best_S)
    
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
                   options={"time_limit": 20, "mip_rel_gap": 0.01})
        
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
    while time.time() - start_time < 8:
        order = random.sample(range(1, N+1), N)
        S = greedy_solve(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Fast local search on best solution
    if best_set and time.time() < deadline - 15:
        improved_set = fast_local_search(best_set, max_iters=50000)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Hill climbing with random perturbations
    if best_set and time.time() < deadline - 10:
        hc_result = hill_climbing(best_set, max_iters=50000)
        if len(hc_result) > best_size:
            best_size = len(hc_result)
            best_set = hc_result
    
    # Final polish: more local search
    while time.time() < deadline - 3:
        improved_set = fast_local_search(best_set, max_iters=10000)
        if len(improved_set) <= best_size:
            break
        best_size = len(improved_set)
        best_set = improved_set
    
    return best_set if best_set else [1]
