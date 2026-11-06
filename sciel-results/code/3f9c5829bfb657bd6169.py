import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    N = 180
    universe = list(range(1, N + 1))
    
    # Precompute all bad triples {a, b, c} such that 1/a = 1/b + 1/c
    # This implies bc = a(b+c), or (b-a)(c-a) = a^2
    bad_triples = set()
    
    for a in range(1, N + 1):
        a2 = a * a
        for x in range(1, int(np.sqrt(a2)) + 1):
            if a2 % x == 0:
                y = a2 // x
                if x == y:
                    continue  # b == c, not distinct
                b = a + x
                c = a + y
                if b <= N and c <= N:
                    bad_triples.add((a, b, c))
    
    # Build conflict_pairs: for each element x, the set of pairs (u,v) that conflict with x
    # Using the efficient formula: for pair (u,v) with u<v:
    #   Role 1: x = uv/(u+v) if integer
    #   Role 2: x = uv/(v-u) if integer
    conflict_pairs = {x: set() for x in range(1, N + 1)}
    
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            num1 = u * v
            den1 = u + v
            if num1 % den1 == 0:
                x1 = num1 // den1
                if 1 <= x1 <= N and x1 != u and x1 != v:
                    conflict_pairs[x1].add(frozenset((u, v)))
            
            num2 = u * v
            den2 = v - u
            if den2 > 0 and num2 % den2 == 0:
                x2 = num2 // den2
                if 1 <= x2 <= N and x2 != u and x2 != v:
                    conflict_pairs[x2].add(frozenset((u, v)))
    
    def is_conflict_with_addition(S, x):
        """Check if adding x to S creates a conflict."""
        for pair in conflict_pairs[x]:
            u, v = tuple(pair)
            if u in S and v in S:
                return True
        return False
    
    conflict_count = {x: len(conflict_pairs[x]) for x in range(1, N + 1)}
    
    best_set = set()
    best_size = 0
    
    def greedy_construct(order):
        S = set()
        for x in order:
            if not is_conflict_with_addition(S, x):
                S.add(x)
        return S
    
    def local_search(S, max_steps=200):
        """Improve S by trying to add excluded elements."""
        S = set(S)
        excluded = set(universe) - S
        improved = True
        steps = 0
        while improved and steps < max_steps and time.time() < deadline - 5:
            improved = False
            steps += 1
            # Try to add each excluded element
            for x in list(excluded):
                if not is_conflict_with_addition(S, x):
                    S.add(x)
                    excluded.discard(x)
                    improved = True
                    break
            # If no additions possible, try swaps: remove one from S, try to add two from excluded
            if not improved:
                for x in list(S):
                    S_remove = S - {x}
                    added = 0
                    for y in list(excluded):
                        if not is_conflict_with_addition(S_remove, y):
                            S_remove.add(y)
                            added += 1
                    if added > 1:
                        S = S_remove
                        excluded = set(universe) - S
                        improved = True
                        break
        return S
    
    # Try diverse initialization strategies
    orders = [
        list(range(N, 0, -1)),  # Descending
        list(range(1, N + 1)),  # Ascending
        sorted(range(1, N + 1), key=lambda x: conflict_count[x]),  # Fewest conflicts first
        sorted(range(1, N + 1), key=lambda x: conflict_count[x], reverse=True),  # Most conflicts first
    ]
    
    for order in orders:
        S = greedy_construct(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Random restarts with local search
    start_time = time.time()
    while time.time() < deadline - 10:
        order = random.sample(range(1, N + 1), N)
        S = greedy_construct(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Try MILP if time permits
    if time.time() < deadline - 15:
        try:
            n_vars = N
            c = np.ones(n_vars)
            n_constraints = len(bad_triples)
            
            if n_constraints > 0:
                A_list = []
                for (a, b, c_val) in bad_triples:
                    row = [0] * N
                    row[a-1] = 1
                    row[b-1] = 1
                    row[c_val-1] = 1
                    A_list.append(row)
                
                A = np.array(A_list)
                b_ub = np.full(n_constraints, 2)
                
                constraints = LinearConstraint(A, ub=b_ub)
                integrality = np.ones(n_vars)
                bounds = Bounds(0, 1)
                
                res = milp(c=-c, constraints=constraints, integrality=integrality, 
                          bounds=bounds, options={"time_limit": 10})
                
                if res.status == 0 and res.x is not None:
                    x_int = np.round(res.x).astype(int)
                    if np.all(A @ x_int <= 2 + 1e-5):
                        candidate_set = set(i+1 for i in range(N) if x_int[i] == 1)
                        if len(candidate_set) > best_size:
                            best_size = len(candidate_set)
                            best_set = candidate_set
        except Exception:
            pass
    
    return list(best_set)
