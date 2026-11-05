"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        max_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            pivot_candidates = list(P) + list(X)
            if not pivot_candidates:
                return
            
            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u
            
            P_set = set(P)
            X_set = set(X)
            
            if best_pivot is not None:
                candidates = [v for v in P_set if v not in adj[best_pivot]]
            else:
                candidates = list(P_set)
                
            for v in candidates:
                new_R = R | {v}
                new_P = P_set & adj[v]
                new_X = X_set & adj[v]
                
                bk(new_R, new_P, new_X)
                
                P_set.remove(v)
                X_set.add(v)
                
                if time.time() > deadline - 5:
                    return

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def solve_milp(n, cliques):
        if not cliques:
            return set()
            
        num_cliques = len(cliques)
        # x_v = 1 if vertex v is selected
        
        # Objective: minimize sum(x_v)
        c = np.ones(n)
        
        # Constraints: for each clique c_i, sum_{v in c_i} x_v >= 1
        # A_ub * x <= b_ub is standard, but scipy milp uses LinearConstraint(A, lb, ub)
        # We want A @ x >= 1
        # So lb = 1, ub = inf
        
        # Construct A matrix (num_cliques x n)
        A = np.zeros((num_cliques, n))
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i, v] = 1.0
        
        lb = np.ones(num_cliques)
        ub = np.full(num_cliques, np.inf)
        
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(n)  # All variables are binary/integer
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                # res.x contains the solution
                selected = set(int(v) for v in res.x if v > 0.5)
                return selected
        except Exception:
            pass
            
        # Fallback if MILP fails or times out (should not happen for n<=40)
        return set(range(n))

    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
            # Fallback: return all vertices (guaranteed valid)
            results.append(list(range(n)))
            continue
            
        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        
        # Filter cliques of size < 2
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
            
        sol = solve_milp(n, cliques)
        
        if sol is None or len(sol) > n:
            sol = set(range(n))
            
        results.append(sorted(sol))
        
    return results
# EVOLVE-BLOCK-END
