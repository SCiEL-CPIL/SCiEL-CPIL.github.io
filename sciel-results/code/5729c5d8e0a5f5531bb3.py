import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        if P:
            u_candidates = list(P) + list(X)
            if not u_candidates:
                return
            max_deg = -1
            pivot = None
            for u in u_candidates:
                deg = len(adj[u].intersection(P))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            for v in list(P):
                if v in adj[pivot]:
                    continue
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                P.remove(v)
                X.add(v)
        else:
             pass

    def find_maximal_cliques(n, adj):
        cliques = []
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        return cliques

    def solve_hitting_set_ilp(n, cliques):
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        if not relevant_cliques:
            return []
        
        m = len(relevant_cliques)
        # Build constraint matrix A
        # A[i][v] = 1 if v in relevant_cliques[i]
        A = np.zeros((m, n))
        for i, c in enumerate(relevant_cliques):
            for v in c:
                A[i][v] = 1
        
        # Objective: min sum(x)
        c_obj = np.ones(n)
        
        # Constraints: A @ x >= 1
        # LinearConstraint(A, lb=1, ub=np.inf)
        constraints = LinearConstraint(A, lb=1, ub=np.inf)
        
        # Bounds: x in [0, 1]
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: x is integer
        integrality = np.ones(n)
        
        try:
            res = milp(c=c_obj, constraints=constraints, bounds=bounds, integrality=integrality)
            if res.status == 0:
                # res.x is the optimal solution
                sol = [i for i in range(n) if res.x[i] > 0.5]
                return sorted(sol)
            else:
                # Fallback if ILP fails
                return solve_hitting_set_greedy(n, relevant_cliques)
        except Exception:
            return solve_hitting_set_greedy(n, relevant_cliques)

    def solve_hitting_set_greedy(n, cliques):
        remaining = list(cliques)
        T = set()
        while remaining:
            counts = defaultdict(int)
            for c in remaining:
                for v in c:
                    counts[v] += 1
            if not counts:
                break
            best_v = max(counts, key=counts.get)
            T.add(best_v)
            remaining = [c for c in remaining if best_v not in c]
        return sorted(T)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges, adj = generate_graph(n, p, seed)
        cliques = find_maximal_cliques(n, adj)
        
        transversal = solve_hitting_set_ilp(n, cliques)
        
        results.append(transversal)
        
    return results
