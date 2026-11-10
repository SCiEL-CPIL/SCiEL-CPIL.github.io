# EVOLVE-BLOCK-START
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            pivot_pool = P | X
            if not pivot_pool:
                return
            pivot = max(pivot_pool, key=lambda v: len(P & adj[v]))
            candidates = P - adj[pivot]
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        expand(set(), set(range(n)), set())
        return maximal_cliques

    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        if not cliques:
            return []
        
        # Filter cliques of size >= 2
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []

        universe = set()
        for clique in cliques:
            universe.update(clique)
        
        vertices = sorted(universe)
        v_index = {v: i for i, v in enumerate(vertices)}
        n_vars = len(vertices)
        n_cliques = len(cliques)

        # ILP Formulation
        # Minimize sum(x_v)
        c = np.ones(n_vars)
        
        # Constraints: For each clique, sum(x_v for v in clique) >= 1
        # A matrix of shape (n_cliques, n_vars)
        A = np.zeros((n_cliques, n_vars))
        b_lower = np.ones(n_cliques)
        
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i, v_index[v]] = 1.0
        
        # Bounds: x_v in {0, 1}
        integrality = np.ones(n_vars)
        bounds = Bounds(lb=0, ub=1)
        constraints = LinearConstraint(A, lb=b_lower, ub=np.inf)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0: # Optimal solution found
                selected_indices = np.where(res.x > 0.5)[0]
                selected_vertices = [vertices[i] for i in selected_indices]
                return sorted(selected_vertices)
            else:
                # Fallback if solver fails
                pass
        except Exception:
            pass
        
        # Fallback: Greedy Set Cover
        # This ensures we always return a valid solution even if MILP fails
        vertex_to_cliques = {v: set() for v in universe}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(i)
        
        uncovered = set(range(n_cliques))
        selected = set()
        
        while uncovered:
            best_v = None
            best_count = -1
            for v in universe:
                if v in selected:
                    continue
                count = len(vertex_to_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count == 0:
                break
            
            selected.add(best_v)
            uncovered -= vertex_to_cliques[best_v]
        
        return sorted(selected)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        solution = solve_instance(n, p, seed)
        out.append(solution)
        
    return out
# EVOLVE-BLOCK-END
