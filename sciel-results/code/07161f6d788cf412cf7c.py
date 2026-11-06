import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

def construct():
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(graph, n):
        """
        Finds all maximal cliques in a graph using bitmasks for speed.
        """
        adj = [0] * n
        for u, v in graph:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        cliques = []
        
        def bk(R, P, X):
            if P == 0 and X == 0:
                if R != 0:
                    # Convert bitmask to list
                    verts = []
                    for i in range(n):
                        if R & (1 << i):
                            verts.append(i)
                    cliques.append(verts)
                return
            
            # Pivot selection
            # We want a pivot v in P | X such that P & adj[v] is minimal (to minimize iterations)
            # Standard BK: iterate over P \ N(pivot).
            # If we pick pivot with max degree in P, P \ N(pivot) is small.
            
            union_px = P | X
            if union_px == 0:
                return
            
            # Find pivot
            pivot = -1
            max_deg = -1
            # Iterate through bits in union_px
            temp = union_px
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                v = int(np.log2(lsb)) # Slow. Use bit_length
                v = (lsb.bit_length() - 1)
                deg = bin(P & adj[v]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
                temp -= lsb
            
            if pivot == -1:
                return
                
            # Iterate over P \ N(pivot)
            candidates = P & ~adj[pivot]
            
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                
                R_new = R | lsb
                P_new = P & adj[v]
                X_new = X & adj[v]
                
                bk(R_new, P_new, X_new)
                
                P &= ~lsb
                X |= lsb
                temp -= lsb
                
        bk(0, (1 << n) - 1, 0)
        return cliques

    def solve_transversal(n, edges, cliques):
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []
        
        try:
            num_vars = n
            c = np.ones(num_vars)
            
            A = np.zeros((len(relevant_cliques), num_vars))
            for i, clique in enumerate(relevant_cliques):
                for v in clique:
                    A[i, v] = 1
            
            constraint = LinearConstraint(A, 1, np.inf)
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(num_vars)
            
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                selected = [i for i in range(n) if res.x[i] > 0.5]
                return selected
        except Exception:
            pass
            
        # Fallback: Greedy Set Cover (kept for safety, though MILP should work)
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(relevant_cliques):
            for v in clique:
                vertex_to_cliques[v].append(i)
                
        covered = [False] * len(relevant_cliques)
        selected = []
        remaining_cliques = set(range(len(relevant_cliques)))
        
        while remaining_cliques:
            best_v = None
            best_count = -1
            for v in range(n):
                count = sum(1 for ci in vertex_to_cliques[v] if ci in remaining_cliques)
                if count > best_count:
                    best_count = count
                    best_v = v
                    
            if best_count == 0:
                break
                
            selected.append(best_v)
            for ci in vertex_to_cliques[best_v]:
                if ci in remaining_cliques:
                    remaining_cliques.remove(ci)
                    
        return selected

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        cliques = bron_kerbosch(edges, n)
        transversal = solve_transversal(n, edges, cliques)
        out.append(sorted(transversal))
        
    return out
