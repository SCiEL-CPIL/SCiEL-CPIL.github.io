import sys
from random import Random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        # Generate edges
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        
        # Build adjacency bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Find all maximal cliques
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                maximal_cliques.append(R.copy())
                return
            
            if not P:
                return

            # Choose pivot
            pivot = -1
            max_deg = -1
            p_temp = P | X
            while p_temp:
                lsb = p_temp & -p_temp
                idx = lsb.bit_length() - 1
                deg = (adj[idx] & (P | X)).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = idx
                p_temp ^= lsb
                
            if pivot == -1:
                return
            
            candidates = P & ~(adj[pivot])
            
            c_temp = candidates
            while c_temp:
                lsb = c_temp & -c_temp
                v = lsb.bit_length() - 1
                R.append(v)
                expand(R, P & adj[v], X & adj[v])
                R.pop()
                P ^= lsb
                X |= lsb
                c_temp ^= lsb
        
        all_vertices = (1 << n) - 1
        expand([], all_vertices, 0)
        
        # Filter cliques to size >= 2
        cliques = [c for c in maximal_cliques if len(c) >= 2]
        
        if not cliques:
            return [[u, v] for u, v in edges]

        edge_list = list(edges)
        edge_index = {e: i for i, e in enumerate(edge_list)}
        num_edges = len(edge_list)
        num_cliques = len(cliques)
        
        # If too many cliques, use heuristic
        if num_cliques > 2000:
            # Heuristic: Greedy Set Partition
            # Sort by size desc
            cliques_sorted = sorted(cliques, key=len, reverse=True)
            remaining_edges = set(edge_list)
            partition = []
            
            for clique in cliques_sorted:
                clique_edges = set()
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        e = (clique[i], clique[j])
                        if e in remaining_edges:
                            clique_edges.add(e)
                
                if clique_edges and clique_edges.issubset(remaining_edges):
                    partition.append(clique)
                    remaining_edges -= clique_edges
            
            for e in remaining_edges:
                partition.append(list(e))
            return partition

        # Build A_eq
        A_eq = np.zeros((num_edges, num_cliques), dtype=np.float64)
        for j, clique in enumerate(cliques):
            for i in range(len(clique)):
                for k in range(i + 1, len(clique)):
                    e = (clique[i], clique[k])
                    if e in edge_index:
                        A_eq[edge_index[e], j] = 1.0

        c = np.ones(num_cliques)
        constraints = [LinearConstraint(A_eq, 1.0, 1.0)]
        integrality = np.ones(num_cliques)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                solution_cliques = []
                for j in range(num_cliques):
                    if res.x[j] > 0.5:
                        solution_cliques.append(cliques[j])
                return solution_cliques
        except Exception:
            pass
        
        # Fallback if ILP fails
        cliques_sorted = sorted(cliques, key=len, reverse=True)
        remaining_edges = set(edge_list)
        partition = []
        
        for clique in cliques_sorted:
            clique_edges = set()
            for i in range(len(clique)):
                for k in range(i + 1, len(clique)):
                    e = (clique[i], clique[k])
                    if e in remaining_edges:
                        clique_edges.add(e)
                
            if clique_edges and clique_edges.issubset(remaining_edges):
                partition.append(clique)
                remaining_edges -= clique_edges
        
        for e in remaining_edges:
            partition.append(list(e))
        return partition

    res1 = instance(32, 0.6, 601)
    res2 = instance(64, 0.55, 602)
    
    return [res1, res2]
