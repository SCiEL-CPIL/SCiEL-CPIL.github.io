import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = set()
        # Edges within K
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Edges between K and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find all maximal cliques
        # Bron-Kerbosch with pivot
        maximal_cliques = []
        
        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Pivot selection
            if P:
                # Choose u in P U X with max degree in P
                all_nodes = P.union(X)
                max_deg = -1
                pivot = None
                for u in all_nodes:
                    deg = len(P.intersection(adj[u]))
                    if deg > max_deg:
                        max_deg = deg
                        pivot = u
                if pivot is not None:
                    candidates = P - adj[pivot]
                else:
                    candidates = P.copy()
            else:
                candidates = P.copy()
                
            for u in list(candidates):
                bronk(R.union({u}), P.intersection(adj[u]), X.intersection(adj[u]))
                P.remove(u)
                X.add(u)
                
        bronk(set(), set(range(n)), set())
        
        # Candidate cliques: Maximal cliques + all edges
        candidates = set(maximal_cliques)
        for u, v in edges:
            candidates.add(frozenset((u, v)))
            
        # Convert to list for indexing
        cliques_list = list(candidates)
        M = len(cliques_list)
        
        if M == 0:
            return []
            
        # Build constraint matrix A_eq
        # Rows: edges, Cols: cliques
        # A_eq[i, j] = 1 if edge i is in clique j
        edge_list = list(edges)
        m = len(edge_list)
        
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        A_eq = np.zeros((m, M), dtype=int)
        for j, clique in enumerate(cliques_list):
            clique_set = set(clique)
            # Iterate over pairs in clique
            # Since cliques are small or sparse, this is fast
            clique_list = list(clique_set)
            for i in range(len(clique_list)):
                for l in range(i + 1, len(clique_list)):
                    u, v = clique_list[i], clique_list[l]
                    if u > v:
                        u, v = v, u
                    if (u, v) in edge_to_idx:
                        A_eq[edge_to_idx[(u, v)], j] = 1
                    elif (v, u) in edge_to_idx:
                        A_eq[edge_to_idx[(v, u)], j] = 1
                    # Note: edges are stored as (min, max) or (u,v) from generator?
                    # Generator adds (u,v) where u<v in K, and (u,v) where u<k, v>=k.
                    # So (u,v) is consistent with u<v?
                    # In generator: 
                    # for u in range(k): for v in range(u+1, k): edges.add((u,v)) -> u<v
                    # for v in range(k,n): for u in range(k): if ... edges.add((u,v)) -> u<k<=v -> u<v
                    # So all edges are (min, max).
                    
        # Objective: minimize sum(x)
        c = np.ones(M)
        
        # Constraints: A_eq * x = 1
        constraint = LinearConstraint(A_eq, 1, 1)
        bounds = Bounds(0, 1)
        integrality = np.ones(M)
        
        try:
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
            if res.status == 0:
                selected_indices = np.where(res.x > 0.5)[0]
                final_cliques = [list(cliques_list[i]) for i in selected_indices]
                # Ensure types are int
                final_cliques = [[int(x) for x in c] for c in final_cliques]
                return final_cliques
            else:
                # Fallback if ILP fails
                pass
        except Exception:
            pass
            
        # Fallback: Greedy Set Cover on candidates
        # Sort candidates by size descending? Or by "efficiency"?
        # Let's use a greedy approach: pick clique that covers most uncovered edges
        remaining_edges = set(edges)
        used_cliques = []
        
        # Precompute edges covered by each candidate
        clique_edges = []
        for clique in cliques_list:
            cl_set = set(clique)
            cl_list = list(cl_set)
            cl_edges = set()
            for i in range(len(cl_list)):
                for l in range(i + 1, len(cl_list)):
                    u, v = cl_list[i], cl_list[l]
                    if u > v: u, v = v, u
                    if (u, v) in remaining_edges:
                        cl_edges.add((u, v))
            clique_edges.append(cl_edges)
            
        # Greedy
        available_cliques = list(range(M))
        while remaining_edges:
            best_idx = -1
            best_count = -1
            for idx in available_cliques:
                count = len(clique_edges[idx].intersection(remaining_edges))
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1:
                break
            used_cliques.append(cliques_list[best_idx])
            remaining_edges.difference_update(clique_edges[best_idx])
            available_cliques.remove(best_idx)
            
        final_cliques = [list(c) for c in used_cliques]
        final_cliques = [[int(x) for x in c] for c in final_cliques]
        return final_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_instance(*spec))
        
    return results
