import numpy as np
from itertools import combinations

def construct():
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(graph, n):
        """
        Finds all maximal cliques in a graph.
        graph: list of edges
        n: number of vertices
        Returns a list of sets, each set is a maximal clique.
        """
        # Build adjacency list and sets for fast intersection
        adj = [set() for _ in range(n)]
        for u, v in graph:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if R: # Only add if not empty, though R can be empty for isolated vertices (size 1)
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            pivot = None
            max_degree = -1
            for v in P.union(X):
                deg = len(P.intersection(adj[v]))
                if deg > max_degree:
                    max_degree = deg
                    pivot = v
            
            # Iterate over vertices in P \ N(pivot)
            for v in list(P - adj[pivot]):
                R_new = R | {v}
                P_new = P & adj[v]
                X_new = X & adj[v]
                
                bk(R_new, P_new, X_new)
                
                P.remove(v)
                X.add(v)
                
        bk(set(), set(range(n)), set())
        return [list(c) for c in cliques]

    def solve_transversal(n, edges, cliques):
        # Filter cliques of size >= 2
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []
        
        num_cliques = len(relevant_cliques)
        full_mask = (1 << num_cliques) - 1
        
        # Precompute vertex -> cliques covered (as bitmask)
        # and vertex -> list of clique indices
        vertex_clique_indices = [set() for _ in range(n)]
        for i, c in enumerate(relevant_cliques):
            for v in c:
                vertex_clique_indices[v].add(i)
                
        vertex_clique_mask = [0] * n
        for v in range(n):
            mask = 0
            for i in vertex_clique_indices[v]:
                mask |= (1 << i)
            vertex_clique_mask[v] = mask
            
        # Sort vertices by coverage descending
        order = sorted(range(n), key=lambda v: len(vertex_clique_indices[v]), reverse=True)
        
        # Greedy Set Cover for initial upper bound
        def greedy_cover():
            covered = 0
            selected = []
            remaining = full_mask
            while remaining:
                best_v = -1
                best_gain = 0
                for v in order:
                    gain = bin(vertex_clique_mask[v] & remaining).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1:
                    break
                selected.append(best_v)
                remaining &= ~vertex_clique_mask[best_v]
            return selected
            
        init_sol = greedy_cover()
        best_size = len(init_sol)
        best_solution = list(init_sol)
        
        # Lower bound function
        # Precompute max coverage for any subset? No.
        # Use: ceil(num_remaining / max_possible_coverage)
        # max_possible_coverage for a specific remaining set is max(popcount(mask[v] & remaining))
        
        def lower_bound(remaining_mask):
            if remaining_mask == 0:
                return 0
            # Find max coverage among all vertices for the remaining cliques
            max_cov = 0
            for v in range(n):
                cov = bin(vertex_clique_mask[v] & remaining_mask).count('1')
                if cov > max_cov:
                    max_cov = cov
            if max_cov == 0:
                return 0 # Should not happen if remaining_mask != 0 and valid input
            num_rem = bin(remaining_mask).count('1')
            return (num_rem + max_cov - 1) // max_cov
            
        def search(idx, current_size, covered_mask, current_set):
            nonlocal best_size, best_solution
            
            if current_size >= best_size:
                return
                
            if covered_mask == full_mask:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(current_set)
                return
                
            if idx == n:
                return
                
            remaining_mask = full_mask & ~covered_mask
            
            # Prune with lower bound
            if current_size + lower_bound(remaining_mask) >= best_size:
                return
                
            v = order[idx]
            
            # Check if v is mandatory
            # If v covers a clique that no other vertex in order[idx+1:] covers, we must take v.
            mandatory = False
            for ci in vertex_clique_indices[v]:
                if (remaining_mask >> ci) & 1:
                    # Check if any u in order[idx+1:] covers ci
                    found = False
                    for k in range(idx + 1, n):
                        u = order[k]
                        if ci in vertex_clique_indices[u]:
                            found = True
                            break
                    if not found:
                        mandatory = True
                        break
                        
            # Branch 1: Include v
            if vertex_clique_mask[v]:
                search(idx + 1, current_size + 1, covered_mask | vertex_clique_mask[v], current_set + [v])
                
            # Branch 2: Exclude v (if not mandatory)
            if not mandatory:
                search(idx + 1, current_size, covered_mask, current_set)
                
        search(0, 0, 0, [])
        
        return sorted(best_solution)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        cliques = bron_kerbosch(edges, n)
        transversal = solve_transversal(n, edges, cliques)
        out.append(sorted(transversal))
        
    return out
