"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build adjacency for cross edges
        # C: 0..k-1, I: k..n-1
        # neighbors[v] for v in I is a set of u in C
        neighbors = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].add(u)
        return n, k, neighbors

    def solve_instance(n, k, neighbors):
        cliques = []
        # covered_edges is a set of frozensets or tuples (u, v) with u < v
        covered_internal_edges = set()
        
        # Step 1: Handle independent set vertices
        for v in range(k, n):
            N_v = neighbors[v]
            if not N_v:
                continue
            # Create clique {v} U N_v
            clique = [v] + list(N_v)
            cliques.append(clique)
            
            # Mark internal edges of N_v as covered
            N_v_list = list(N_v)
            for i in range(len(N_v_list)):
                for j in range(i + 1, len(N_v_list)):
                    u1 = N_v_list[i]
                    u2 = N_v_list[j]
                    if u1 > u2:
                        u1, u2 = u2, u1
                    covered_internal_edges.add((u1, u2))
        
        # Step 2: Handle remaining internal edges in C
        # Total internal edges in C
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_internal_edges:
                    remaining_edges.add((u, w))
        
        # Greedy cover of remaining_edges
        # We work on the graph induced by remaining_edges
        # To speed up, we can maintain a list of edges and remove them
        # Since k is small (<=30), we can just iterate
        
        # Build adjacency list for the remaining graph
        # adj[u] = set of w such that (min(u,w), max(u,w)) in remaining_edges
        adj = [set() for _ in range(k)]
        for (u, w) in remaining_edges:
            adj[u].add(w)
            adj[w].add(u)
            
        # Greedy Max Clique
        # We need to find a maximal clique, remove its edges, repeat.
        # To be more efficient, we can pick a vertex with max degree, 
        # then find a maximal clique in its neighborhood.
        
        while remaining_edges:
            # Find vertex with max degree
            max_deg = -1
            start_v = -1
            for u in range(k):
                deg = len(adj[u])
                if deg > max_deg:
                    max_deg = deg
                    start_v = u
            
            if start_v == -1:
                break # Should not happen if remaining_edges is not empty
                
            # Find a maximal clique containing start_v
            # Candidates must be neighbors of start_v
            # We use a simple greedy: 
            # Start with {start_v}
            # Add neighbors of all current members that are mutually adjacent
            
            current_clique = [start_v]
            # Intersection of neighbors of all members in current_clique
            # Initially neighbors of start_v
            candidates = set(adj[start_v])
            
            while True:
                # Pick a candidate that is adjacent to all current_clique members
                # Since candidates is intersection, all are adjacent to all.
                # We just need to pick one and update candidates to its neighbors.
                # To maximize clique size, pick one with max degree in the subgraph?
                # Or just first one?
                # Let's pick the one with max degree in the remaining graph (or candidates)
                best_cand = -1
                best_deg = -1
                for c in candidates:
                    # Degree in the subgraph induced by candidates?
                    # Or just global degree?
                    # Let's use degree in candidates to keep it tight
                    deg_in_cand = len(adj[c] & candidates)
                    if deg_in_cand > best_deg:
                        best_deg = deg_in_cand
                        best_cand = c
                
                if best_cand == -1:
                    break
                
                current_clique.append(best_cand)
                # Update candidates: must be neighbors of best_cand
                candidates = candidates & adj[best_cand]
                
            # We have a maximal clique (in the sense that no more vertices can be added)
            # Add to cliques
            cliques.append(current_clique)
            
            # Remove edges of this clique from remaining_edges and adj
            clique_set = set(current_clique)
            # Generate all edges in clique
            clique_edges = set()
            clique_list = sorted(current_clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    u1 = clique_list[i]
                    u2 = clique_list[j]
                    clique_edges.add((u1, u2))
            
            for (u, w) in clique_edges:
                if (u, w) in remaining_edges:
                    remaining_edges.remove((u, w))
                    adj[u].remove(w)
                    adj[w].remove(u)
                    
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        n, k, neighbors = instance(*spec)
        cliques = solve_instance(n, k, neighbors)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
