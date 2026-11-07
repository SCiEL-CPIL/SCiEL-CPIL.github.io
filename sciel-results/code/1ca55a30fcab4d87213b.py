"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Generate internal edges of the clique C
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        # Generate cross edges between C and I
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency sets for faster lookup
        # adj[u] is a set of neighbors of u
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cliques = []
        
        # Set of internal edges (u, v) with u < v < k that are already covered
        covered_internal_edges = set()
        
        # Process each vertex w in the independent set I (indices k to n-1)
        for w in range(k, n):
            # Find neighbors of w in C (indices 0 to k-1)
            neighbors_in_c = [u for u in range(k) if u in adj[w]]
            
            if len(neighbors_in_c) >= 2:
                # Form clique K_w = neighbors_in_c + [w]
                clique = neighbors_in_c + [w]
                cliques.append(clique)
                
                # Mark all internal edges within neighbors_in_c as covered
                for i in range(len(neighbors_in_c)):
                    for j in range(i + 1, len(neighbors_in_c)):
                        u = neighbors_in_c[i]
                        v = neighbors_in_c[j]
                        covered_internal_edges.add((min(u, v), max(u, v)))
                        
            elif len(neighbors_in_c) == 1:
                # Form clique {u, w}
                u = neighbors_in_c[0]
                cliques.append([u, w])
                # No internal edges covered here
        
        # Now, find internal edges of C that are NOT covered
        # Internal edges are (u, v) with 0 <= u < v < k
        remaining_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) in adj[u]: # It is an edge in the graph
                    if (u, v) not in covered_internal_edges:
                        remaining_edges.append((u, v))
        
        # We need to cover remaining_edges with cliques from C.
        # Since these are edges in a clique C, any subset of vertices forms a clique.
        # We want to partition the set of edges into cliques.
        # This is equivalent to finding a clique cover of the graph G_rem = (C, remaining_edges).
        # Greedy approach:
        # While remaining_edges is not empty:
        #   Find a clique S in G_rem that covers the most edges.
        #   A clique in G_rem is a set of vertices S such that all pairs in S are in remaining_edges.
        #   This is the same as finding a clique in the graph where edges are remaining_edges.
        #   We can find the maximum clique in the remaining graph.
        
        # To make it efficient, we can just try to find large cliques.
        # Since k is small (<= 30), we can use a simple heuristic.
        
        rem_edges_set = set(remaining_edges)
        
        while rem_edges_set:
            # Find a clique in the graph defined by rem_edges_set
            # Let's try to find the maximum clique.
            # Simple heuristic: Start with a vertex with high degree in rem graph, expand.
            
            # Build adjacency for remaining graph
            rem_adj = {u: set() for u in range(k)}
            for u, v in rem_edges_set:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                
            # Pick a vertex with max degree
            if not rem_adj:
                break
                
            max_deg_vertex = max(rem_adj, key=lambda x: len(rem_adj[x]))
            
            # Try to grow a clique starting from max_deg_vertex
            # Simple greedy: add vertices that are connected to all current clique members
            current_clique = [max_deg_vertex]
            candidates = list(rem_adj[max_deg_vertex])
            
            # Sort candidates by degree to prioritize
            candidates.sort(key=lambda x: len(rem_adj[x]), reverse=True)
            
            for c in candidates:
                # Check if c is connected to all in current_clique
                if all(c in rem_adj[cc] for cc in current_clique):
                    current_clique.append(c)
            
            # Add this clique
            if len(current_clique) >= 2:
                cliques.append(current_clique)
                
                # Remove edges within this clique from rem_edges_set
                for i in range(len(current_clique)):
                    for j in range(i + 1, len(current_clique)):
                        u = current_clique[i]
                        v = current_clique[j]
                        edge = (min(u, v), max(u, v))
                        if edge in rem_edges_set:
                            rem_edges_set.remove(edge)
                            
                # Update rem_adj (optional for correctness if we just use set, but good for next iter)
                # We rebuild rem_adj in next loop anyway.
            else:
                # Fallback: if we couldn't form a clique of size >= 2 (shouldn't happen if edges exist)
                # Just take one edge
                u, v = next(iter(rem_edges_set))
                cliques.append([u, v])
                rem_edges_set.remove((u, v))
                
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    for n, k, p, seed in specs:
        if time.time() > deadline:
            # Return what we have, padded with empty lists if necessary? 
            # No, we must return 3 lists. If we run out of time, we fail.
            # But the problem is small, so this shouldn't happen.
            pass
            
        cliques = solve_one(n, k, p, seed)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
