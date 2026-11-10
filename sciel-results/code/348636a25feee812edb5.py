"""Rewritten program for clique partition of split graphs."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        # C vertices: 0..k-1
        # I vertices: k..n-1
        
        # Cross edges: list of (u, v) where u < k, v >= k
        cross_edges = []
        c_edges = []
        
        for u, v in edges:
            if u < k and v < k:
                c_edges.append((u, v))
            elif u < k and v >= k:
                cross_edges.append((u, v))
            elif u >= k and v < k:
                cross_edges.append((v, u))
                
        # Map neighbors in C for each v in I
        neighbors_in_c = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            neighbors_in_c[v].append(u)
            
        # Sort I vertices by degree descending to prioritize large merges
        i_vertices = sorted([v for v in range(k, n) if neighbors_in_c[v]], key=lambda x: len(neighbors_in_c[x]), reverse=True)
        
        cliques = []
        
        # We will track which internal edges of C are covered.
        # Initially, none are covered by "integrated" cliques.
        # We will decide at the end how to cover the rest.
        
        # Set of internal edges covered by integrated cliques
        covered_internal_edges = set()
        
        for v in i_vertices:
            neigh = neighbors_in_c[v]
            if not neigh:
                continue
                
            # We form a clique with v and all its neighbors in C.
            # This is the greedy choice to maximize savings per clique.
            clique = neigh + [v]
            cliques.append(clique)
            
            # Mark all internal edges within 'neigh' as covered
            for i in range(len(neigh)):
                for j in range(i+1, len(neigh)):
                    u1, u2 = neigh[i], neigh[j]
                    if u1 > u2:
                        u1, u2 = u2, u1
                    covered_internal_edges.add((u1, u2))
                    
        # Now we must cover the remaining internal edges of C.
        # Remaining edges are C_edges - covered_internal_edges
        remaining_edges = [e for e in c_edges if e not in covered_internal_edges]
        
        if remaining_edges:
            # Heuristic to cover remaining edges with minimal cliques
            # The remaining graph is a subgraph of a clique.
            # We can use a greedy approach: repeatedly find the largest clique in the remaining graph.
            # Since it's a subgraph of a complete graph, a clique is just a set of vertices pairwise connected in the remaining graph.
            
            # Build adjacency set for remaining edges for fast lookup
            rem_adj = {u: set() for u in range(k)}
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                
            remaining_edge_set = set(remaining_edges)
            
            while remaining_edge_set:
                # Find a vertex with the highest degree in the remaining graph
                # Or just pick the first edge's vertex
                if not remaining_edge_set:
                    break
                    
                # Pick a vertex u with max degree in remaining graph
                max_deg = -1
                start_u = -1
                for u in range(k):
                    deg = len(rem_adj[u] & {x for x in range(k) if (min(u,x), max(u,x)) in remaining_edge_set}) 
                    # This is inefficient. Let's just pick an edge.
                    break
                
                # Better: Just pick an arbitrary edge (u,v) and expand to a clique.
                # Find an edge
                u, v = next(iter(remaining_edge_set))
                
                # Start with clique {u, v}
                current_clique = [u, v]
                # Try to add other vertices w such that w is connected to all in current_clique in remaining graph
                # Since it's a subgraph of a clique, w is connected to all in current_clique iff all edges (w,x) for x in current_clique are in remaining_edge_set.
                
                # To make it fast, we just check candidates.
                # Candidates are neighbors of u in remaining graph.
                candidates = list(rem_adj[u])
                candidates.sort(key=lambda x: len(rem_adj[x]), reverse=True)
                
                for w in candidates:
                    if w == v:
                        continue
                    # Check if w is connected to all in current_clique
                    can_add = True
                    for x in current_clique:
                        edge = (min(w, x), max(w, x))
                        if edge not in remaining_edge_set:
                            can_add = False
                            break
                    if can_add:
                        current_clique.append(w)
                        
                # Add this clique
                cliques.append(current_clique)
                
                # Remove all edges within this clique from remaining_edge_set
                clique_set = set(current_clique)
                edges_to_remove = []
                for i in range(len(current_clique)):
                    for j in range(i+1, len(current_clique)):
                        u1, u2 = current_clique[i], current_clique[j]
                        edges_to_remove.append((min(u1, u2), max(u1, u2)))
                        
                for e in edges_to_remove:
                    remaining_edge_set.discard(e)
                    
                # Update rem_adj? Not strictly necessary if we rely on remaining_edge_set for validation,
                # but rem_adj helps find candidates. We can just rebuild or leave it (it will have stale edges, 
                # but the check `edge not in remaining_edge_set` filters them).
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback to naive
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for u, v in edges:
                if u < k and v >= k:
                    cliques.append([u, v])
                elif v < k and u >= k:
                    cliques.append([v, u])
            results.append(cliques)
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
