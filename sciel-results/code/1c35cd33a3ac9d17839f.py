"""Clique partition of dense random graphs - minimize number of cliques."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    _start = time.time()
    _DEADLINE = 145.0  # safety margin before 150s kill

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_clique_partition(edges, n, start_time, deadline):
        """
        Optimized greedy clique partition.
        Uses set membership for remaining edges to avoid rebuilding adjacency lists.
        """
        if not edges:
            return []

        remaining = set(edges)
        cliques = []
        
        # Pre-compute adjacency lists for the original graph
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        while remaining:
            if time.time() > deadline - 1.0:
                break

            # Find vertices that still have edges in 'remaining'
            # We can just iterate through all vertices and check if they have any remaining edges
            # Or maintain a set of active vertices. For n=64, iterating all is fine.
            
            # Calculate degree in remaining graph for each vertex to prioritize
            deg = [0] * n
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            # Get active vertices sorted by degree
            active_verts = [i for i in range(n) if deg[i] > 0]
            if not active_verts:
                break
            active_verts.sort(key=lambda x: deg[x], reverse=True)
            
            best_clique = None
            best_size = 0

            # Try to build a clique starting from top candidates
            # Limit candidates to top 10-20 to save time
            candidates_to_try = active_verts[:min(20, len(active_verts))]
            
            for start in candidates_to_try:
                # Get neighbors of 'start' in the remaining graph
                rem_neighbors = []
                for v in adj[start]:
                    if (min(start, v), max(start, v)) in remaining:
                        rem_neighbors.append(v)
                
                if not rem_neighbors:
                    continue

                # Greedy clique expansion
                clique = [start]
                candidates = set(rem_neighbors)
                
                while candidates:
                    # Pick candidate with highest "local degree" within candidates
                    # i.e., number of neighbors in 'candidates' that are also in 'remaining'
                    best_cand = None
                    max_local_deg = -1
                    for c in candidates:
                        local_deg = 0
                        for w in adj[c]:
                            if w in candidates and (min(c, w), max(c, w)) in remaining:
                                local_deg += 1
                        if local_deg > max_local_deg:
                            max_local_deg = local_deg
                            best_cand = c
                    
                    if best_cand is None:
                        break
                        
                    clique.append(best_cand)
                    
                    # Update candidates: keep only those connected to best_cand in remaining graph
                    new_candidates = set()
                    for w in candidates:
                        if w != best_cand and (min(best_cand, w), max(best_cand, w)) in remaining:
                            new_candidates.add(w)
                    candidates = new_candidates
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique

            if best_clique is None or len(best_clique) < 2:
                # Fallback: take any single edge
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                continue

            # Add the selected clique
            cliques.append(best_clique)
            
            # Remove all edges covered by this clique from 'remaining'
            clique_set = set(best_clique)
            edges_to_remove = set()
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    edges_to_remove.add((min(u, v), max(u, v)))
            
            remaining -= edges_to_remove

        # Handle remaining edges as 2-cliques if deadline hit
        for e in remaining:
            cliques.append(list(e))

        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = generate_edges(n, p, seed)
        
        # Allocate time budget
        # 70s per instance, 5s buffer
        instance_deadline = _start + 70.0
        
        cliques = greedy_clique_partition(edges, n, _start, instance_deadline)
        
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
