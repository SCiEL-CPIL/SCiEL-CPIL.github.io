"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def clique_partition(n, p, seed):
        adj = get_graph(n, p, seed)
        # Build list of all edges
        edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.add((u, v))
        
        remaining_edges = set(edges)
        cliques = []
        
        # Helper to greedily expand a clique starting from a seed edge (u, v)
        def expand_clique(seed_u, seed_v, current_remaining):
            # Start with the seed edge
            clique_vertices = {seed_u, seed_v}
            
            # Candidates are common neighbors of seed_u and seed_v
            # that are connected to all current clique vertices
            # And crucially, the edges connecting them to the clique must be in current_remaining
            candidates = list(adj[seed_u] & adj[seed_v])
            
            # Sort candidates by degree in the remaining graph? Or just iterate?
            # For speed, we'll just iterate through candidates and try to add them.
            # To make it greedy "largest", we might want to prioritize vertices with high potential.
            # But for now, simple greedy addition.
            
            while candidates:
                best_candidate = None
                best_candidate_score = -1
                
                # To improve quality, we could score candidates by how many other candidates they connect to
                # But that's expensive. Let's try a simpler heuristic:
                # Pick the candidate that connects to the most other candidates?
                # Or just pick the first valid one?
                # "Largest clique" greedy usually implies picking the vertex that allows the most future growth.
                
                # Let's try: for each candidate w, check if it connects to all in clique_vertices
                # AND all those edges are in current_remaining.
                # If multiple, pick one with highest degree in the subgraph of candidates?
                
                valid_candidates = []
                for w in candidates:
                    # Check if w connects to all in clique_vertices
                    ok = True
                    for x in clique_vertices:
                        e = (min(w, x), max(w, x))
                        if e not in current_remaining:
                            ok = False
                            break
                    if ok:
                        valid_candidates.append(w)
                
                if not valid_candidates:
                    break
                
                # Heuristic: pick the candidate that is connected to the most other valid candidates
                # This tends to build larger cliques
                best_w = None
                max_connections = -1
                for w in valid_candidates:
                    connections = 0
                    for other_w in valid_candidates:
                        if w == other_w:
                            continue
                        # Check if w and other_w are connected in the original graph
                        # And if the edge is in remaining? No, just original graph connectivity is enough for candidate set
                        # But actually, for the clique to be valid, w and other_w don't need to be connected yet,
                        # but if they are, it's a sign of density.
                        # Let's just pick the one with highest degree in the original graph among valid candidates?
                        # Or just the first one?
                        # Let's try: pick the one with the most connections to other valid candidates in the ORIGINAL graph.
                        if w in adj[other_w]:
                            connections += 1
                    if connections > max_connections:
                        max_connections = connections
                        best_w = w
                
                if best_w is None:
                    break
                    
                # Add best_w to clique
                clique_vertices.add(best_w)
                candidates.remove(best_w)
                
                # Update candidates: remove any that are not connected to best_w
                # Or more strictly, remove any for which the edge to best_w is not in remaining
                # Since we checked connectivity to clique_vertices (which now includes best_w) in the next loop,
                # we can just let the loop handle it. But we should prune candidates that can no longer form a clique.
                new_candidates = []
                for w in candidates:
                    # Check if w is still connected to all in clique_vertices via remaining edges
                    ok = True
                    for x in clique_vertices:
                        e = (min(w, x), max(w, x))
                        if e not in current_remaining:
                            ok = False
                            break
                    if ok:
                        new_candidates.append(w)
                candidates = new_candidates
                
            return clique_vertices

        while remaining_edges:
            if time.time() > deadline - 5.0:
                break
            
            # Find the most constrained edge
            # Constrained = fewest common neighbors in the remaining graph?
            # Or fewest common neighbors in the original graph that are still "available"?
            # Let's use: for edge (u,v), count how many w in adj[u] & adj[v] 
            # such that (u,w) and (v,w) are in remaining_edges.
            # If this count is 0, the edge is "stranded" (cannot be extended).
            # We want to process these first to avoid leaving them for last?
            # Actually, if we process them first, we just take the edge (clique of size 2).
            # If we process them last, they might be stranded.
            # So processing "most constrained" (lowest extension potential) first is good.
            
            best_edge = None
            min_constraints = float('inf')
            
            # Sample edges if too many for performance, but n=64 is small enough to check all?
            # Number of edges ~ 64*63/2 * 0.55 ~ 1100.
            # Checking constraints for 1100 edges * 64 candidates = ~70k ops. Fast.
            
            for e in remaining_edges:
                u, v = e
                # Count extensions
                extensions = 0
                for w in adj[u]:
                    if w in adj[v]:
                        # Check if edges (u,w) and (v,w) are in remaining
                        # (u,v) is already in remaining
                        if (min(u, w), max(u, w)) in remaining_edges and (min(v, w), max(v, w)) in remaining_edges:
                            extensions += 1
                
                if extensions < min_constraints:
                    min_constraints = extensions
                    best_edge = e
            
            if best_edge is None:
                break
                
            # Expand clique from best_edge
            u, v = best_edge
            clique_vertices = expand_clique(u, v, remaining_edges)
            
            # Remove all edges of this clique from remaining_edges
            clique_edges = set()
            verts = list(clique_vertices)
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    e = (min(verts[i], verts[j]), max(verts[i], verts[j]))
                    if e in remaining_edges:
                        clique_edges.add(e)
            
            if clique_edges:
                cliques.append(verts)
                remaining_edges -= clique_edges
            else:
                # Should not happen if best_edge is in remaining
                break
        
        # Handle any remaining edges if we hit the deadline or something went wrong
        if remaining_edges:
            for e in remaining_edges:
                cliques.append([e[0], e[1]])
        
        return cliques
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = clique_partition(n, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
