"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return edges, cross_edges

    def solve_one(n, k, p, seed):
        c_edges, cross_edges = instance(n, k, p, seed)
        
        # Build adjacency for I vertices to C vertices
        # adj_i[v] = set of u in C adjacent to v in I
        adj_i = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            adj_i[v].add(u)
            
        # Identify all potential Type B cliques (triangles u, v, w)
        # u, v in C, w in I
        # Condition: w adjacent to u and v
        potential_triangles = []
        for u, v in c_edges:
            # Find w in I adjacent to both u and v
            # We can iterate through I vertices or use intersection
            # For efficiency, precompute neighbors in I for each C vertex?
            pass
        
        # Better: For each w in I, find all pairs (u, v) in C adjacent to w
        # These pairs form a clique in C (since C is a clique), so (u, v, w) is a triangle
        # We need to select a set of such triangles such that:
        # 1. Each C-edge (u, v) is in at most one triangle.
        # 2. For each w, the C-vertices used in triangles with w are disjoint (matching).
        
        # Let's build a list of candidate triangles
        candidates = []
        for w in range(k, n):
            neighbors = adj_i[w]
            # All pairs in neighbors form valid triangles
            for u, v in combinations(neighbors, 2):
                candidates.append((u, v, w))
        
        # Heuristic selection
        # Sort candidates by degree of u and v? Or just random?
        # Let's try to maximize count.
        # Greedy: Pick a candidate, mark its C-edge as used, mark its C-vertices as used for w.
        
        used_c_edge = set()
        used_c_vert_for_w = {w: set() for w in range(k, n)}
        selected = []
        
        # Shuffle to avoid bias? Or sort by something?
        # Sorting by number of common neighbors?
        # Let's just shuffle and pick greedily.
        random.shuffle(candidates)
        
        for u, v, w in candidates:
            if (u, v) in used_c_edge:
                continue
            if u in used_c_vert_for_w[w] or v in used_c_vert_for_w[w]:
                continue
            
            # Select this triangle
            selected.append((u, v, w))
            used_c_edge.add((u, v))
            used_c_vert_for_w[w].add(u)
            used_c_vert_for_w[w].add(v)
            
        # Now we have selected Type B cliques
        cliques = [[u, v, w] for u, v, w in selected]
        
        # Remaining C-edges
        remaining_c_edges = [e for e in c_edges if e not in used_c_edge]
        
        # Cover remaining C-edges with Type A cliques
        # Greedy Max Clique on the graph of remaining edges
        # Build adjacency for remaining C subgraph
        rem_adj = {u: set() for u in range(k)}
        for u, v in remaining_c_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        covered_verts = set()
        while remaining_c_edges:
            # Find a large clique in remaining_c_edges
            # Simple greedy: pick a vertex with high degree in remaining, then its neighbors
            # Or just pick an edge and expand?
            # Let's pick the vertex with max degree in remaining graph
            max_deg_vert = max(remaining_c_edges[0][0], remaining_c_edges[0][1], key=lambda x: len(rem_adj[x]))
            # Actually, degrees change as we remove edges.
            # Let's just pick the first edge and try to expand it into a clique
            u, v = remaining_c_edges[0]
            current_clique = {u, v}
            # Try to add other vertices
            # Vertices that are connected to all in current_clique
            candidates_for_clique = rem_adj[u].intersection(rem_adj[v])
            # Remove u, v from candidates
            candidates_for_clique -= {u, v}
            
            # Greedily add vertices
            while candidates_for_clique:
                # Pick one
                w_cand = next(iter(candidates_for_clique))
                current_clique.add(w_cand)
                # Update candidates: must be connected to w_cand
                candidates_for_clique = candidates_for_clique.intersection(rem_adj[w_cand])
                candidates_for_clique -= {w_cand}
                
            # This clique covers all edges within current_clique
            cliques.append(list(current_clique))
            
            # Remove these edges from remaining_c_edges and rem_adj
            verts_in_clique = list(current_clique)
            for i in range(len(verts_in_clique)):
                for j in range(i+1, len(verts_in_clique)):
                    u, v = verts_in_clique[i], verts_in_clique[j]
                    if (u, v) in rem_adj[u]:
                        rem_adj[u].remove(v)
                        rem_adj[v].remove(u)
                        
            # Filter remaining_c_edges
            remaining_c_edges = [(u, v) for u, v in remaining_c_edges if (u, v) in rem_adj[u]]
            
        # Cover remaining cross-edges with Type C cliques (edges)
        # Cross edges not covered by Type B
        covered_cross = set()
        for u, v, w in selected:
            covered_cross.add((u, w))
            covered_cross.add((v, w))
            
        for u, w in cross_edges:
            if (u, w) not in covered_cross:
                cliques.append([u, w])
                
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback: return seed solution for remaining
            from random import Random
            def instance_fallback(n, k, p, seed):
                rng = Random(seed)
                edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
                edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
                return sorted(edges)
            results.append([list(e) for e in instance_fallback(*spec)])
            continue
            
        cliques = solve_one(*spec)
        results.append(cliques)
        
    return results

# EVOLVE-BLOCK-END
