"""Optimized split-graph clique partition using matching-based assignment."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        start_time = time.time()
        deadline = start_time + 115
        
        rng = random.Random(seed)
        
        # Generate cross edges and neighborhoods
        N = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    N[v].add(u)
        
        # Strategy: 
        # 1. Decompose the clique C (first k vertices) into edge-disjoint matchings
        # 2. Assign each I-vertex v to a matching M where v is adjacent to all vertices in M
        # 3. The clique for v is M ∪ {v}
        # 4. Remaining C-edges are covered by additional cliques
        
        # First, let's try a simpler greedy approach:
        # For each I-vertex, try to find the largest subset of C it's connected to
        # that forms a matching (independent set in C's complement within its neighborhood)
        
        # Actually, let's use a different approach:
        # Greedily pack I-vertices into cliques with C-vertices
        
        # Key insight: If v ∈ I is adjacent to a set S ⊆ C, and S is a clique in C
        # (which it always is since C is a clique), then {v} ∪ S is a clique.
        # But we need to cover each edge exactly once.
        
        # The constraint is: each C-edge can appear in at most one clique.
        # So if we use {v} ∪ S_v for v, then all edges within S_v are "used up".
        
        # Strategy: 
        # - Start with all C-edges uncovered
        # - For each I-vertex v, find the largest subset S of N[v] such that 
        #   all edges within S are still uncovered
        # - Use {v} ∪ S as a clique, mark those C-edges as covered
        # - After processing all I-vertices, cover remaining C-edges with cliques
        
        # This is equivalent to: assign each I-vertex to a matching in the 
        # subgraph of C induced by its neighbors, where the matchings are edge-disjoint
        
        # Let's implement this:
        
        # Track which C-edges are still uncovered
        c_edges_remaining = set()
        for u in range(k):
            for w in range(u + 1, k):
                c_edges_remaining.add((u, w))
        
        cliques = []
        
        # Sort I-vertices by degree (larger first) to maximize packing
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda v: len(N[v]), reverse=True)
        
        for v in i_vertices:
            if time.time() > deadline:
                break
            nv = N[v]
            if not nv:
                continue
            
            # Find the largest subset S of nv such that all edges within S are in c_edges_remaining
            # This is a maximum clique problem in the graph where two vertices are adjacent
            # if their edge is still remaining
            
            # Build adjacency among nv based on remaining edges
            nv_list = sorted(nv)
            adj = {u: set() for u in nv_list}
            for i in range(len(nv_list)):
                for j in range(i + 1, len(nv_list)):
                    u, w = nv_list[i], nv_list[j]
                    if (u, w) in c_edges_remaining:
                        adj[u].add(w)
                        adj[w].add(u)
            
            # Greedy: find a large clique in this graph
            # Start with the vertex of highest degree
            if not nv_list:
                continue
            
            # Use a simple greedy: repeatedly add vertex with max connections to current clique
            clique_c = []
            candidates = list(nv_list)
            
            while candidates:
                # Find vertex with max degree to current clique
                best = None
                best_score = -1
                for c in candidates:
                    score = len([x for x in clique_c if x in adj.get(c, set())])
                    if score > best_score:
                        best_score = score
                        best = c
                if best is None or best_score < 0:
                    break
                clique_c.append(best)
                candidates.remove(best)
                # Remove vertices not connected to best
                candidates = [c for c in candidates if best in adj.get(c, set())]
            
            if not clique_c:
                # No C-vertices can be added, just make a 2-clique with one neighbor
                if nv:
                    u = min(nv)
                    cliques.append(sorted([u, v]))
                    # Remove edge (u, v) - but this is a cross edge, not a C-edge
                    # Cross edges are only covered by cliques containing both endpoints
                    # So we don't need to track them separately
                continue
            
            # Use {v} ∪ clique_c as a clique
            clique = [v] + sorted(clique_c)
            cliques.append(clique)
            
            # Mark C-edges within clique_c as covered
            for i in range(len(clique_c)):
                for j in range(i + 1, len(clique_c)):
                    u, w = clique_c[i], clique_c[j]
                    c_edges_remaining.discard((min(u, w), max(u, w)))
        
        # Cover remaining C-edges
        if c_edges_remaining:
            # Build adjacency for remaining C-edges
            rem_adj = {u: set() for u in range(k)}
            for u, w in c_edges_remaining:
                rem_adj[u].add(w)
                rem_adj[w].add(u)
            
            rem_copy = set(c_edges_remaining)
            while rem_copy:
                if time.time() > deadline:
                    # Fallback: cover each remaining edge as a 2-clique
                    for u, w in rem_copy:
                        cliques.append([u, w])
                    break
                
                # Find a maximal clique in the remaining graph
                # Use greedy: pick vertex with max degree in remaining edges
                degrees = {}
                for u in range(k):
                    degrees[u] = len([w for w in rem_adj[u] if (min(u, w), max(u, w)) in rem_copy])
                
                if not any(d > 0 for d in degrees.values()):
                    break
                
                start = max(degrees, key=degrees.get)
                clique = [start]
                candidates = [c for c in rem_adj[start] if (min(start, c), max(start, c)) in rem_copy]
                
                for c in candidates:
                    if all((min(c, u), max(c, u)) in rem_copy for u in clique):
                        clique.append(c)
                
                cliques.append(sorted(clique))
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        rem_copy.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
        
        # Verify: every clique must have at least 2 vertices
        # Filter out any invalid cliques (shouldn't happen, but just in case)
        cliques = [c for c in cliques if len(c) >= 2]
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
