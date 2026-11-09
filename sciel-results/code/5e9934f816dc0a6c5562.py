"""New program using Iterative Maximal Clique Extraction."""
import numpy as np
import random

def construct():
    def instance(n, p, seed):
        r = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if r.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, edges):
        if not edges:
            return []
        
        # Build initial adjacency masks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            
        remaining_edges = set(edges)
        cliques = []
        
        while remaining_edges:
            # Find a vertex with the highest degree in the remaining graph
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            if max(deg) == 0:
                break
                
            # Pick vertex with max degree
            u_start = max(range(n), key=lambda x: deg[x])
            
            # Greedy Max Clique Expansion from u_start
            clique = [u_start]
            
            # Candidates must be neighbors of u_start AND have remaining edges to u_start
            candidates = []
            for v in range(n):
                if v == u_start:
                    continue
                if not (adj[u_start] & (1 << v)):
                    continue
                e = (min(u_start, v), max(u_start, v))
                if e in remaining_edges:
                    candidates.append(v)
                    
            while candidates:
                # Find candidate v that maximizes the number of remaining edges to other candidates
                best_v = -1
                best_score = -1
                
                for v in candidates:
                    score = 0
                    for w in candidates:
                        if w == v:
                            continue
                        if not (adj[v] & (1 << w)):
                            continue
                        e = (min(v, w), max(v, w))
                        if e in remaining_edges:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_v = v
                        
                if best_v == -1:
                    break
                    
                clique.append(best_v)
                
                # Update candidates: must be connected to best_v via remaining edge
                new_candidates = []
                for w in candidates:
                    if w == best_v:
                        continue
                    if not (adj[best_v] & (1 << w)):
                        continue
                    e = (min(best_v, w), max(best_v, w))
                    if e in remaining_edges:
                        new_candidates.append(w)
                candidates = new_candidates
                
            if len(clique) >= 2:
                cliques.append(clique)
                # Remove all edges within this clique from remaining_edges
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (min(a, b), max(a, b))
                        remaining_edges.discard(e)
            else:
                # Fallback for isolated edges (should be rare)
                # Find one remaining edge
                u, v = next(iter(remaining_edges))
                cliques.append([u, v])
                remaining_edges.discard((u, v))

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = solve_instance(n, edges)
        result.append(cliques)

    return result
