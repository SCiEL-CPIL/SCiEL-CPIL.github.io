"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.add((u, v))
        
        # Greedy clique partition
        remaining = set(edges)
        cliques = []
        
        while remaining:
            # Pick an edge to start from (highest degree sum heuristic)
            best_edge = None
            best_score = -1
            # Sample some edges to find a good starting point
            sample = list(remaining)
            random.Random(seed + len(cliques)).shuffle(sample)
            for e in sample[:min(100, len(sample))]:
                u, v = e
                score = len(adj[u] & adj[v])
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                # No edges left that form triangles, take any
                best_edge = next(iter(remaining))
            
            # Build a clique starting from best_edge
            u0, v0 = best_edge
            clique = {u0, v0}
            # Common neighbors of all in clique
            common = adj[u0] & adj[v0]
            
            # Greedily add vertices from common that are connected to all in clique
            candidates = list(common)
            random.Random(seed + len(cliques) + 1).shuffle(candidates)
            for w in candidates:
                if w in clique:
                    continue
                # Check if w is connected to all in clique
                if all(w in adj[x] for x in clique):
                    clique.add(w)
                    common = common & adj[w]
            
            # Convert to list, ensure at least 2 vertices
            clique_list = sorted(clique)
            if len(clique_list) >= 2:
                cliques.append(clique_list)
                # Remove all edges in this clique from remaining
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        e = (clique_list[i], clique_list[j])
                        if e in remaining:
                            remaining.discard(e)
            else:
                # Shouldn't happen since we started with an edge
                break
        
        return cliques
    
    return [instance(32, 0.6, 601), instance(64, 0.55, 602)]
# EVOLVE-BLOCK-END
