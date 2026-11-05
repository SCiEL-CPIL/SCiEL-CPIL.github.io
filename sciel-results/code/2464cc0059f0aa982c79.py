import time
import numpy as np
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        adj = np.zeros((n, n), dtype=bool)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u, v] = True
                    adj[v, u] = True
        
        if not edges:
            return []

        remaining_edges = set(edges)
        cliques = []
        
        # Precompute neighbor lists for fast access?
        # adj is boolean matrix.
        
        while remaining_edges:
            if time.time() > deadline:
                break

            # Pick an arbitrary edge to start a clique
            u, v = next(iter(remaining_edges))
            
            # Start with {u, v}
            clique = [u, v]
            
            # Candidates are vertices w such that:
            # 1. adj[u][w] and adj[v][w]
            # 2. (min(u,w), max(u,w)) in remaining_edges
            # 3. (min(v,w), max(v,w)) in remaining_edges
            
            # Get common neighbors of u and v
            common = np.where(adj[u] & adj[v])[0]
            
            valid_candidates = []
            for w in common:
                if w == u or w == v:
                    continue
                e1 = (min(u, w), max(u, w))
                e2 = (min(v, w), max(v, w))
                if e1 in remaining_edges and e2 in remaining_edges:
                    valid_candidates.append(w)
            
            # Greedily add candidates
            # To make the clique larger, we should pick candidates that are connected to each other.
            # Simple heuristic: pick the one with highest degree in the subgraph induced by valid_candidates.
            # Or just pick the first one?
            # Let's pick the one with highest degree in the remaining graph.
            
            while valid_candidates:
                # Find best candidate
                best_w = -1
                best_deg = -1
                for w in valid_candidates:
                    # Degree in remaining graph
                    deg = 0
                    # We can just use sum(adj[w]) but that includes removed edges.
                    # We need degree in remaining graph.
                    # This is expensive.
                    # Let's just pick the first one for speed.
                    best_w = w
                    break
                
                # Add best_w
                clique.append(best_w)
                
                # Update valid_candidates
                # Must be connected to best_w
                # And edges to best_w must be in remaining
                new_valid = []
                for z in valid_candidates:
                    if z == best_w:
                        continue
                    if adj[best_w][z]:
                        e = (min(best_w, z), max(best_w, z))
                        if e in remaining_edges:
                            new_valid.append(z)
                valid_candidates = new_valid

            # Remove edges in clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u_c, v_c = clique[i], clique[j]
                    edge = (min(u_c, v_c), max(u_c, v_c))
                    if edge in remaining_edges:
                        remaining_edges.remove(edge)
            
            if len(clique) >= 2:
                cliques.append(clique)
            else:
                # Fallback
                cliques.append([u, v])
                remaining_edges.remove((u, v))

        # Cover remaining edges if deadline hit
        if remaining_edges:
            for e in remaining_edges:
                cliques.append(list(e))
        
        return cliques

    res1 = instance(32, 0.6, 601)
    res2 = instance(64, 0.55, 602)
    return [res1, res2]
