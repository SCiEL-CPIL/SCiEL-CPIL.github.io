import time
import numpy as np
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        # Generate edges
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

        # active_adj tracks which edges are still available to be covered
        active_adj = adj.copy()
        cliques = []
        
        # Pre-compute initial degrees to find high-degree vertices quickly
        # We will recompute or update as needed, but for N=64, full scan is cheap.
        
        while True:
            if time.time() > deadline:
                # Fallback: cover remaining edges individually to ensure validity
                # This is O(E) which is fine for N=64 (max 2016 edges)
                # But we need to find them.
                for u in range(n):
                    for v in range(u + 1, n):
                        if active_adj[u, v]:
                            cliques.append([u, v])
                            active_adj[u, v] = False
                            active_adj[v, u] = False
                return cliques

            # Find the vertex with the highest active degree
            degrees = active_adj.sum(axis=1)
            # If no edges left, break
            if degrees.max() == 0:
                break
            
            # Pick the vertex with max degree
            start_v = int(np.argmax(degrees))
            
            # Start clique with start_v
            clique = [start_v]
            
            # Candidates are neighbors of start_v that have an active edge to start_v
            # Since active_adj is symmetric and we only add edges if they exist, 
            # neighbors are just where active_adj[start_v] is True.
            candidates = [int(i) for i in np.where(active_adj[start_v])[0] if i != start_v]
            
            # Greedily expand the clique
            # We want to add a vertex w to the clique if it is connected to ALL vertices in the clique.
            # And we want to maximize the potential for further expansion, so we pick the candidate
            # that has the most connections to other candidates.
            
            while candidates:
                # Check which candidates are connected to all vertices in the current clique
                # For efficiency, we can filter candidates:
                # A candidate w is valid if active_adj[w, c] is True for all c in clique.
                
                valid_candidates = []
                for w in candidates:
                    is_connected_to_all = True
                    for c in clique:
                        if not active_adj[w, c]:
                            is_connected_to_all = False
                            break
                    if is_connected_to_all:
                        valid_candidates.append(w)
                
                if not valid_candidates:
                    break
                
                # Pick the best candidate from valid_candidates
                # Heuristic: Maximize connections to other valid_candidates
                best_w = -1
                max_conn = -1
                for w in valid_candidates:
                    conn = 0
                    for z in valid_candidates:
                        if w != z and active_adj[w, z]:
                            conn += 1
                    if conn > max_conn:
                        max_conn = conn
                        best_w = w
                
                if best_w == -1:
                    break
                
                # Add best_w to clique
                clique.append(best_w)
                
                # Update candidates:
                # New candidates must be connected to best_w (and implicitly to all previous clique members 
                # because we will re-filter in the next iteration, but we can prune now for speed)
                # Actually, the next iteration's filter `is_connected_to_all` will handle the clique constraint.
                # We just need to remove best_w from candidates and optionally prune those not connected to best_w 
                # to reduce the search space for the next step's "valid_candidates" calculation.
                
                # Prune candidates: keep only those connected to best_w
                candidates = [w for w in candidates if w != best_w and active_adj[best_w, w]]
                
                # If no candidates left, break
                if not candidates:
                    break
            
            # Remove edges within the clique from active_adj
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u_c, v_c = clique[i], clique[j]
                    active_adj[u_c, v_c] = False
                    active_adj[v_c, u_c] = False
            
            if len(clique) >= 2:
                cliques.append(clique)
            else:
                # Should not happen if we picked a vertex with degree > 0, 
                # but just in case, if clique is size 1, we didn't cover any edges?
                # If clique is size 1, it means no candidate was connected to it?
                # But we picked start_v with max degree > 0, so it has neighbors.
                # If no neighbor was "valid" (connected to all in clique), that's impossible for size 1 clique 
                # because the only constraint is being a neighbor.
                # So clique should always be size >= 2 if degree > 0.
                pass

        return cliques

    res1 = instance(32, 0.6, 601)
    res2 = instance(64, 0.55, 602)
    return [res1, res2]
