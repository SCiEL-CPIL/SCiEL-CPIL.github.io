import sys
from random import Random

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        # Generate edges
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        
        # Build adjacency bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        remaining_adj = list(adj)
        cliques = []
        
        def find_max_clique(P, local_adj):
            # P is bitmask of candidates
            # local_adj is list of bitmasks for the subgraph induced by P
            # Actually, we can just use the global remaining_adj and mask with P?
            # No, local_adj is more efficient if we pass it.
            # But in the main loop, we have remaining_adj.
            # The subgraph is induced by P.
            # The adjacency of u in this subgraph is remaining_adj[u] & P.
            # So we don't need to build a separate local_adj list.
            # We just need to pass remaining_adj and P.
            
            best = []
            
            def bk(P, X, R):
                nonlocal best
                if not P and not X:
                    if len(R) > len(best):
                        best = list(R)
                    return
                if not P:
                    return
                    
                # Choose pivot
                pivot = -1
                max_deg = -1
                p_temp = P
                while p_temp:
                    lsb = p_temp & -p_temp
                    idx = lsb.bit_length() - 1
                    deg = (remaining_adj[idx] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        pivot = idx
                    p_temp ^= lsb
                    
                if pivot == -1:
                    return
                
                # Candidates P \ N(pivot)
                # N(pivot) in P is remaining_adj[pivot] & P
                candidates = P & ~(remaining_adj[pivot] & P)
                
                c_temp = candidates
                while c_temp:
                    lsb = c_temp & -c_temp
                    v = lsb.bit_length() - 1
                    # Recursive call
                    new_P = P & remaining_adj[v]
                    bk(new_P, X, R + [v])
                    
                    P ^= lsb
                    X |= lsb
                    c_temp ^= lsb
                    
            bk(P, 0, [])
            return best

        while True:
            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if remaining_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break
                
            # Compute degrees
            deg = [0] * n
            for i in range(n):
                deg[i] = remaining_adj[i].bit_count()
                
            # Top K vertices
            active = [i for i in range(n) if deg[i] > 0]
            if not active:
                break
                
            # Sort by degree descending
            active.sort(key=lambda x: deg[x], reverse=True)
            top_k = active[:5]
            
            best_clique = []
            
            for v in top_k:
                # Neighbors of v in remaining graph
                P = remaining_adj[v]
                if P == 0:
                    continue
                
                # Find max clique in subgraph induced by P
                # Note: The clique will be subset of P.
                # We add v to it.
                clique_in_neighbors = find_max_clique(P, remaining_adj)
                candidate = clique_in_neighbors + [v]
                
                if len(candidate) > len(best_clique):
                    best_clique = candidate
            
            if not best_clique:
                # Fallback: pick any edge
                # Find first edge
                found = False
                for i in range(n):
                    if remaining_adj[i]:
                        lsb = remaining_adj[i] & -remaining_adj[i]
                        j = lsb.bit_length() - 1
                        best_clique = [i, j]
                        found = True
                        break
                if not found:
                    break
            
            # Remove edges of best_clique
            for i in range(len(best_clique)):
                u = best_clique[i]
                mask = 0
                for j in range(len(best_clique)):
                    if i != j:
                        mask |= (1 << best_clique[j])
                remaining_adj[u] &= ~mask
                
            cliques.append(best_clique)
            
        return cliques

    res1 = instance(32, 0.6, 601)
    res2 = instance(64, 0.55, 602)
    
    return [res1, res2]
