"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    _deadline = time.time() + 145.0  # Safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        # Generate edges using the required instance() function
        edges = instance(n, k, p, seed)
        
        # Separate edges
        k_edges = []
        cross = []
        for u, v in edges:
            if u < k and v < k:
                k_edges.append((u, v))
            else:
                cross.append((u, v))
        
        # Build adjacency for K-vertices
        rem_adj = [0] * k
        for u, v in k_edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        # Helper to find a maximal clique in a candidate set using rem_adj
        def find_maximal_clique(candidates_mask):
            clique = []
            current_mask = candidates_mask
            
            while current_mask:
                # Pick vertex with max degree in current induced subgraph
                best_v = -1
                best_deg = -1
                m = current_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    deg = (current_mask & rem_adj[v]).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    m ^= lsb
                
                if best_v == -1:
                    break
                    
                clique.append(best_v)
                current_mask &= rem_adj[best_v]
                
            return clique

        result = []
        
        # Process cross edges: for each v in I, form cliques {v} U S where S is clique in K
        neighbors_of_v = {v: [] for v in range(k, n)}
        for u, v in cross:
            neighbors_of_v[v].append(u)
        
        # Sort I-vertices by neighborhood size (larger first) to form bigger cliques
        i_vertices = [v for v in range(k, n) if neighbors_of_v[v]]
        i_vertices.sort(key=lambda v: -len(neighbors_of_v[v]))
        
        for v in i_vertices:
            if not neighbors_of_v[v]:
                continue
            
            candidates_mask = 0
            for u in neighbors_of_v[v]:
                candidates_mask |= (1 << u)
            
            while candidates_mask:
                if time.time() > _deadline:
                    # Fallback: add remaining as pairs
                    for u in neighbors_of_v[v]:
                        if (1 << u) & candidates_mask:
                            result.append([v, u])
                    break
                
                S = find_maximal_clique(candidates_mask)
                if not S:
                    break
                    
                clique_vertices = [v] + S
                result.append(clique_vertices)
                
                # Consume internal edges of S
                for i in range(len(S)):
                    u = S[i]
                    for j in range(i + 1, len(S)):
                        w = S[j]
                        rem_adj[u] &= ~(1 << w)
                        rem_adj[w] &= ~(1 << u)
                
                for u in S:
                    candidates_mask &= ~(1 << u)
        
        # Cover remaining K edges
        while True:
            if time.time() > _deadline:
                # Fallback: add all remaining edges as pairs
                for u in range(k):
                    for v in range(u + 1, k):
                        if (rem_adj[u] >> v) & 1:
                            result.append([u, v])
                            rem_adj[u] &= ~(1 << v)
                            rem_adj[v] &= ~(1 << u)
                break
            
            active_mask = 0
            for u in range(k):
                if rem_adj[u]:
                    active_mask |= (1 << u)
            
            if active_mask == 0:
                break
            
            # Pick an edge (u, v) from active graph
            u = -1
            m = active_mask
            while m:
                lsb = m & -m
                candidate = lsb.bit_length() - 1
                if rem_adj[candidate]:
                    u = candidate
                    break
                m ^= lsb
            
            if u == -1:
                break
                
            v_mask = rem_adj[u] & active_mask
            if not v_mask:
                break
            
            lsb_v = v_mask & -v_mask
            v = lsb_v.bit_length() - 1
            
            common_neighbors = rem_adj[u] & rem_adj[v]
            S = find_maximal_clique(common_neighbors)
            
            clique_vertices = [u, v] + S
            result.append(clique_vertices)
            
            # Remove edges in this clique
            rem_adj[u] &= ~(1 << v)
            rem_adj[v] &= ~(1 << u)
            
            for s in S:
                rem_adj[u] &= ~(1 << s)
                rem_adj[s] &= ~(1 << u)
                rem_adj[v] &= ~(1 << s)
                rem_adj[s] &= ~(1 << v)
                for s2 in S:
                    if s < s2:
                        rem_adj[s] &= ~(1 << s2)
                        rem_adj[s2] &= ~(1 << s)
        
        # Validate and fix: ensure all cliques have at least 2 vertices
        # and all edges are covered exactly once
        final_result = []
        for clique in result:
            if len(clique) >= 2:
                # Ensure distinct vertices
                unique_clique = list(set(clique))
                if len(unique_clique) >= 2:
                    final_result.append(unique_clique)
        
        return final_result

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _deadline - 5:
            break
        results.append(solve_one(*spec))
    
    return results
# EVOLVE-BLOCK-END
