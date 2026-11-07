import time
import random

_DEADLINE = 140.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for C (0..k-1) and cross edges
        # adj_c[u] is a bitmask of neighbors in C
        # adj_i[v] is a bitmask of neighbors in C for v in I (k..n-1)
        
        adj_c = [0] * k
        adj_i = [0] * (n - k)
        
        for (u, v) in edges:
            if u < k and v < k:
                # Clique edge
                adj_c[u] |= (1 << v)
                adj_c[v] |= (1 << u)
            elif u < k and v >= k:
                # Cross edge
                # v in I is index v-k
                idx_v = v - k
                adj_i[idx_v] |= (1 << u)
            elif u >= k and v < k:
                # Cross edge
                idx_v = u - k
                adj_i[idx_v] |= (1 << v)
            else:
                # Independent set edge? Should not happen in split graph definition 
                # but just in case, ignore or handle? 
                # The problem states "independent set on the rest", so no edges within I.
                pass

        # Uncovered clique edges in C
        # Represented as a list of bitmasks? Or a set of pairs?
        # For max clique finding, we need to check if a set S is a clique.
        # Let's maintain `uncovered_adj` as a list of bitmasks for C.
        # uncovered_adj[u] has bit v set if edge (u,v) is uncovered.
        
        uncovered_adj = [0] * k
        for u in range(k):
            # Initially all edges in C are uncovered
            uncovered_adj[u] = adj_c[u]

        cliques = []

        # Helper: Check if a set of vertices (bitmask) forms a clique in uncovered_adj
        def is_clique(mask):
            # For each vertex in mask, its neighbors in mask must be exactly mask minus itself
            # Or simply: for all u, v in mask, edge (u,v) exists in uncovered_adj
            # Efficient check:
            # Iterate through set bits
            temp = mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                temp ^= lsb
                # Check if all other vertices in mask are neighbors of u in uncovered_adj
                # (uncovered_adj[u] & (mask ^ lsb)) == (mask ^ lsb)
                if (uncovered_adj[u] & (mask ^ lsb)) != (mask ^ lsb):
                    return False
            return True

        # Helper: Find a maximal clique in the induced subgraph of `candidates` (bitmask)
        # using uncovered_adj
        def find_max_clique(candidates):
            # Greedy:
            # 1. Order vertices in candidates by degree in the induced subgraph (descending)
            # 2. Pick the first, add to clique
            # 3. Remove all non-neighbors from candidates
            # 4. Repeat
            
            # Get list of vertices in candidates
            verts = []
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                verts.append(v)
                temp ^= lsb
            
            if not verts:
                return 0

            # Calculate degrees within candidates
            # deg[v] = popcount(uncovered_adj[v] & candidates)
            # Sort by degree
            verts.sort(key=lambda v: (uncovered_adj[v] & candidates).bit_count(), reverse=True)
            
            clique_mask = 0
            cand_mask = candidates
            
            while cand_mask:
                # Pick vertex with max degree in cand_mask
                # We already sorted initial list, but cand_mask changes.
                # For efficiency with small k, re-sorting or finding max on the fly is fine.
                best_v = -1
                best_deg = -1
                temp = cand_mask
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    deg = (uncovered_adj[v] & cand_mask).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                
                if best_v == -1:
                    break
                
                clique_mask |= (1 << best_v)
                
                # Update cand_mask: keep only neighbors of best_v in cand_mask
                # Also remove best_v itself
                cand_mask = cand_mask & uncovered_adj[best_v]
                cand_mask &= ~(1 << best_v)

            return clique_mask

        # Process independent vertices
        # Sort by degree (popcount of adj_i) descending
        i_indices = list(range(n - k))
        i_indices.sort(key=lambda idx: adj_i[idx].bit_count(), reverse=True)

        for idx in i_indices:
            if _deadline_hit():
                # Fallback: cover remaining cross edges with 2-cliques
                # and remaining clique edges with 2-cliques
                # This is inefficient but valid
                pass
            
            # We need to cover all cross edges for this vertex v (index k+idx)
            # Its neighbors in C are given by adj_i[idx]
            remaining_neighbors = adj_i[idx]
            
            while remaining_neighbors:
                if _deadline_hit():
                    # Fallback
                    # Just add 2-cliques for remaining cross edges
                    temp = remaining_neighbors
                    while temp:
                        lsb = temp & -temp
                        u = lsb.bit_length() - 1
                        temp ^= lsb
                        cliques.append([k + idx, u])
                        # Remove edge (k+idx, u) - cross edges are unique to v, so no conflict with other v
                        # But we need to ensure we don't double cover. 
                        # Since we are in fallback, we assume we just dump 2-cliques.
                    break

                # Find a clique S in remaining_neighbors using uncovered_adj
                # S must be a clique in the uncovered graph of C
                S_mask = find_max_clique(remaining_neighbors)
                
                if S_mask == 0:
                    # No edge in remaining_neighbors is uncovered?
                    # This means all edges between pairs in remaining_neighbors are already covered.
                    # We must pick single vertices?
                    # If S_mask is 0, it means no vertex in remaining_neighbors has any uncovered edge to another in remaining_neighbors.
                    # So we can only pick single vertices?
                    # A clique of size 1 is not allowed (must be >= 2).
                    # But {v, u} is a clique of size 2.
                    # So we pick one vertex u from remaining_neighbors.
                    lsb = remaining_neighbors & -remaining_neighbors
                    u = lsb.bit_length() - 1
                    S_mask = lsb
                else:
                    # S_mask is a clique of size >= 2?
                    # find_max_clique returns a maximal clique.
                    # If the graph is empty, it might return 0.
                    # If it returns a mask with 1 bit, that's a single vertex.
                    # We need clique size >= 2 for the final list.
                    # If S_mask has only 1 bit, we can still form {v, u}.
                    pass

                # Form clique {v} U S
                clique = [k + idx]
                temp = S_mask
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    temp ^= lsb
                    clique.append(u)
                
                cliques.append(clique)

                # Remove covered edges from uncovered_adj
                # Edges within S_mask
                temp = S_mask
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    temp ^= lsb
                    # Remove edges from u to all other vertices in S_mask
                    neighbors_in_S = S_mask ^ lsb
                    uncovered_adj[u] &= ~neighbors_in_S
                    # Also update the other side? 
                    # uncovered_adj is symmetric. We only read from it.
                    # But find_max_clique uses it.
                    # We must update all involved vertices.
                    # For efficiency, we can just update u's row.
                    # But when checking is_clique or find_max_clique, we check (uncovered_adj[v] & mask).
                    # If we only update u, then when checking v, we might see the edge as still there.
                    # So we MUST update both sides.
                    
                # Better: Iterate over all pairs in S_mask and clear bits in both directions
                verts_in_S = []
                temp = S_mask
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    verts_in_S.append(u)
                    temp ^= lsb
                
                for i in range(len(verts_in_S)):
                    u = verts_in_S[i]
                    for j in range(i + 1, len(verts_in_S)):
                        w = verts_in_S[j]
                        uncovered_adj[u] &= ~(1 << w)
                        uncovered_adj[w] &= ~(1 << u)

                # Remove S from remaining_neighbors
                remaining_neighbors &= ~S_mask

        # Process remaining uncovered edges in C
        # Find cliques in the graph defined by uncovered_adj
        # We need to cover all remaining edges.
        # Greedy: while there are uncovered edges:
        #   Find max clique in the current uncovered graph
        #   Add it to cliques
        #   Remove its edges from uncovered_adj
        
        # To find max clique in the whole graph C, we can use find_max_clique with candidates = (1<<k)-1
        # But find_max_clique is greedy and might get stuck in small cliques if the graph is sparse.
        # However, it picks max degree first.
        
        # Optimization: We can just call find_max_clique on the full mask repeatedly.
        # But find_max_clique modifies nothing, it just finds a clique.
        # We need to ensure we make progress.
        
        full_mask = (1 << k) - 1
        
        while True:
            if _deadline_hit():
                # Fallback: 2-cliques for all remaining edges
                # Iterate over all pairs
                for u in range(k):
                    for w in range(u + 1, k):
                        if (uncovered_adj[u] >> w) & 1:
                            cliques.append([u, w])
                            uncovered_adj[u] &= ~(1 << w)
                            uncovered_adj[w] &= ~(1 << u)
                break
            
            # Check if any edge remains
            has_edge = False
            for u in range(k):
                if uncovered_adj[u]:
                    has_edge = True
                    break
            
            if not has_edge:
                break
            
            # Find a clique
            # To avoid picking a single vertex (which covers no edges), we need a clique of size >= 2.
            # find_max_clique might return a single vertex if no edges exist for that vertex?
            # No, it picks max degree. If max degree is 0, it picks a vertex with 0 neighbors.
            # Then cand_mask becomes 0. Clique size 1.
            # If we add a clique of size 1, it's invalid.
            # So we need to find a clique of size >= 2.
            
            # Modify find_max_clique to prefer size >= 2?
            # Or just check if the returned mask has bit_count >= 2.
            # If not, we need to handle it.
            # If no clique of size >= 2 exists, it means the graph is a matching?
            # No, if there is an edge, there is a clique of size 2.
            # So if has_edge is True, there must be a clique of size >= 2.
            # Why would find_max_clique return size 1?
            # If the graph is a star, max degree is high. It picks center.
            # Neighbors are leaves. Leaves are not connected to each other.
            # So cand_mask becomes leaves.
            # Next iteration, picks a leaf. Degree 0 in cand_mask (leaves).
            # So it adds leaf.
            # Clique = {center, leaf}. Size 2.
            # So it should work.
            
            S_mask = find_max_clique(full_mask)
            
            if S_mask.bit_count() < 2:
                # Fallback: pick any edge
                # Find u with uncovered_adj[u] != 0
                u = 0
                while u < k and not uncovered_adj[u]:
                    u += 1
                if u == k: break
                # Find w in uncovered_adj[u]
                lsb = uncovered_adj[u] & -uncovered_adj[u]
                w = lsb.bit_length() - 1
                S_mask = (1 << u) | (1 << w)
            
            # Add clique
            clique = []
            temp = S_mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                temp ^= lsb
                clique.append(u)
            
            cliques.append(clique)
            
            # Remove edges
            verts_in_S = []
            temp = S_mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                verts_in_S.append(u)
                temp ^= lsb
            
            for i in range(len(verts_in_S)):
                u = verts_in_S[i]
                for j in range(i + 1, len(verts_in_S)):
                    w = verts_in_S[j]
                    uncovered_adj[u] &= ~(1 << w)
                    uncovered_adj[w] &= ~(1 << u)

        # Convert to lists of ints
        # The cliques are already lists of ints.
        # Ensure types are int.
        result = []
        for c in cliques:
            result.append([int(x) for x in c])
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            # Fallback: seed-like (each edge as 2-clique)
            out.append([list(e) for e in instance(*spec)])
        else:
            out.append(_solve_split(*spec))
    return out
