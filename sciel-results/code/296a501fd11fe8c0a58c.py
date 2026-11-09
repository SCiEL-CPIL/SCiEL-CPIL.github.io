"""Biclique partition via bitmask greedy with geometric-mean scoring."""
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import math

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmask for fast intersection
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Track remaining edges as a set of frozenset pairs
    remaining_edges = set()
    for u, v in edges:
        remaining_edges.add((min(u,v), max(u,v)))
    
    def popcount(x):
        return bin(x).count('1')
    
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        return verts
    
    bicliques = []
    
    while remaining_edges:
        if time.time() > deadline - 1.0:
            break
        
        # Find the edge with the most potential to form a large biclique
        # Strategy: for each remaining edge (u,v), compute common neighbors
        # that also have all edges to each other (within remaining graph)
        
        best_score = -1
        best_left = None
        best_right = None
        best_edges = None
        
        # Sample a subset of edges for speed if too many
        edge_list = list(remaining_edges)
        if len(edge_list) > 500:
            import random
            random.seed(42)
            edge_list = random.sample(edge_list, 500)
        
        for (u, v) in edge_list:
            # Common neighbors of u and v in the remaining graph
            # A neighbor w is "remaining-adjacent" to u if (min(u,w),max(u,w)) in remaining_edges
            common = adj[u] & adj[v]
            # Remove u and v themselves
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            # Filter to only vertices that still have edges to u and v in remaining
            # Actually adj includes ALL edges, not just remaining. Need to be careful.
            # For a valid biclique [L, R], every pair (l, r) for l in L, r in R must be an edge in the ORIGINAL graph.
            # The constraint is: cross pairs must be host edges (original graph edges).
            # So we can use the full adjacency matrix for validity.
            # But we need to cover remaining edges.
            
            # Let's think differently:
            # Start with seed edge (u, v). 
            # L = {u}, R = {v} initially, or we can try L={u}, R=common neighbors of u that connect to v
            # Actually for a biclique [L, R]: every l in L must connect to every r in R.
            # So R must be subset of N(u) ∩ N(v) ∩ ... for all l in L
            # And L must be subset of N(v) ∩ N(u) ∩ ... for all r in R
            
            # Greedy expansion:
            # Start: L = {u}, R = {v}
            # Then try to add more to R: candidates = N(u) ∩ (remaining neighbors of u)
            # But also each candidate r must connect to all current L.
            
            # Simpler approach: for seed (u,v), compute:
            # R_candidates = {w : w != u, (u,w) in remaining, and for all l in L, (l,w) is original edge}
            # L_candidates = {w : w != v, (w,v) in remaining, and for all r in R, (w,r) is original edge}
            
            # Let's do iterative expansion:
            L_mask = (1 << u)
            R_mask = (1 << v)
            
            # Expand R: add all w such that (u,w) is a remaining edge and w connects to all in L
            # For L={u}, R candidates = remaining neighbors of u that also connect to v (they do since we're building)
            # Actually R candidates must be: w where (u,w) in remaining AND for all l in L, (l,w) is original edge
            # Since L={u}, just need (u,w) in remaining.
            
            # But wait - we also need the biclique to only cover remaining edges.
            # The cross pairs (l,r) for l in L, r in R must ALL be remaining edges? No!
            # The task says "cross pair must be a host edge" (original edge) and "cross-edge sets must partition all pinned graph edges exactly once."
            # So cross pairs must be original edges, and each original edge is covered exactly once.
            # The biclique's cross edges are { (l,r) : l in L, r in R } and these must all be original edges.
            # Additionally, these cross edges must be a subset of the remaining edges (to partition).
            
            # So: for a biclique [L,R], every (l,r) pair must be BOTH an original edge AND a remaining edge.
            # This means R must be subset of {w : (u,w) in remaining} for all u in L, AND (u,w) is original edge.
            # Since remaining ⊆ original, if (u,w) in remaining then it's automatically an original edge.
            # So we just need: for all l in L, r in R: (l,r) in remaining_edges.
            
            # This is stricter! Let me redo.
            
            # R candidates for L={u}: w where (u,w) in remaining_edges
            # L candidates for R={v}: w where (w,v) in remaining_edges
            
            # Iterate to fixpoint:
            changed = True
            while changed:
                changed = False
                # Expand R
                new_R = R_mask
                m = L_mask
                while m:
                    lsb = m & (-m)
                    l = lsb.bit_length() - 1
                    # Remaining neighbors of l
                    rn = 0
                    for (a, b) in remaining_edges:
                        if a == l:
                            rn |= (1 << b)
                        elif b == l:
                            rn |= (1 << a)
                    new_R &= rn
                    m ^= lsb
                if new_R != R_mask:
                    R_mask = new_R
                    changed = True
                
                # Expand L
                new_L = L_mask
                m = R_mask
                while m:
                    lsb = m & (-m)
                    r = lsb.bit_length() - 1
                    rn = 0
                    for (a, b) in remaining_edges:
                        if a == r:
                            rn |= (1 << b)
                        elif b == r:
                            rn |= (1 << a)
                    new_L &= rn
                    m ^= lsb
                if new_L != L_mask:
                    L_mask = new_L
                    changed = True
            
            # Ensure u in L and v in R
            if not (L_mask & (1 << u)) or not (R_mask & (1 << v)):
                continue
            
            L_count = popcount(L_mask)
            R_count = popcount(R_mask)
            edge_count = L_count * R_count
            
            # Score: geometric mean of side sizes, with preference for more edges
            if L_count > 0 and R_count > 0:
                score = math.log(L_count) + math.log(R_count) + 0.1 * math.log(edge_count)
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_left = L_mask
                best_right = R_mask
                best_edges = set()
                lm = L_mask
                while lm:
                    lsb = lm & (-lm)
                    l = lsb.bit_length() - 1
                    rm = R_mask
                    while rm:
                        rsb = rm & (-rm)
                        r = rsb.bit_length() - 1
                        best_edges.add((min(l,r), max(l,r)))
                        rm ^= rsb
                    lm ^= lsb
        
        if best_left is None or best_right is None:
            # Fallback: just take one edge
            (u, v) = next(iter(remaining_edges))
            bicliques.append([u, v])
            remaining_edges.discard((u, v))
            continue
        
        # Verify all cross edges are in remaining
        valid = True
        for e in best_edges:
            if e not in remaining_edges:
                valid = False
                break
        
        if not valid:
            # Fallback: shrink to just the seed edge
            # Find a valid edge
            for (u, v) in edge_list:
                bicliques.append([u, v])
                remaining_edges.discard((u, v))
                break
            continue
        
        # Convert to vertex lists
        left_verts = get_vertices(best_left)
        right_verts = get_vertices(best_right)
        
        # Ensure nonempty and disjoint
        if not left_verts or not right_verts:
            (u, v) = next(iter(remaining_edges))
            bicliques.append([u, v])
            remaining_edges.discard((u, v))
            continue
        
        # Check disjointness
        if set(left_verts) & set(right_verts):
            # Shouldn't happen with proper construction, but fallback
            (u, v) = next(iter(remaining_edges))
            bicliques.append([u, v])
            remaining_edges.discard((u, v))
            continue
        
        bicliques.append([left_verts, right_verts])
        
        # Remove covered edges
        for e in best_edges:
            remaining_edges.discard(e)
    
    # Handle any leftover edges (shouldn't happen if logic is correct)
    while remaining_edges:
        if time.time() > deadline - 0.5:
            break
        (u, v) = next(iter(remaining_edges))
        bicliques.append([u, v])
        remaining_edges.discard((u, v))
    
    # Convert to proper types
    result = []
    for bc in bicliques:
        if isinstance(bc, list) and len(bc) == 2:
            l, r = bc
            if isinstance(l, int) and isinstance(r, int):
                result.append([[int(l)], [int(r)]])
            else:
                result.append([[int(x) for x in l], [int(x) for x in r]])
        else:
            result.append(bc)
    
    return result
