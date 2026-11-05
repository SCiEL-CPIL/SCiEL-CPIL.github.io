import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit)
    
    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    
    best_count = 0
    best_adj = [[False] * n for _ in range(n)]
    
    # Q3 structure
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    
    # Precompute for each Q3 edge (i,j) and orientation, the constraints for remaining vertices
    # For a given assignment of Q3 vertices to graph vertices, we need to check all Q3 edges exist
    
    def creates_q3_fast(u, v, adj_bits):
        """Check if adding edge (u,v) to graph with adjacency bitmasks creates a Q3."""
        # adj_bits[i] is a bitmask of neighbors of vertex i
        # u and v are not yet connected in adj_bits
        
        # Try all 24 orientations (12 edges * 2 directions)
        for i, j in q3_edges:
            # Case 1: u->i, v->j
            if _check_embedding_fast(i, j, u, v, adj_bits, q3_adj):
                return True
            # Case 2: u->j, v->i
            if _check_embedding_fast(j, i, u, v, adj_bits, q3_adj):
                return True
        return False
    
    def _check_embedding_fast(q3_u, q3_v, g_u, g_v, adj_bits, q3_adj):
        """
        Check if there's an embedding of Q3 where q3_u->g_u, q3_v->g_v,
        and the edge (g_u, g_v) is the new edge being added.
        adj_bits does NOT include edge (g_u, g_v) yet.
        """
        # Mapped Q3 vertices to graph vertices
        mapping = {q3_u: g_u, q3_v: g_v}
        used_g = {g_u, g_v}
        remaining_q3 = [x for x in range(8) if x not in (q3_u, q3_v)]
        
        # For backtracking, we need to track which Q3 vertices are mapped
        # and use adjacency bitmasks for fast edge checks
        
        def backtrack(idx):
            if idx == len(remaining_q3):
                # All mapped. Verify all Q3 edges exist in G (including the new edge).
                for a, b in q3_edges:
                    ga = mapping[a]
                    gb = mapping[b]
                    if ga == g_u and gb == g_v:
                        # This is the new edge, it's being added
                        continue
                    if ga == g_v and gb == g_u:
                        continue
                    # Check if edge exists in adj_bits
                    if not (adj_bits[ga] & (1 << gb)):
                        return False
                return True
            
            q3_vertex = remaining_q3[idx]
            
            # Find candidates: graph vertices not yet used
            # and adjacent to all already-mapped Q3 neighbors of q3_vertex
            candidates = []
            for g_cand in range(n):
                if g_cand in used_g:
                    continue
                # Check adjacency to all mapped Q3 neighbors
                ok = True
                for q3_neighbor in q3_adj[q3_vertex]:
                    if q3_neighbor in mapping:
                        g_neighbor = mapping[q3_neighbor]
                        if g_neighbor == g_u or g_neighbor == g_v:
                            # This might be the new edge case
                            # The new edge is (g_u, g_v), so if q3_neighbor maps to g_u or g_v
                            # and q3_vertex is adjacent to q3_neighbor in Q3,
                            # then g_cand must be adjacent to g_neighbor in G.
                            # But g_u-g_v is the new edge, not yet in adj_bits.
                            # So if g_cand == g_u and g_neighbor == g_v, that's the new edge.
                            # Actually, g_cand is a candidate for q3_vertex, and g_neighbor is for q3_neighbor.
                            # If q3_vertex and q3_neighbor are adjacent in Q3, then in G,
                            # mapping[q3_vertex] and mapping[q3_neighbor] must be adjacent.
                            # mapping[q3_vertex] = g_cand, mapping[q3_neighbor] = g_neighbor.
                            # So we need edge (g_cand, g_neighbor) in G.
                            # If g_cand == g_u and g_neighbor == g_v, or vice versa, that's the new edge.
                            if (g_cand == g_u and g_neighbor == g_v) or (g_cand == g_v and g_neighbor == g_u):
                                # This is the new edge being added, so it's OK
                                continue
                        if not (adj_bits[g_cand] & (1 << g_neighbor)):
                            ok = False
                            break
                if ok:
                    candidates.append(g_cand)
            
            if not candidates:
                return False
            
            for g_cand in candidates:
                mapping[q3_vertex] = g_cand
                used_g.add(g_cand)
                
                if backtrack(idx + 1):
                    return True
                
                del mapping[q3_vertex]
                used_g.remove(g_cand)
            
            return False
        
        return backtrack(0)
    
    # Run multiple random trials
    trials = 20
    for t in range(trials):
        if time.time() > deadline:
            break
        
        random.seed(t * 7919 + 42)
        
        # Use bitmask adjacency for speed
        adj_bits = [0] * n
        
        # Start with K3,13
        for u in range(3):
            for v in range(3, 16):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        
        # Build list of remaining edges
        existing = set()
        for u in range(3):
            for v in range(3, 16):
                existing.add((u, v))
        
        remaining_edges = [e for e in all_edges if e not in existing]
        random.shuffle(remaining_edges)
        
        for u, v in remaining_edges:
            if time.time() > deadline:
                break
            if not creates_q3_fast(u, v, adj_bits):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        
        # Count edges
        count = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
        
        if count > best_count:
            best_count = count
            # Convert to boolean matrix for storage
            best_adj = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if adj_bits[i] & (1 << j):
                        best_adj[i][j] = True
    
    # Convert best_adj to edge list
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result.append([i, j])
    
    return result
