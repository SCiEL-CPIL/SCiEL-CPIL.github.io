"""Dense Q3-free graph on 16 vertices via complete multipartite initialization."""
import time
import random

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Q3 structure: 8 vertices, 12 edges.
    # Vertices are 3-bit strings. Edges connect strings differing by 1 bit.
    # We need to check if adding edge (u,v) creates a subgraph isomorphic to Q3.
    
    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3.
        adj[i] is a bitmask of neighbors of i.
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        # In Q3, u and v are connected.
        # Let's try to find the other 6 vertices.
        # The neighbors of u in Q3 are 3 vertices. One is v.
        # So u has 2 other neighbors in Q3, let's call them a2, a4.
        # These must be neighbors of u in the graph (since edge exists in Q3).
        # Also, in Q3, neighbors of u are not connected to each other.
        # But we are looking for a subgraph, so extra edges are allowed.
        # However, the structure must be preserved.
        
        # Let's iterate over potential pairs of neighbors (a2, a4) of u (excluding v).
        # Note: a2 and a4 must be distinct.
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                # Now we have u, v, a2, a4.
                # In Q3, v is connected to u.
                # v is also connected to 2 other vertices in Q3 (besides u).
                # Let's call them a3, a5.
                # In Q3, a3 is connected to a2 and a4? No.
                # Let's trace the cube.
                # Let u = 000, v = 001.
                # Neighbors of u: 001(v), 010(a2), 100(a4).
                # Neighbors of v: 000(u), 011(a3), 101(a5).
                # So a3 is connected to a2 (010) and v (001)? 
                # 010 and 011 differ by 1 bit. Yes.
                # a5 is connected to a4 (100) and v (001)?
                # 100 and 101 differ by 1 bit. Yes.
                # So a3 must be a neighbor of a2 and v.
                # a5 must be a neighbor of a4 and v.
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                # Exclude already used vertices
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << v) & ~(1 << u)
                a5_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << v) & ~(1 << u)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        # Used: u, v, a2, a3, a4, a5
                        # Remaining: a6, a7
                        # In Q3:
                        # a6 is connected to a2, a4, a3?
                        # Let's check 000(u), 001(v), 010(a2), 100(a4), 011(a3), 101(a5).
                        # Remaining vertices: 110(a6), 111(a7).
                        # a6 (110) is connected to a2 (010), a4 (100), a7 (111).
                        # a7 (111) is connected to a3 (011), a5 (101), a6 (110).
                        
                        # So a6 must be neighbor of a2 and a4.
                        # a7 must be neighbor of a3 and a5.
                        # And a6 must be neighbor of a7.
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            # Check if a6 is connected to a7
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    def count_edges(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def greedy_add(adj, edge_order):
        added = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def local_search(adj, max_iters=200):
        """
        Local search: try to add edges. If stuck, try swaps.
        """
        current_count = count_edges(adj)
        best_adj = adj[:]
        best_count = current_count
        
        # Phase 1: Greedy passes
        for pass_i in range(5):
            if time.time() > deadline - 5:
                break
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
            if not missing:
                break
            random.shuffle(missing)
            greedy_add(adj, missing)
            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
        
        # Phase 2: Swaps
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            missing_now = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
            if not missing_now:
                break
            
            # Pick a random missing edge
            u_miss, v_miss = random.choice(missing_now)
            
            # Find existing edges to remove
            # Try removing one edge incident to u_miss or v_miss
            candidates = []
            for neighbor in range(n):
                if neighbor != u_miss and neighbor != v_miss:
                    if adj[u_miss] & (1 << neighbor):
                        candidates.append((u_miss, neighbor))
                    if adj[v_miss] & (1 << neighbor):
                        candidates.append((v_miss, neighbor))
            
            if not candidates:
                continue
            
            u_rem, v_rem = random.choice(candidates)
            
            trial = best_adj[:]
            trial[u_rem] &= ~(1 << v_rem)
            trial[v_rem] &= ~(1 << u_rem)
            
            # Try to add the missing edge
            if not creates_q3_fast(u_miss, v_miss, trial):
                trial[u_miss] |= (1 << v_miss)
                trial[v_miss] |= (1 << u_miss)
            
            # Greedy fill
            missing_trial = [(u, v) for u in range(n) for v in range(u+1, n) if not (trial[u] & (1 << v))]
            random.shuffle(missing_trial)
            greedy_add(trial, missing_trial)
            
            new_count = count_edges(trial)
            if new_count > best_count:
                best_count = new_count
                best_adj = trial[:]
                adj = trial[:]
        
        return best_adj, best_count

    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    # Initialize with K_{4,4,4,4}
    # Parts: [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]
    parts = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    
    adj = [0] * n
    for i in range(4):
        for j in range(i + 1, 4):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    # Initial count: 4 parts of size 4. Edges between parts: 4 * C(4,2) * 4*4?
    # No. Pairs of parts: C(4,2)=6. Edges per pair: 4*4=16. Total 96.
    # K16 has 120 edges. So we have 96. Missing 24 (intra-part).
    
    # Run local search
    adj, cnt = local_search(adj)
    if cnt > best_count:
        best_count = cnt
        best_adj = adj
    
    # Try a few random restarts with slightly different initializations
    # e.g. remove one random inter-part edge and see if we can gain more intra-part edges
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        adj_restart = best_adj[:]
        # Remove a random edge
        existing = []
        for u in range(n):
            mask = adj_restart[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    existing.append((u, v))
                mask -= bit
        
        if not existing:
            break
            
        u_rem, v_rem = random.choice(existing)
        adj_restart[u_rem] &= ~(1 << v_rem)
        adj_restart[v_rem] &= ~(1 << u_rem)
        
        adj_restart, cnt = local_search(adj_restart)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj_restart

    # Final polish
    if time.time() < deadline - 2:
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                     if not (best_adj[u] & (1 << v))]
        random.shuffle(remaining)
        greedy_add(best_adj, remaining)
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)
