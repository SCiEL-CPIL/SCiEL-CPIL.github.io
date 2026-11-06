import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 115  # Safety margin before 120s kill

    def time_left():
        return deadline - (time.time() - start_time)

    # Precompute masks for vertices
    # adj_mask[i] is an integer where bit j is 1 if i and j are connected
    def check_c6_free_mask(adj_mask, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        Returns True if safe (no C6 created).
        A C6 is u-a-b-c-d-v.
        """
        # If u and v are already connected, this check is irrelevant for adding, but we assume they aren't.
        # We need to ensure no path of length 5 exists between u and v in the current graph.
        # Path: u - a - b - c - d - v
        
        # Neighbors of u (excluding v, though v isn't connected yet, so just N(u))
        # Actually, we are checking if adding (u,v) creates a cycle.
        # The cycle would be u-v-d-c-b-a-u.
        # This is equivalent to checking if there is a path of length 5 from u to v in the graph WITHOUT the edge (u,v).
        # Since (u,v) is not in the graph yet, we just check for a path of length 5 from u to v in the current graph.
        
        # Let's trace paths of length 5 from u to v.
        # u -> a -> b -> c -> d -> v
        
        # Optimization: 
        # 1. Get neighbors of u: A = N(u)
        # 2. For each a in A:
        #    2.1 Get neighbors of a: B_a = N(a) \ {u, v} (v is not neighbor, u is)
        #    2.2 For each b in B_a:
        #        2.2.1 Get neighbors of b: C_b = N(b) \ {u, v, a}
        #        2.2.2 For each c in C_b:
        #             2.2.2.1 Get neighbors of c: D_c = N(c) \ {u, v, a, b}
        #             2.2.2.2 If D_c intersects N(v), then we found d such that d is neighbor of c and v.
        #                     This forms u-a-b-c-d-v.
        
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # Iterate neighbors of u
        # To iterate bits in Python efficiently:
        # while m: lsb = m & -m; idx = lsb.bit_length()-1; m ^= lsb
        
        m_a = mask_u
        while m_a:
            lsb_a = m_a & -m_a
            a = lsb_a.bit_length() - 1
            m_a ^= lsb_a
            
            if a == v: continue # Should not happen if edge not present, but safe
            
            m_b = adj_mask[a] & ~mask_u & ~mask_v # Exclude u and v
            # Note: mask_v might not have bit a set, but we exclude it to be safe for path simplicity
            # Actually, b cannot be v because v is not connected to u yet? No, b is neighbor of a.
            # b cannot be u.
            # b cannot be v? If b==v, path is u-a-v... length 2. We need length 5.
            # So b != v is correct.
            
            while m_b:
                lsb_b = m_b & -m_b
                b = lsb_b.bit_length() - 1
                m_b ^= lsb_b
                
                m_c = adj_mask[b] & ~mask_u & ~mask_v & ~mask_a # Exclude u, v, a
                # c cannot be u, v, a.
                
                while m_c:
                    lsb_c = m_c & -m_c
                    c = lsb_c.bit_length() - 1
                    m_c ^= lsb_c
                    
                    # Check if any neighbor of c (excluding u, v, a, b) is a neighbor of v
                    # D_c = N(c) \ {u, v, a, b}
                    # We need D_c & N(v) != 0
                    
                    mask_d = adj_mask[c] & ~mask_u & ~mask_v & ~mask_a & ~mask_b
                    if mask_d & mask_v:
                        return False # C6 found
            
        return True

    def count_edges_mask(adj_mask):
        return sum(bin(m).count('1') for m in adj_mask) // 2

    def greedy_add_edges_mask(adj_mask, rng):
        """Try to add all possible missing edges in random order."""
        added_any = True
        while added_any and time_left() > 1:
            added_any = False
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj_mask[u] >> v) & 1:
                        candidates.append((u, v))
            rng.shuffle(candidates)
            
            for u, v in candidates:
                if time_left() < 1:
                    return adj_mask
                if not (adj_mask[u] >> v) & 1:
                    if check_c6_free_mask(adj_mask, u, v):
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        added_any = True
        return adj_mask

    def local_search_mask(adj_mask, rng):
        """
        Simple local search: Try to swap edges or add edges if possible.
        Given the slowness of check_c6_free, we limit this.
        """
        improved = True
        while improved and time_left() > 2:
            improved = False
            # Try adding edges
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj_mask[u] >> v) & 1:
                        candidates.append((u, v))
            rng.shuffle(candidates)
            
            for u, v in candidates:
                if time_left() < 1:
                    return adj_mask
                if not (adj_mask[u] >> v) & 1:
                    if check_c6_free_mask(adj_mask, u, v):
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        improved = True
        return adj_mask

    best_edge_count = -1
    best_adj_mask = None

    # Fewer restarts, but faster checks
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time_left() < 5:
            break

        rng = random.Random(restart * 1000 + 42)

        # Start with empty graph
        adj_mask = [0] * n
        
        # Optional: Start with a random structure to help greedy
        # Let's just rely on greedy from empty, it's fast enough with bitmasks
        
        adj_mask = greedy_add_edges_mask(adj_mask, rng)
        adj_mask = local_search_mask(adj_mask, rng)

        ec = count_edges_mask(adj_mask)
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj_mask = list(adj_mask)

        if best_edge_count >= 105:
            break

    if best_adj_mask is None:
        # Fallback: 8 disjoint K5s (which are C6-free)
        # K5 has 10 edges. 8 * 10 = 80 edges.
        best_adj_mask = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    best_adj_mask[u] |= (1 << v)
                    best_adj_mask[v] |= (1 << u)

    edges = []
    for u in range(n):
        m = best_adj_mask[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            m ^= lsb
            
    return edges
