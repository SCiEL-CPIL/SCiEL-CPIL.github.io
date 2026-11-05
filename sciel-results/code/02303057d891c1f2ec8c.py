import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    best_edges = 0
    best_adj = None
    
    # Abstract Q3 edges
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    
    # Precompute masks for Q3 structure relative to edge (0,1)
    # Neighbors of 0: 1, 2, 4. (1 is v, so 2,4 are candidates for a,b)
    # Neighbors of 1: 0, 3, 5. (0 is u, so 3,5 are candidates for c,d)
    
    def creates_q3(adj, u, v):
        # adj is list of ints (bitmasks)
        # u, v are the endpoints of the edge being added
        # We check if (u,v) completes a Q3.
        
        # Candidates for a (img 2) and b (img 4) are neighbors of u (excluding v)
        # Candidates for c (img 3) and d (img 5) are neighbors of v (excluding u)
        
        # Note: In the current state (before adding u-v), u and v are NOT connected.
        # So adj[u] does not have bit v, and adj[v] does not have bit u.
        
        mask_u = adj[u]
        mask_v = adj[v]
        
        # Get lists of candidates
        # To avoid list creation overhead, we can iterate over bits.
        # But n=16, so lists are small.
        
        cand_a_b = []
        m = mask_u
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if idx != v:
                cand_a_b.append(idx)
            m ^= lsb
            
        cand_c_d = []
        m = mask_v
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if idx != u:
                cand_c_d.append(idx)
            m ^= lsb
            
        # Iterate over pairs (a, b) from cand_a_b
        # a is img(2), b is img(4)
        for i in range(len(cand_a_b)):
            a = cand_a_b[i]
            for j in range(len(cand_a_b)):
                if i == j: continue
                b = cand_a_b[j]
                
                # a and b must be distinct from u, v (guaranteed by construction)
                # a and b distinct from each other (guaranteed by i!=j)
                
                # Iterate over pairs (c, d) from cand_c_d
                # c is img(3), d is img(5)
                for k in range(len(cand_c_d)):
                    c = cand_c_d[k]
                    if c == a or c == b: continue
                    
                    for l in range(len(cand_c_d)):
                        if k == l: continue
                        d = cand_c_d[l]
                        if d == a or d == b or d == c: continue
                        
                        # Check edges (2,3) and (4,5)
                        # (a,c) must be in adj
                        if not (adj[a] & (1 << c)): continue
                        if not (adj[b] & (1 << d)): continue
                        
                        # Now find f (img 6) and g (img 7)
                        # f must be in N(a) & N(b)
                        # g must be in N(c) & N(d) & N(f)
                        
                        # Candidates for f
                        mask_f = adj[a] & adj[b]
                        
                        # Forbidden set for f: u, v, a, b, c, d
                        forbidden_f = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        mask_f &= ~forbidden_f
                        
                        m_f = mask_f
                        while m_f:
                            lsb = m_f & -m_f
                            f = lsb.bit_length() - 1
                            
                            # Candidates for g
                            # g must be in N(c) & N(d) & N(f)
                            mask_g = adj[c] & adj[d] & adj[f]
                            
                            # Forbidden set for g: u, v, a, b, c, d, f
                            forbidden_g = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << f)
                            mask_g &= ~forbidden_g
                            
                            if mask_g:
                                return True
                            
                            m_f ^= lsb
        
        return False

    # Greedy Algorithm with multiple restarts
    # Use diverse starting points?
    # 1. Random order
    # 2. Start with K_{3,13} structure?
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        adj = [0] * n
        
        # Generate random order of edges
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                edges.append((u, v))
        
        random.shuffle(edges)
        
        # Add edges
        for u, v in edges:
            if time.time() > deadline - 5:
                break
            if not creates_q3(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Count edges
        num_edges = sum(bin(x).count("1") for x in adj) // 2
        
        if num_edges > best_edges:
            best_edges = num_edges
            best_adj = adj[:]
            
        # Optional: Local improvement
        # Try to add edges that were skipped?
        # The greedy order is fixed.
        # We can try a second pass with a different order on the remaining edges.
        
    # Construct the output list of edges
    if best_adj is None:
        # Fallback
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    result_edges = []
    for u in range(n):
        m = best_adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                result_edges.append([u, v])
            m ^= lsb
                
    return result_edges
