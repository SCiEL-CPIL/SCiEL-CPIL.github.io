import time
import math

def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # Safety margin
    
    # The task is for k=3, g=16.
    # We use the Lubotzky-Phillips-Sarnak (LPS) construction.
    # LPS graphs are 3-regular.
    # We try primes q such that the graph has girth >= 16.
    # N = q(q^2 - 1)
    
    def modinv(a, m):
        return pow(a, m-2, m)

    def lps_graph(q):
        # Generate all matrices in SL(2, F_q)
        # a, b, c, d such that ad - bc = 1
        # We can iterate a, b, c and compute d.
        # If a != 0, d = (1 + bc) * a^(-1)
        # If a == 0, -bc = 1 => bc = -1.
        
        idx = {}
        mats = []
        
        # Precompute inverses
        inv = [0] * q
        for i in range(1, q):
            inv[i] = pow(i, q-2, q)
            
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    if a == 0:
                        # bc = -1 mod q
                        if b == 0:
                            continue
                        # c = -b^(-1)
                        c_val = (-inv[b]) % q
                        if c_val != c:
                            continue
                        d_val = 0 # d is arbitrary? No, ad-bc=1 => 0 - b(-b^-1) = 1. Correct.
                        # Wait, if a=0, d can be anything?
                        # No, the matrix is (0, b, c, d).
                        # Determinant is 0*d - b*c = -bc = 1.
                        # So d is not constrained by the determinant?
                        # Yes, d is free?
                        # No, SL(2) requires det=1.
                        # If a=0, -bc=1. d is free?
                        # Let's check: (0, b, c, d). det = -bc.
                        # So for any d, det is 1.
                        # So there are q matrices for each (b,c) pair with bc=-1?
                        # No, (0, b, c, d) and (0, b, c, d') are different.
                        # So if a=0, we have q values for d.
                        # Let's iterate d as well?
                        # But we are iterating a,b,c.
                        # If a=0, we found a valid (b,c).
                        # Then d can be 0..q-1.
                        for d in range(q):
                            m = (a, b, c, d)
                            idx[m] = len(mats)
                            mats.append(m)
                    else:
                        # a != 0
                        # d = (1 + bc) * a^(-1)
                        d_val = ((1 + b*c) % q * inv[a]) % q
                        m = (a, b, c, d_val)
                        idx[m] = len(mats)
                        mats.append(m)
                        
        n = len(mats)
        # Expected n = q(q^2-1)
        # For q=11, 11*(121-1) = 1320.
        
        # Generators
        # S1 = [[0, -1], [1, 0]]
        # S2 = [[1, 1], [0, 1]]
        # S3 = [[1, 0], [1, 1]]
        
        def mat_mul(M1, M2, q):
            a1, b1, c1, d1 = M1
            a2, b2, c2, d2 = M2
            a = (a1*a2 + b1*c2) % q
            b = (a1*b2 + b1*d2) % q
            c = (c1*a2 + d1*c2) % q
            d = (c1*b2 + d1*d2) % q
            return (a, b, c, d)
            
        S1 = (0, q-1, 1, 0) # [[0, -1], [1, 0]]
        S2 = (1, 1, 0, 1)   # [[1, 1], [0, 1]]
        S3 = (1, 0, 1, 1)   # [[1, 0], [1, 1]]
        
        edges = []
        adj = [[] for _ in range(n)]
        
        for i, m in enumerate(mats):
            for S in [S1, S2, S3]:
                m2 = mat_mul(m, S, q)
                j = idx[m2]
                if i < j:
                    edges.append((i, j))
                adj[i].append(j)
                
        return n, edges, adj

    def verify_girth(n, adj, min_girth):
        max_dist = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= max_dist:
                    continue
                du = dist[u]
                pu = parent[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    dv = dist[v]
                    if dv == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif dv >= du:
                        cycle_len = du + dv + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Try q values
    # q=7 -> 336 vertices. Likely girth < 16.
    # q=11 -> 1320 vertices. Likely girth >= 16.
    # q=13 -> 2184 vertices.
    
    candidates = [11, 13, 17, 19]
    
    for q in candidates:
        if time.time() - t0 > deadline - 10:
            break
            
        n, edges, adj = lps_graph(q)
        
        # Check regularity
        is_reg = all(len(adj[i]) == k for i in range(n))
        if not is_reg:
            continue
            
        # Check girth
        if verify_girth(n, adj, g):
            return n, edges
            
    # Fallback to LUW if LPS fails or is too slow
    # (LUW is slower to generate but we have time)
    # Actually, if LPS fails, we should return the best LPS found or LUW.
    # But LPS should work.
    
    # If we reach here, return a dummy or LUW
    # Let's just return the last valid LPS or LUW
    # For safety, return LUW result from previous logic if needed?
    # No, just return the best LPS.
    
    return 0, [] # Should not happen

