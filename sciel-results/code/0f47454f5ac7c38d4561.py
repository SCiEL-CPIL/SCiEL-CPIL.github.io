import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 110  # Safety margin
    
    def time_check():
        return time.time() - start_time > deadline

    best_edges = 0
    best_adj = None

    # Run multiple greedy attempts with different random seeds
    for seed in range(20): # Try up to 20 restarts
        if time_check():
            break
            
        random.seed(seed)
        
        # Initialize with 8 disjoint K5s
        # Vertices 0-4 in K5_0, 5-9 in K5_1, etc.
        adj = [0] * n
        edges = []
        
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        # Precompute non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        def would_create_c6(u, v):
            """
            Check if adding edge (u,v) creates a C6.
            We need to check if there is a path of length 5 from u to v.
            Path: u - w - z - y - x - v
            w in N(u), x in N(v).
            Need path w-z-y-x of length 3.
            """
            # Get neighbors as lists for iteration, or iterate bits
            # Using bit iteration is faster in Python for sparse-ish graphs, 
            # but here degrees are ~10-20.
            
            # We need to check for existence of w, x, z, y distinct.
            # w in N(u), x in N(v).
            # z in N(w) \ {x}, y in N(z) \ {w, x} such that y in N(x).
            
            # Optimization: 
            # For each w in N(u):
            #   For each x in N(v):
            #     if w == x: continue
            #     # Check if there is a path of length 3 from w to x 
            #     # that does not use u or v (u,v are not in N(w) or N(x) 
            #     # unless they are connected, which they aren't yet, but we are 
            #     # adding the edge, so the path cannot use the new edge).
            #     # Actually, the path w-z-y-x is in the OLD graph.
            #     # So u and v are not involved in the path w-z-y-x.
            #     # We just need simple path w-z-y-x.
            
            #     # Path w-z-y-x exists iff 
            #     # exists z in N(w) \ {x} such that 
            #     # (N(z) \ {w}) & N(x) is not empty.
            #     # Note: y must be in N(z) and N(x).
            #     # y cannot be w (so y in N(z) \ {w}).
            #     # y cannot be x (x not in N(x)).
            #     # y cannot be z (simple graph).
            #     # So y in (N(z) & N(x)) \ {w}.
            
            nu = adj[u]
            nv = adj[v]
            
            # Iterate w in N(u)
            w = nu
            while w:
                # Get lowest set bit
                wb = w & -w
                wi = wb.bit_length() - 1
                w ^= wb
                
                # Iterate x in N(v)
                x = nv
                while x:
                    xb = x & -x
                    xi = xb.bit_length() - 1
                    x ^= xb
                    
                    if wi == xi:
                        continue
                    
                    # Check path w-z-y-x
                    # z in N(wi) \ {xi}
                    nw = adj[wi]
                    # Remove x from consideration for z? No, z can be x? 
                    # If z == x, path is w-x-... no, z is intermediate.
                    # If z == x, then w~x. Path w-x-y-x? No, simple path.
                    # So z != x.
                    nw_minus_x = nw & ~(1 << xi)
                    
                    z = nw_minus_x
                    while z:
                        zb = z & -z
                        zi = zb.bit_length() - 1
                        z ^= zb
                        
                        # y in N(zi) & N(xi) \ {wi}
                        ny = adj[zi] & adj[xi]
                        # Remove w
                        ny_minus_w = ny & ~(1 << wi)
                        
                        if ny_minus_w:
                            return True
            
            return False

        added = 0
        for u, v in non_edges:
            if time_check():
                break
            if not would_create_c6(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                added += 1
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_adj = [list(e) for e in edges]
            
    return best_adj
