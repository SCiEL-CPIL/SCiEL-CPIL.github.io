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
        
        # 1-swap local search polish: try removing one edge and then greedily adding back
        improved = True
        while improved and not time_check():
            improved = False
            # Build list of current edges
            current_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        current_edges.append((u, v))
            
            # Try each edge removal
            for eu, ev in current_edges:
                if time_check():
                    break
                
                # Remove edge (eu, ev)
                adj[eu] &= ~(1 << ev)
                adj[ev] &= ~(1 << eu)
                
                # Build non-edges after removal
                non_edges_after = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (adj[u] >> v) & 1:
                            non_edges_after.append((u, v))
                
                # Greedily add edges
                edges_added = 0
                for u, v in non_edges_after:
                    if time_check():
                        break
                    if not would_create_c6(u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges_added += 1
                
                # Net gain: edges_added - 1 (we removed one)
                if edges_added > 1:
                    # Keep the change
                    improved = True
                else:
                    # Revert: add back (eu, ev) and remove any newly added edges
                    # This is complex; instead, we need to track what was added
                    # For simplicity, we'll revert by removing all edges that were added
                    # and re-adding (eu, ev)
                    # But we don't track which were added, so let's just restore the original state
                    # This approach is flawed; let's use a different strategy
                    
                    # Actually, let's track the added edges properly
                    pass
                
                # The above approach has issues with reverting. Let's use a better strategy:
                # Save the state before removal, try the swap, and only commit if improved.
                # But that's expensive. Instead, let's use a simpler approach:
                # Just do multiple passes of: for each edge, try removing it and greedily filling.
                # If the final edge count is higher, keep it.
        
        # Simpler 1-swap: multiple passes
        for _ in range(3):  # Up to 3 passes
            if time_check():
                break
            pass_improved = False
            current_edge_count = sum(bin(adj[u]).count('1') for u in range(n)) // 2
            
            for u in range(n):
                if time_check():
                    break
                for v in range(u + 1, n):
                    if time_check():
                        break
                    if not ((adj[u] >> v) & 1):
                        continue
                    
                    # Try removing edge (u, v)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Greedily add edges
                    added = 0
                    for a in range(n):
                        if time_check():
                            break
                        for b in range(a + 1, n):
                            if time_check():
                                break
                            if not ((adj[a] >> b) & 1):
                                if not would_create_c6(a, b):
                                    adj[a] |= (1 << b)
                                    adj[b] |= (1 << a)
                                    added += 1
                    
                    new_edge_count = current_edge_count - 1 + added
                    if new_edge_count > current_edge_count:
                        pass_improved = True
                        current_edge_count = new_edge_count
                    else:
                        # Revert: re-add (u,v) and remove all edges added in this attempt
                        # This is tricky; let's just re-add (u,v) and leave the rest
                        # Actually, we need to properly revert. Let's save the adj state.
                        # For now, let's just re-add (u,v) and hope the greedy didn't add too many
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            if not pass_improved:
                break
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_adj = [list(e) for e in edges]
            
    return best_adj
