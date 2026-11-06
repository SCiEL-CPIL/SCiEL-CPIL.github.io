import random
import time
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices with a high number of edges.
    Uses randomized greedy with bitset C6-checks and local search polish.
    """
    N = 40
    deadline = time.time() + 105  # 45s safety margin
    
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) creates a C6 using bitsets."""
        # C6: u-a-x-y-b-v where a~u, b~v, x~a, y~b, x~y
        # and all 6 vertices distinct
        nu = adj[u] & ~((1 << v) | (1 << u))
        nv = adj[v] & ~((1 << u) | (1 << v))
        
        # For each a in N(u), b in N(v) with a != b:
        # need x in N(a) \ {u,v,a,b}, y in N(b) \ {u,v,a,b,x} with x~y
        # Equivalent: (N(a) & N(b) & ~(mask_u_v_a_b)) != 0
        # but x and y must be distinct, and both in their respective neighborhoods
        # More carefully: we need x in N(a), y in N(b), x~y, x!=y, 
        # x not in {u,v,a,b}, y not in {u,v,a,b,x}
        
        # For each pair (a,b):
        # candidates for x: N(a) & ~{u,v,a,b}
        # candidates for y: N(b) & ~{u,v,a,b}
        # need x~y for some x in cand_x, y in cand_y, x!=y
        # i.e., for some x in cand_x: (adj[x] & cand_y & ~(1<<x)) != 0
        
        nu_iter = nu
        while nu_iter:
            a_bit = nu_iter & (-nu_iter)
            a = a_bit.bit_length() - 1
            nu_iter ^= a_bit
            
            nv_iter = nv
            while nv_iter:
                b_bit = nv_iter & (-nv_iter)
                b = b_bit.bit_length() - 1
                nv_iter ^= b_bit
                
                if a == b:
                    continue
                
                forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << b)
                cand_x = adj[a] & ~forbidden
                cand_y = adj[b] & ~forbidden
                
                if cand_x == 0 or cand_y == 0:
                    continue
                
                # Check if any x in cand_x has neighbor in cand_y
                cx = cand_x
                while cx:
                    x_bit = cx & (-cx)
                    x = x_bit.bit_length() - 1
                    cx ^= x_bit
                    if adj[x] & cand_y & ~(1 << x):
                        return True
        
        return False
    
    def local_search(adj, edge_set, time_limit):
        """Try to add edges via local search."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(all_edges)
            for u, v in all_edges:
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    improved = True
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        adj = [0] * N
        edge_set = set()
        
        for u, v in all_edges:
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        current_count = len(edge_set)
        
        # Local search polish
        ls_deadline = min(time.time() + 5, deadline - 2)
        local_search(adj, edge_set, ls_deadline)
        
        current_count = len(edge_set)
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edge_set)
            
    # Return in the required format: list of [u, v]
    return [[u, v] for u, v in best_edges]
