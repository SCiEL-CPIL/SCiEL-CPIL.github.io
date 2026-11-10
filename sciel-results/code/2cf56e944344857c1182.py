# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145
    n = 40
    
    # Helper: check if adding edge (u,v) creates a C6 in graph with adjacency bitsets adj
    def creates_c6(adj, u, v):
        mask_all = (1 << n) - 1
        mask_uv = mask_all ^ (1 << u) ^ (1 << v)
        nu = adj[u] & mask_uv
        nv = adj[v] & mask_uv
        
        if nu == 0 or nv == 0:
            return False
        
        # We need to find w,x,y,z such that:
        # u~w, w~x, x~y, y~z, z~v, all distinct, and not using edge (u,v)
        # This is a path of length 4 from u to v in G - {u,v}
        
        # Optimized approach: 
        # For each w in N(u)\{v}, for each z in N(v)\{u},
        #   check if there's a path of length 2 from w to z in G - {u,v,w,z}
        #   i.e., exists x,y such that w~x, x~y, y~z with x,y not in {u,v,w,z}
        
        # Even faster: for each w in N(u)\{v}, compute neighbors of w (excluding u,v)
        # For each z in N(v)\{u}, compute if any neighbor of z (excluding u,v) is within distance 2 of w
        # But that's still complex.
        
        # Let's use the direct 4-level approach but with early termination:
        w_iter = nu
        while w_iter:
            lb = w_iter & (-w_iter)
            w_iter ^= lb
            wi = lb.bit_length() - 1
            nw = adj[wi] & mask_uv & ~(1 << wi)
            if nw == 0:
                continue
            x_iter = nw
            while x_iter:
                lbx = x_iter & (-x_iter)
                x_iter ^= lbx
                xi = lbx.bit_length() - 1
                nx = adj[xi] & mask_uv & ~(1 << xi)
                if nx == 0:
                    continue
                # Now check if any y in nx has a neighbor in nv (excluding u,v,y)
                # Combine: does (union of N(y) for y in nx) intersect nv?
                # But we need y != u,v and z != u,v,w,x
                # Let's iterate y in nx and check if adj[y] & nv != 0
                y_iter = nx
                while y_iter:
                    lby = y_iter & (-y_iter)
                    y_iter ^= lby
                    yi = lby.bit_length() - 1
                    # z must be in N(v) \ {u,v,y} and N(y) \ {u,v,x}
                    if adj[yi] & nv:
                        return True
        return False
    
    def run_sa(seed_val):
        random.seed(seed_val)
        
        # Start with 8 disjoint K5's
        edge_set = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edge_set.add((5 * k + a, 5 * k + b))
        
        adj = [0] * n
        for (a, b) in edge_set:
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        
        current_edges = len(edge_set)
        best_edges = current_edges
        best_edge_set = set(edge_set)
        
        # Precompute all possible edges
        all_edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
        
        temp = 50.0
        cooling = 0.9995
        
        while time.time() < deadline - 2:
            if temp < 0.1:
                break
            
            # Random move: add or remove an edge
            if random.random() < 0.6 and current_edges < 200:
                # Try to add an edge
                u, v = random.choice(all_edges)
                if (u, v) in edge_set:
                    continue
                if not creates_c6(adj, u, v):
                    edge_set.add((u, v))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_edge_set = set(edge_set)
                else:
                    # Reject with probability
                    pass
            else:
                # Remove an edge
                if current_edges <= 20:
                    continue
                u, v = random.choice(list(edge_set))
                edge_set.discard((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
            
            temp *= cooling
        
        return list(best_edge_set)
    
    # Run multiple SA trials
    best_result = []
    best_count = 0
    
    for trial in range(150):
        if time.time() > deadline - 5:
            break
        result = run_sa(trial)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
            if best_count >= 105:
                break
    
    return [[a, b] for (a, b) in sorted(best_result)]
# EVOLVE-BLOCK-END
