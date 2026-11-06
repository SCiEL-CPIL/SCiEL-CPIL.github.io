import time
import random

def construct():
    """
    Constructs a simple graph on 20 vertices with edge colours 0..19.
    Maximizes the number of edges while ensuring every 4-cycle is rainbow.
    
    Strategy:
    1. Use Greedy Construction with Randomized Restarts.
    2. Start with an empty graph.
    3. Try to add edges in a random order.
    4. An edge (u, v) can be added if it does not create a 4-cycle.
       This is equivalent to checking if there is no path of length 3 between u and v.
       A path of length 3 u-a-b-v exists if there are a, b such that 
       (u,a), (a,b), (b,v) are edges.
    5. Repeat for multiple restarts to escape local optima.
    6. Return the densest valid graph found.
    """
    n = 20
    deadline = time.time() + 115  # Safety margin
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m_total = len(all_edges)
    
    def has_path_length_3(adj, u, v):
        """
        Check if there is a path of length 3 between u and v.
        Path: u - a - b - v
        Requires: a in N(u), b in N(v), (a,b) in E, a != b.
        """
        # Optimization: iterate over the smaller neighbor set
        nu = adj[u]
        nv = adj[v]
        
        if len(nu) > len(nv):
            nu, nv = nv, nu
            u, v = v, u # Symmetry, though we just need existence
        
        for a in nu:
            if a == v:
                continue
            # For each neighbor a of u, check if a is connected to any neighbor b of v
            # such that b is not u and b is not a (a!=b is implied if b in nv and a in nu, 
            # but we must ensure the edge (a,b) exists)
            # Actually, we just need to know if any a in N(u) is connected to any b in N(v) 
            # via an edge (a,b).
            # Wait, the path is u-a-b-v.
            # So we need (u,a) [true], (a,b) [check], (b,v) [true if b in N(v)].
            # So we check if there exists a in N(u) and b in N(v) such that (a,b) is an edge.
            # And a != b (which is true if u != v and graph is simple, unless triangle? 
            # If a==b, path is u-a-a-v which is not a simple path. 
            # But (a,a) is not an edge. So a!=b is naturally satisfied if (a,b) is an edge).
            
            # Check if any neighbor of a is in N(v)
            # Iterate over N(a) and check intersection with N(v)
            # This might be slow if degrees are high.
            # Alternative: Precompute adjacency matrix?
            # With n=20, we can use bitmasks.
            pass
            
        # Bitmask approach is much faster for n=20
        # Let's use a global bitmask adjacency matrix for speed
        return False # Placeholder, will use bitmask version

    # Use bitmasks for adjacency
    # adj_mask[i] is an integer where bit j is set if (i,j) is an edge
    
    def greedy_construct(seed):
        random.seed(seed)
        adj_mask = [0] * n
        edge_count = 0
        
        # Random order of edges
        edges = all_edges[:]
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u,v) creates a C4
            # This happens if there is a path of length 3 from u to v.
            # Path u-a-b-v.
            # This requires existence of a, b such that:
            # 1. (u,a) is edge -> a in N(u)
            # 2. (b,v) is edge -> b in N(v)
            # 3. (a,b) is edge
            # 4. a != b (implied if (a,b) is edge in simple graph)
            # 5. a != v, b != u (implied if u,v not adjacent yet, but we are checking before adding)
            
            # Let N_u = set of neighbors of u
            # Let N_v = set of neighbors of v
            # We need to check if there is an edge (a,b) with a in N_u and b in N_v.
            
            # Using bitmasks:
            # N_u_mask = adj_mask[u]
            # N_v_mask = adj_mask[v]
            
            # We want to know if there exists a in N_u and b in N_v such that (a,b) is an edge.
            # This is equivalent to: 
            # For any a in N_u, is there a neighbor b of a that is in N_v?
            # Or: Is the intersection of N_u and (union of N(a) for a in N_u) ... no.
            
            # Let's iterate over a in N_u.
            # If N_u is small, this is fast.
            # n=20, max degree 19.
            
            n_u_mask = adj_mask[u]
            n_v_mask = adj_mask[v]
            
            # Iterate over bits in n_u_mask
            a = n_u_mask
            found = False
            while a:
                # Get lowest set bit
                lsb = a & -a
                a ^= lsb
                idx_a = lsb.bit_length() - 1
                
                # Check if any neighbor of idx_a is in n_v_mask
                # neighbors of idx_a is adj_mask[idx_a]
                # Intersection with n_v_mask
                if (adj_mask[idx_a] & n_v_mask) != 0:
                    found = True
                    break
            
            if not found:
                # Safe to add
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edge_count += 1
        
        return edge_count, adj_mask

    best_edge_count = 0
    best_adj_mask = [0] * n
    
    num_restarts = 1000  # Adjust based on time
    
    # Estimate time per restart
    # Greedy is O(m * deg^2) roughly. m=190. deg~10. 190*100 = 19000 ops.
    # Should be very fast.
    
    start_time = time.time()
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Use different random seeds or just let random run
        # To ensure diversity, we can also try different "priorities"
        # But simple shuffle is a good start.
        
        ec, am = greedy_construct(i)
        
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj_mask = am
            
        # If we reached a very high density, we might stop early?
        # Max edges for C4-free graph on 20 vertices?
        # Ex(20, C4) is approx n^{3/2} / 2 = 20^{1.5} / 2 = 89.4 / 2 = 44.7?
        # Wait, Reiman bound: e(n, C4) <= n/4 (1 + sqrt(4n-3)).
        # For n=20: 5 * (1 + sqrt(77)) = 5 * (1 + 8.77) = 48.85.
        # So max edges is around 48-50.
        # If we find 50, we can likely stop.
        if best_edge_count >= 50:
            break

    # Construct final edge list
    coloured_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj_mask[u] & (1 << v)):
                # Assign a colour. Since graph is C4-free, any colouring is valid for the "rainbow 4-cycle" constraint 
                # because there are NO 4-cycles. 
                # The condition "every 4-cycle is rainbow" is vacuously true if there are no 4-cycles.
                # So we can assign colour 0 to all edges.
                coloured_edges.append([u, v, 0])
                
    return coloured_edges
