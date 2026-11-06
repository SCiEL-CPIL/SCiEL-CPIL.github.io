import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 118  # Safety margin for 120s limit

    def time_check():
        return time.time() - start_time > deadline

    best_edge_count = 0
    best_edges = None

    # Pre-compute all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        C6: u - w - z - y - x - v
        w in N(u), x in N(v).
        Path w-z-y-x of length 3 in old graph.
        """
        nu = adj[u]
        nv = adj[v]
        
        w_mask = nu
        while w_mask:
            wb = w_mask & -w_mask
            wi = wb.bit_length() - 1
            w_mask ^= wb
            
            x_mask = nv
            while x_mask:
                xb = x_mask & -x_mask
                xi = xb.bit_length() - 1
                x_mask ^= xb
                
                if wi == xi:
                    continue
                
                nw = adj[wi]
                nw_minus_x = nw & ~(1 << xi)
                
                z_mask = nw_minus_x
                while z_mask:
                    zb = z_mask & -z_mask
                    zi = zb.bit_length() - 1
                    z_mask ^= zb
                    
                    common = adj[zi] & adj[xi]
                    common_minus_w = common & ~(1 << wi)
                    
                    if common_minus_w:
                        return True
        return False

    def try_add_edge(adj, edges, u, v):
        """Try to add edge (u,v) if it doesn't create a C6."""
        if (adj[u] >> v) & 1:
            return False
        if would_create_c6(adj, u, v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append([u, v])
        return True

    def local_search(adj, edges):
        """
        1-swap local search: try removing one edge to enable adding more edges.
        Also do a final polish pass adding any remaining addable edges.
        """
        improved = True
        while improved:
            improved = False
            
            # First, try to add any edge that can be added
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        if try_add_edge(adj, edges, u, v):
                            improved = True
            
            # Try 1-swap: remove edge (a,b), then try to add edges
            for a in range(n):
                for b in range(a + 1, n):
                    if time_check():
                        return adj, edges
                    if not (adj[a] >> b) & 1:
                        continue
                    
                    # Save state
                    adj_save = [x for x in adj]
                    edges_save = [list(e) for e in edges]
                    
                    # Remove edge (a,b)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    # Find and remove from edges list
                    for i in range(len(edges) - 1, -1, -1):
                        if (edges[i][0] == a and edges[i][1] == b) or (edges[i][0] == b and edges[i][1] == a):
                            edges.pop(i)
                            break
                    
                    # Try to add edges after removal
                    added_count = 0
                    for u in range(n):
                        for v in range(u + 1, n):
                            if not (adj[u] >> v) & 1:
                                if try_add_edge(adj, edges, u, v):
                                    added_count += 1
                    
                    # If we added more than 1 edge (net gain), keep the change
                    if added_count > 1:
                        improved = True
                    else:
                        # Restore state
                        adj = adj_save
                        edges = edges_save
        
        return adj, edges

    # We will try multiple restarts
    for seed in range(500):
        if time_check():
            break
            
        random.seed(seed)
        
        # Initialize with 8 disjoint K5s
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
        
        # Get non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        # Greedy phase: add edges in random order
        added = 0
        for u, v in non_edges:
            if time_check():
                break
            if try_add_edge(adj, edges, u, v):
                added += 1
        
        # Local search phase
        adj, edges = local_search(adj, edges)
        
        if len(edges) > best_edge_count:
            best_edge_count = len(edges)
            best_edges = [list(e) for e in edges]
            
            # If we hit 105, we can stop early
            if best_edge_count >= 105:
                break

    return best_edges
