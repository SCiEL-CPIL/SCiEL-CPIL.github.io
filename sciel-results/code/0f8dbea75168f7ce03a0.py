import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (150s kill)

    def time_check():
        return time.time() > deadline

    best_edge_count = 0
    best_edges = None

    def make_adj_from_edges(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u,v) creates a C6 in the graph defined by adj.
        C6: u - w - z - y - x - v where all vertices distinct.
        w in N(u), x in N(v), path w-z-y-x of length 3 in old graph.
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

    def try_add_edges_greedy(adj, edges, non_edges_order):
        """Try to add as many edges as possible from non_edges_order."""
        for u, v in non_edges_order:
            if time_check():
                break
            if not (adj[u] >> v) & 1:
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        return edges

    def local_search(adj, edges, max_iters=2000):
        """
        Local search: try to swap one edge for two non-edges to increase edge count.
        Or add a single edge if possible.
        Returns improved adj and edges, or None if no improvement.
        """
        improved = False
        for _ in range(max_iters):
            if time_check():
                break
            
            # First, try to add any available non-edge
            found_add = False
            for u in range(n):
                if time_check():
                    break
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    if not would_create_c6(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
                        found_add = True
                        improved = True
                        break
                if found_add:
                    break
            
            if found_add:
                continue
            
            # Try edge swaps: remove edge (a,b), add (a,c) and (b,d) or (a,d) and (b,c)
            # This is complex; try simpler: remove one edge, try to add two others
            # For now, just try removing a random edge and see if we can add two
            if len(edges) < 2:
                break
                
            # Pick a random edge to potentially remove
            eidx = random.randrange(len(edges))
            a, b = edges[eidx]
            
            # Temporarily remove
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            edges.pop(eidx)
            
            # Try to add two new edges
            added_back = []
            for u in range(n):
                if time_check():
                    break
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    if not would_create_c6(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
                        added_back.append((u, v))
                        if len(added_back) == 2:
                            break
                if len(added_back) == 2:
                    break
            
            # If we added at least 1, we're better off (net gain if 2, tie if 1)
            # Actually, if we added 0, revert. If 1, it's a wash but might enable more later.
            # Let's only keep if we added >= 1 (to allow exploration)
            if len(added_back) < 1:
                # Revert
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                edges.append([a, b])
            else:
                improved = True

        return adj, edges, improved

    # Phase 1: Multiple greedy restarts
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
        
        # Greedy add
        edges = try_add_edges_greedy(adj, edges, non_edges)
        
        # Local search refinement
        adj, edges, improved = local_search(adj, edges)
        
        if len(edges) > best_edge_count:
            best_edge_count = len(edges)
            best_edges = [list(e) for e in edges]
            
            if best_edge_count >= 105:
                break

    # Phase 2: If we have a good solution, do more intensive local search on it
    if best_edges is not None and not time_check():
        adj = make_adj_from_edges(best_edges)
        edges = [list(e) for e in best_edges]
        
        # Run more local search iterations
        for _ in range(10):
            if time_check():
                break
            adj, edges, improved = local_search(adj, edges, max_iters=5000)
            if not improved:
                break
            if len(edges) > best_edge_count:
                best_edge_count = len(edges)
                best_edges = [list(e) for e in edges]

    return best_edges
