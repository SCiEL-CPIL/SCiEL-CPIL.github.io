"""
C6-free graph construction on 40 vertices.
Starts with 8 disjoint K5s (80 edges) and greedily adds edges
that do not create a C6, then performs local search to improve.
Uses bitmasks for fast adjacency checks and multiple restarts.
"""
import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill

    # Initialize with 8 disjoint K5s
    def init_adj():
        adj = [0] * n  # bitmask for neighbors
        edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        return adj, edges

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A new C6 containing (u,v) would be u-w-z-y-x-v-u,
        which requires a path of length 5 from u to v in the current graph.
        w in N(u), x in N(v).
        Need path w-z-y-x of length 3.
        This exists if there are z, y such that w~z, z~y, y~x.
        """
        nu = adj[u]
        nv = adj[v]
        # Iterate over neighbors of u and v
        # For each w in N(u), x in N(v), check if dist(w,x) == 3
        # dist(w,x)==3 iff exists z in N(w) such that N(z) & N(x) != {w} (need y != w)
        
        # Get list of neighbors for iteration
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        
        for w in nu_list:
            nw = adj[w]
            for x in nv_list:
                if w == x:
                    continue
                # Check if there's a path w-z-y-x of length 3
                # For each z in N(w) \ {x}, check if N(z) & N(x) has element != w
                nx = adj[x]
                # z must be in N(w) and z != x
                # We need: exists z in N(w) such that (N(z) & N(x)) - {w} != empty
                # But also need z != x (already handled) and the path to be simple
                # Let's iterate z in N(w)
                zw = nw
                # For each z in N(w):
                while zw:
                    z_bit = zw & (-zw)
                    z = z_bit.bit_length() - 1
                    zw ^= z_bit
                    if z == x:
                        continue
                    # Check if N(z) & N(x) has any element other than w
                    common = adj[z] & nx
                    # Remove w from common
                    common &= ~(1 << w)
                    if common:
                        return True
        return False

    def greedy_add_edges(adj, edges, non_edges, deadline):
        """Greedily add edges that don't create C6."""
        added = 0
        for u, v in non_edges:
            if time.time() > deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                added += 1
        return added

    def local_search(adj, edges, deadline):
        """Try to improve by removing one edge and adding two."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each edge and see if we can add more
            # This is expensive, so limit attempts
            for i in range(len(edges)):
                if time.time() > deadline:
                    return adj, edges
                u, v = edges[i]
                # Temporarily remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges_removed = edges.pop(i)
                
                # Try to add edges that were previously blocked
                added_any = False
                # Check all non-edges
                for a in range(n):
                    if time.time() > deadline:
                        break
                    for b in range(a + 1, n):
                        if (adj[a] >> b) & 1:
                            continue
                        if not would_create_c6(adj, a, b):
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
                            edges.append([a, b])
                            added_any = True
                
                if not added_any:
                    # Restore the edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append(edges_removed)
                else:
                    # Check if we actually gained edges
                    # If we added at least 2, it's a net gain
                    # For simplicity, accept if we added any
                    improved = True
                    break
        
        return adj, edges

    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edge_count = 0
    best_adj = None
    best_edges = None
    
    # Multiple restarts with different random orderings
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        adj, edges = init_adj()
        non_edges = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
        
        # Shuffle non-edges with different seed
        random.seed(restart * 7 + 13)
        random.shuffle(non_edges)
        
        # Greedy phase
        greedy_add_edges(adj, edges, non_edges, deadline - 5)
        
        current_count = len(edges)
        if current_count > best_edge_count:
            best_edge_count = current_count
            best_adj = adj[:]
            best_edges = edges[:]
        
        # Local search if we have time
        if time.time() < deadline - 15 and current_count >= best_edge_count - 2:
            adj, edges = local_search(adj, edges, deadline - 5)
            current_count = len(edges)
            if current_count > best_edge_count:
                best_edge_count = current_count
                best_adj = adj[:]
                best_edges = edges[:]
    
    return best_edges

# EVOLVE-BLOCK-END
