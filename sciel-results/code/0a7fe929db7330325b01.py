import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    N = 40
    
    # Initialize with the seed graph: 8 disjoint K5s
    # Vertices 0-4, 5-9, ..., 35-39
    adj = [0] * N
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    best_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u] & (1 << v):
                best_edges.append([u, v])
                
    best_score = len(best_edges)
    
    def get_edge_list(adj_mat):
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if adj_mat[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def check_c6_path(u, v, adj_mat):
        """
        Returns True if there is a path of length 5 from u to v in adj_mat.
        This would form a C6 if (u,v) is added.
        """
        # R_k is the set of vertices reachable from u in exactly k steps?
        # No, "path of length 5" usually implies simple path in cycle context?
        # If we have a walk of length 5, it might repeat vertices.
        # A C6 requires a simple cycle.
        # If there is a walk of length 5, does it imply a C6?
        # If the walk repeats a vertex, it contains a shorter cycle.
        # If it contains a C3, C4, or C5, adding the edge might create a C6?
        # Example: u-a-b-c-d-e-v (length 5). If a=c, then u-a-b-a-d-e-v.
        # This walk contains a C3 (a-b-a? No, a-b and b-a is same edge).
        # Simple path is required for the cycle to be C6.
        # However, if we just check for ANY walk of length 5, we might reject valid edges
        # that only connect via walks that form smaller cycles.
        # But if we have a smaller cycle, say C4 u-a-b-c-u.
        # And we want to add (u, v).
        # If there is a walk u-a-b-c-x-v of length 5.
        # If x is not u, and path is simple, it's a C6.
        # If the walk is not simple, it might not form a C6.
        # 
        # Given the density, it's safer to check for simple paths?
        # But bitset multiplication gives walks.
        # Let's assume that if a walk of length 5 exists, and the graph is "sparse enough" or we are careful, 
        # we might get false positives.
        # 
        # Alternative: Check if (u,v) is in the 5th power of the adjacency matrix?
        # This counts walks.
        # For now, let's use the walk check. If we get stuck, we can refine.
        # Actually, if a walk of length 5 exists, it's highly likely a C6 exists or a smaller cycle exists.
        # If a smaller cycle exists, adding the edge might not create a C6.
        # But we want to MAXIMIZE edges.
        # If we reject an edge because of a false positive (walk exists but no simple C6 path),
        # we might miss edges.
        # 
        # Let's try the walk check first. It's much faster.
        
        # R0 = {u}
        # R1 = N(u)
        # R2 = N(R1)
        # ...
        # R5 = N(R4)
        # If v in R5, return True.
        
        # Note: This allows u to appear in R1..R4.
        
        r0 = 1 << u
        r1 = adj_mat[u]
        r2 = 0
        for x in range(N):
            if r1 & (1 << x):
                r2 |= adj_mat[x]
        
        r3 = 0
        for x in range(N):
            if r2 & (1 << x):
                r3 |= adj_mat[x]
                
        r4 = 0
        for x in range(N):
            if r3 & (1 << x):
                r4 |= adj_mat[x]
                
        r5 = 0
        for x in range(N):
            if r4 & (1 << x):
                r5 |= adj_mat[x]
                
        return bool(r5 & (1 << v))

    def try_add_edge(u, v):
        if adj[u] & (1 << v):
            return False
        if check_c6_path(u, v, adj):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges():
        c = 0
        for u in range(N):
            c += bin(adj[u]).count('1')
        return c // 2

    # Local Search Loop
    while time.time() < deadline:
        improved = False
        # Try to add any valid edge
        # To avoid O(N^2) scan every time if no improvement, we can shuffle or just scan.
        # N=40, N^2=1600. Scan is cheap.
        for u in range(N):
            for v in range(u + 1, N):
                if not (adj[u] & (1 << v)):
                    if try_add_edge(u, v):
                        improved = True
                        # After adding, we might need to restart scan?
                        # Or just continue.
                        # Continuing is fine.
        
        if not improved:
            # Stuck. Perform a kick.
            # Remove k random edges
            edges = get_edge_list(adj)
            if len(edges) == 0:
                break
            
            k = random.randint(3, 5)
            removed = random.sample(edges, min(k, len(edges)))
            for e in removed:
                remove_edge(e[0], e[1])
            
            # Try to re-add or add new edges
            # We can just let the main loop handle re-adding.
            # But to be more aggressive, we can try to add the removed ones back?
            # No, they were likely invalid.
            # Just let the next iteration try to fill gaps.
            
            # Update best
            current_edges = get_edge_list(adj)
            if len(current_edges) > best_score:
                best_score = len(current_edges)
                best_edges = current_edges
                
            # Occasionally, if stuck for long, try a bigger kick or restart?
            # For now, small kicks.
            
            # Check time
            if time.time() > deadline:
                break

    # Final check
    current_edges = get_edge_list(adj)
    if len(current_edges) > best_score:
        best_edges = current_edges

    return best_edges
