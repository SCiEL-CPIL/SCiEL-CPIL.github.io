import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for evaluator kill at 150s
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Fallback seed solution: 8 disjoint K5s
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                best_edges.append([5*k+a, 5*k+b])
    best_count = len(best_edges)

    def has_path_len_4(u, v, adj):
        """
        Check if there is a simple path of length 4 between u and v in the graph defined by adj.
        Path: u - w1 - w2 - w3 - v
        """
        # Get neighbors of u
        nu = [i for i in range(n) if adj[u][i]]
        if not nu:
            return False
        
        # Get vertices at distance 2 from u, mapping w2 -> set of w1
        dist2_u = {}
        for w1 in nu:
            # w1 is neighbor of u
            for w2 in range(n):
                if w2 != u and w2 != w1 and adj[w1][w2]:
                    if w2 not in dist2_u:
                        dist2_u[w2] = set()
                    dist2_u[w2].add(w1)
        
        if not dist2_u:
            return False

        # Get neighbors of v
        nv = [i for i in range(n) if adj[v][i]]
        if not nv:
            return False
        
        # Check for intersection with distance 2 from v
        # We need w2 in dist2_u such that there is w3 in nv with adj[w3][w2]
        # And w1 != w3
        for w3 in nv:
            if adj[w3][w3]: # Self loop check, not needed for simple graph but safe
                continue
            # Check if w3 connects to any w2 in dist2_u
            for w2 in dist2_u:
                if adj[w3][w2]:
                    set_u = dist2_u[w2]
                    # We need w1 in set_u such that w1 != w3
                    # If len(set_u) > 1, we can always pick w1 != w3 (unless set_u == {w3} and len==1)
                    # If len(set_u) == 1:
                    #   If the single element is w3, then w1 must be w3, which is invalid.
                    #   Else, w1 is not w3, valid.
                    if len(set_u) == 1:
                        if list(set_u)[0] == w3:
                            continue
                    # Otherwise, valid
                    return True
        return False

    def prune_graph(adj):
        """Remove edges that are part of a C6 until none exist."""
        changed = True
        while changed:
            changed = False
            for u in range(n):
                if time.time() > deadline - 5:
                    return adj, True
                for v in range(u+1, n):
                    if adj[u][v]:
                        # Check if (u,v) is in a C6
                        # Temporarily remove
                        adj[u][v] = 0
                        adj[v][u] = 0
                        if has_path_len_4(u, v, adj):
                            # Keep removed
                            changed = True
                        else:
                            # Restore
                            adj[u][v] = 1
                            adj[v][u] = 1
                        if changed:
                            break
                if changed:
                    break
        return adj, False

    def optimize_graph(adj):
        """Add edges if they don't create a C6."""
        changed = True
        while changed:
            changed = False
            for u in range(n):
                if time.time() > deadline - 5:
                    return adj, True
                for v in range(u+1, n):
                    if not adj[u][v]:
                        # Check if adding (u,v) creates a C6
                        # This is equivalent to: does path of length 4 exist between u and v in current adj?
                        if has_path_len_4(u, v, adj):
                            # Cannot add
                            continue
                        else:
                            # Add it
                            adj[u][v] = 1
                            adj[v][u] = 1
                            changed = True
            # If changed, restart loop to check new possibilities
        return adj, False

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    # Main search loop
    while time.time() < deadline:
        # 1. Random Init
        adj = [[0]*n for _ in range(n)]
        p = 0.6
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u][v] = 1
                    adj[v][u] = 1
        
        # 2. Prune
        adj, timed_out = prune_graph(adj)
        if timed_out:
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
            continue
        
        # 3. Optimize
        adj, timed_out = optimize_graph(adj)
        if timed_out:
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
            continue
            
        edges = get_edges(adj)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
    return best_edges
