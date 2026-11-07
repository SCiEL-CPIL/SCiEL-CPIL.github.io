def construct():
    n = 15
    import time
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Start with K_{7,8}
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
    
    def is_valid(edge_set, adj, deg):
        """Check if no 3-edge path has endpoints of equal degree."""
        for (u, v) in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def try_add_edge(edge_set, adj, deg, u, v):
        """Try adding edge (u,v). Return (success, new_adj, new_deg) or (False, adj, deg)."""
        if u == v:
            return False, adj, deg
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False, adj, deg
        
        # Check validity with the new edge
        # The new edge (u,v) creates new 3-edge paths through it
        # Also, existing paths might now have different endpoint degrees
        # Actually, adding an edge only changes degrees of u and v
        # So we need to check:
        # 1. New paths using edge (u,v): s-u-v-t where s in adj[u], t in adj[v]
        # 2. Existing paths where an endpoint was u or v (degree changed)
        
        new_deg = deg[:]
        new_deg[u] += 1
        new_deg[v] += 1
        
        # Check new paths through (u,v)
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if new_deg[s] == new_deg[t]:
                    return False, adj, deg
        
        # Check existing paths where endpoint degree changed
        # Paths s-x-y-t where s or t is u or v
        # For paths where s=u: u-x-y-t, need new_deg[u] != new_deg[t]
        # For paths where t=u: s-x-y-u, need new_deg[s] != new_deg[u]
        # Similarly for v
        
        for (x, y) in edge_set:
            if (x == u or y == u or x == v or y == v):
                # This edge is incident to u or v
                pass
        
        # More systematically: check all 3-edge paths where an endpoint is u or v
        # Path s-a-b-t: endpoints s,t. If s==u, check new_deg[u] != new_deg[t]
        # If t==u, check new_deg[s] != new_deg[u]
        # If s==v, check new_deg[v] != new_deg[t]
        # If t==v, check new_deg[s] != new_deg[v]
        
        for (a, b) in edge_set:
            # Paths s-a-b-t
            for s in adj[a]:
                if s == b:
                    continue
                for t in adj[b]:
                    if t == a or t == s:
                        continue
                    if s == u:
                        if new_deg[s] == new_deg[t]:
                            return False, adj, deg
                    elif t == u:
                        if new_deg[s] == new_deg[t]:
                            return False, adj, deg
                    elif s == v:
                        if new_deg[s] == new_deg[t]:
                            return False, adj, deg
                    elif t == v:
                        if new_deg[s] == new_deg[t]:
                            return False, adj, deg
                    # If neither s nor t is u or v, degrees unchanged, was valid before
        
        # Also need to check the new edge itself creates paths
        # Already checked above for (u,v)
        
        return True, adj, new_deg
    
    # Build adjacency and degree
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edge_set = set()
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        edge_set.add((min(u,v), max(u,v)))
    
    # Local search: try adding edges
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u in range(n):
            for v in range(u+1, n):
                if time.time() > deadline:
                    break
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    continue
                success, new_adj, new_deg = try_add_edge(edge_set, adj, deg, u, v)
                if success:
                    edge_set.add(key)
                    adj[u].add(v)
                    adj[v].add(u)
                    deg = new_deg
                    changed = True
            if time.time() > deadline:
                break
    
    # Build final edge list
    final_edges = []
    for (u, v) in edge_set:
        final_edges.append([u, v])
    
    return final_edges
