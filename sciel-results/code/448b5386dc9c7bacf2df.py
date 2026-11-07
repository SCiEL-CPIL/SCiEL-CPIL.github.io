import random
import time

def construct():
    n = 15
    
    def solve():
        # Start with empty graph
        edges = set()
        adj = [[False]*n for _ in range(n)]
        deg = [0]*n
        
        # List of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        def is_valid_addition(u, v):
            # Temporarily add edge (u, v) to check validity
            # We only need to check paths where u or v is an endpoint
            # because only their degrees change
            
            # Degrees after addition
            new_deg_u = deg[u] + 1
            new_deg_v = deg[v] + 1
            
            # Helper to check if edge (a, b) exists in G' (G + (u,v))
            def exists(a, b):
                if a == b: return False
                if (a == u and b == v) or (a == v and b == u):
                    return True
                return adj[a][b]
            
            # Check paths s-a-b-t where s is u or v
            # s must be u or v
            for s in [u, v]:
                new_deg_s = new_deg_u if s == u else new_deg_v
                # a is neighbor of s in G'
                for a in range(n):
                    if a == s: continue
                    if not exists(s, a): continue
                    # b is neighbor of a in G'
                    for b in range(n):
                        if b == a or b == s: continue
                        if not exists(a, b): continue
                        # t is neighbor of b in G'
                        for t in range(n):
                            if t == b or t == a or t == s: continue
                            if not exists(b, t): continue
                            # Path s-a-b-t is simple and exists
                            # Check degree constraint
                            # deg'(s) != deg'(t)
                            new_deg_t = deg[t] + (1 if (t == u or t == v) else 0)
                            # Wait, deg[t] changes only if t is u or v
                            # But s is u or v.
                            # If t == s, we skipped.
                            # If t == u and s == v, then new_deg_t = new_deg_u.
                            # If t == v and s == u, then new_deg_t = new_deg_v.
                            # If t is not u or v, new_deg_t = deg[t].
                            
                            if new_deg_s == new_deg_t:
                                return False
                                
            # Also need to check paths where t is u or v?
            # The loop above covers s in {u, v}.
            # A path s-a-b-t with t=u is the same as path u-b-a-s with s=u.
            # Since the graph is undirected, checking s in {u,v} covers all paths
            # where at least one endpoint is u or v.
            
            return True

        for u, v in all_edges:
            if not adj[u][v]:
                if is_valid_addition(u, v):
                    # Add edge
                    edges.add((u, v))
                    adj[u][v] = True
                    adj[v][u] = True
                    deg[u] += 1
                    deg[v] += 1
                # else: skip edge
            
        return list(edges)

    best_edges = []
    start_time = time.time()
    
    # Run multiple restarts
    while time.time() - start_time < 55:
        random.seed() # Re-seed for variety
        res = solve()
        if len(res) > len(best_edges):
            best_edges = res
            
    # Ensure types are correct
    final_edges = []
    for u, v in best_edges:
        if isinstance(u, bool) or isinstance(v, bool):
            continue
        if not isinstance(u, int) or not isinstance(v, int):
            continue
        final_edges.append([int(u), int(v)])
        
    return final_edges
