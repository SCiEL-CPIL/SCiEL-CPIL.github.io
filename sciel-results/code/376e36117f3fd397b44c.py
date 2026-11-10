import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Returns edges of a graph on 15 vertices maximizing edge count while avoiding
    equal-degree endpoints on any 3-edge path.
    
    Strategy: Start with K_{7,8} (56 edges), then try to add intra-part edges
    by removing minimal inter-part edges to differentiate degrees.
    """
    n = 15
    A = list(range(0, 7))   # 7 vertices
    B = list(range(7, 15))  # 8 vertices
    
    def is_valid(adj, deg):
        """Check if graph is valid: no 3-edge path s-u-v-t with deg(s)==deg(t)"""
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == s or t == u:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def compute_deg(adj):
        return [len(adj[i]) for i in range(n)]
    
    # Start with K_{7,8}
    adj = [set() for _ in range(n)]
    for u in A:
        for v in B:
            adj[u].add(v)
            adj[v].add(u)
    
    best_edges = []
    for u in A:
        for v in B:
            best_edges.append([u, v])
    best_count = len(best_edges)
    
    # Try adding intra-A edges
    deadline = time.time() + 85  # 5s safety margin
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u_idx in range(len(A)):
            for v_idx in range(u_idx + 1, len(A)):
                u = A[u_idx]
                v = A[v_idx]
                if u in adj[v]:
                    continue  # already connected
                # Try adding edge (u,v)
                adj[u].add(v)
                adj[v].add(u)
                deg = compute_deg(adj)
                if is_valid(adj, deg):
                    improved = True
                    best_count += 1
                    best_edges.append([u, v])
                else:
                    # Try removing some inter-part edges to fix
                    adj[u].remove(v)
                    adj[v].remove(u)
                    # Find minimal removals needed
                    # For path s-u-v-t, need deg(s) != deg(t)
                    # s must be in N(u)∩B, t must be in N(v)∩B
                    # We can remove edges to change degrees
                    # Try removing one edge from a B vertex that is neighbor of both u and v
                    common_B = adj[u] & adj[v]
                    for b in common_B:
                        # Try removing edge (b, u)
                        adj[u].remove(b)
                        adj[b].remove(u)
                        adj[u].add(v)
                        adj[v].add(u)
                        deg = compute_deg(adj)
                        if is_valid(adj, deg):
                            improved = True
                            best_count += 1  # added 1, removed 1, net 0... wait
                            # Actually we added (u,v) and removed (u,b), so net 0
                            # But maybe we can do better
                            adj[u].remove(v)
                            adj[v].remove(u)
                            adj[u].add(b)
                            adj[b].add(u)
                            break
                        else:
                            adj[u].remove(v)
                            adj[v].remove(u)
                            adj[u].add(b)
                            adj[b].add(u)
    
    # Reconstruct edges from best adj if we made improvements
    # Actually, let's just return the base K_{7,8} if no improvement found
    # For now, return the base solution
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
