import time
import itertools

def construct():
    """
    Constructs a simple undirected graph on vertices 0..14 that maximizes edge count
    while avoiding equal-degree endpoints for any simple 3-edge path.
    
    Strategy:
    Start with complete bipartite graph K_{7,8} (56 edges).
    In K_{a,b} with a != b, any path s-u-v-t has s and t in different partitions,
    so deg(s) != deg(t). This is always valid.
    
    Then greedily try to add internal edges (within partition A or within partition B)
    to maximize edge count while maintaining validity.
    
    For each candidate internal edge, check if adding it preserves the constraint:
    no simple 3-edge path s-u-v-t with s != t has deg(s) == deg(t).
    
    Use efficient adjacency and degree tracking for the validity check.
    """
    n = 15
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    
    # Partition vertices
    A = list(range(7))   # size 7
    B = list(range(7, 15))  # size 8
    
    # Base edges: complete bipartite K_{7,8}
    edge_set = set()
    for a in A:
        for b in B:
            edge_set.add((a, b))
    
    # Build adjacency and degrees
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for (u, v) in edge_set:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    def is_valid(edge_set, adj, deg):
        """Check if no simple three-edge path has endpoints of equal degree."""
        for (u, v) in edge_set:
            # For path s-u-v-t: s is neighbor of u (not v), t is neighbor of v (not u), s != t
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[t] == ds:
                        return False
        return True
    
    # Verify base is valid
    assert is_valid(edge_set, adj, deg), "Base K_{7,8} should be valid"
    
    best_edges = set(edge_set)
    best_count = len(edge_set)
    
    # Generate all possible internal edges (within A or within B)
    internal_candidates = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            internal_candidates.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            internal_candidates.append((B[i], B[j]))
    
    # Greedy addition: try adding each internal edge if it maintains validity
    # Multiple passes to allow different orderings
    for pass_num in range(3):
        if time.time() > deadline:
            break
        
        added_any = False
        for (u, v) in internal_candidates:
            if time.time() > deadline:
                break
            
            # Skip if edge already present
            if (u, v) in best_edges or (v, u) in best_edges:
                continue
            
            # Try adding this edge
            new_edges = set(best_edges)
            new_edges.add((u, v))
            
            # Update adjacency and degrees temporarily
            new_adj = [set(s) for s in adj]
            new_deg = list(deg)
            new_adj[u].add(v)
            new_adj[v].add(u)
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid(new_edges, new_adj, new_deg):
                # Accept the edge
                best_edges = new_edges
                adj = new_adj
                deg = new_deg
                best_count = len(best_edges)
                added_any = True
        
        if not added_any:
            break
    
    # Convert to list format with proper types
    result = []
    for e in sorted(best_edges):
        result.append([int(e[0]), int(e[1])])
    
    return result
