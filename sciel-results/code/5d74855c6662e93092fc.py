"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 15}

def check_validity(adj, n):
    """Check if no 3-edge path s-u-v-t has deg(s) == deg(t) with s != t."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # u-v is an edge, now find all s-u-v-t paths
            # s is a neighbor of u (s != v), t is a neighbor of v (t != u)
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u:
                        continue
                    if s == t:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s
    
    # Start with star: vertex 0 connected to all others
    edges = set()
    for v in range(1, n):
        edges.add((0, v))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # All possible edges not yet in graph
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    remaining = [p for p in all_pairs if p not in edges]
    
    # Greedy: try to add edges in order of some heuristic
    # Try edges that connect vertices with currently different degree "potential"
    # Simply try all remaining edges, add if valid
    added = True
    while added and time.time() < deadline:
        added = False
        # Sort remaining by some heuristic: prefer edges between low-degree vertices
        deg = [len(adj[i]) for i in range(n)]
        remaining.sort(key=lambda p: (deg[p[0]] + deg[p[1]], p))
        
        for (u, v) in remaining[:]:
            if time.time() >= deadline:
                break
            # Try adding edge (u, v)
            adj[u].add(v)
            adj[v].add(u)
            if check_validity(adj, n):
                edges.add((u, v))
                remaining.remove((u, v))
                added = True
            else:
                adj[u].remove(v)
                adj[v].remove(u)
    
    # Convert to list of [int, int]
    result = [[int(u), int(v)] for (u, v) in edges]
    return result
# EVOLVE-BLOCK-END
