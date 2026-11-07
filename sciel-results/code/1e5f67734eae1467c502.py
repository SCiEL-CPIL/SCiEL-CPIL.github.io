"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 15}

def construct():
    n = P['n']
    # Partition vertices into two sets A and B
    # To maximize edges in a bipartite graph, split as evenly as possible.
    # 15 = 7 + 8.
    # K_{7,8} has 56 edges.
    # Degrees in A: 8. Degrees in B: 7.
    # Any 3-path s-u-v-t in a bipartite graph connects vertices in different partitions.
    # Thus deg(s) != deg(t) always holds for 3-paths.
    
    A = list(range(7))
    B = list(range(7, 15))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
            
    # Optional: Verify validity to ensure no bugs in logic
    # This is fast for n=15
    adj = [[False]*n for _ in range(n)]
    deg = [0]*n
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
        
    # Check all simple 3-paths s-u-v-t
    valid = True
    for s in range(n):
        for u in range(n):
            if s == u or not adj[s][u]:
                continue
            for v in range(n):
                if u == v or not adj[u][v]:
                    continue
                for t in range(n):
                    if v == t or not adj[v][t]:
                        continue
                    if s == t:
                        continue
                    # Simple path check: s != v, u != t (already covered by distinctness in loop mostly, but s!=u, u!=v, v!=t, s!=t)
                    # Need s != v and u != t
                    if s == v or u == t:
                        continue
                    if deg[s] == deg[t]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid sparse graph if verification fails (should not happen)
        return [[0, v] for v in range(1, n)]
        
    return edges
# EVOLVE-BLOCK-END
