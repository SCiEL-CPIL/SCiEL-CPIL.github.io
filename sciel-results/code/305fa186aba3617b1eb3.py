import time

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # The optimal strategy for n=15 is the complete bipartite graph K_{7,8}.
    # Vertices 0..6 form Part A (size 7).
    # Vertices 7..14 form Part B (size 8).
    # All edges connect A to B.
    # Degrees in A are 8. Degrees in B are 7.
    # Any path of length 3 s-u-v-t in a bipartite graph has s and t in opposite parts.
    # Thus deg(s) != deg(t) is always satisfied because 8 != 7.
    
    edges = []
    part_a = range(0, 7)
    part_b = range(7, 15)
    
    for u in part_a:
        for v in part_b:
            edges.append([u, v])
            
    # Verify validity (optional, but good for robustness)
    # This check is O(E^2) or similar, but for 56 edges it is instantaneous.
    def is_valid(edge_list):
        if not edge_list:
            return True
        adj = {i: set() for i in range(n)}
        deg = [0] * n
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        for u, v in edge_list:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    if not is_valid(edges):
        # Fallback if somehow invalid (should not happen)
        return [[0, 1], [1, 2], [2, 3]]
        
    return edges
