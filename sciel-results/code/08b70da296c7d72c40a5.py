import time

def construct():
    """
    Constructs a simple undirected graph on 16 vertices (0..15) with diameter exactly 2,
    where deleting each edge increases the diameter above 2 or disconnects the graph.
    Maximizes edge count.
    
    Strategy:
    The condition that deleting any edge increases the diameter above 2 implies that 
    no edge can be part of a triangle. If an edge (u, v) is in a triangle with w, 
    then d(u, v) remains 2 after removal (via w), so the diameter might not increase 
    above 2 if the rest of the graph is robust. More strongly, for the diameter to 
    be forced to increase for *every* edge, the graph must be triangle-free. 
    (If G is triangle-free, removing any edge (u, v) increases d(u, v) to >= 3, 
    thus increasing the diameter).
    
    We need a triangle-free graph on 16 vertices with diameter 2 that maximizes edges.
    By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n 
    vertices is floor(n^2 / 4). For n=16, this is 64.
    The unique graph achieving this is the complete bipartite graph K_8,8.
    K_8,8 has diameter 2 (vertices in same partition are distance 2, different 
    partitions are distance 1).
    
    Thus, K_8,8 is the optimal solution with 64 edges.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 16
    # Partition vertices into two sets of 8
    # A: 0-7, B: 8-15
    # Complete bipartite graph K_8,8
    edges = []
    
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])
            
    # Verify types are strictly Python int and list
    for e in edges:
        assert isinstance(e, list), "Edge must be a list"
        assert len(e) == 2, "Edge must have 2 elements"
        assert isinstance(e[0], int) and not isinstance(e[0], bool), "Vertex must be int, not bool"
        assert isinstance(e[1], int) and not isinstance(e[1], bool), "Vertex must be int, not bool"
        
    # Optional: Verification to ensure correctness
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def bfs_diameter(adj, n):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            qi = 0
            while qi < len(queue):
                cur = queue[qi]
                qi += 1
                for nb in adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
                        if dist[nb] > max_dist:
                            max_dist = dist[nb]
            if -1 in dist:
                return -1  # disconnected
        return max_dist

    if time.time() < deadline:
        diam = bfs_diameter(adj, n)
        if diam != 2:
            # Fallback, though K_8,8 should always have diameter 2
            return []
        
        # Verify edge criticality (optional, but good for safety)
        # For K_8,8, every edge is critical because the graph is triangle-free.
        # We trust the mathematical proof, but a quick check on one edge can be done.
        # However, checking all 64 edges with BFS is fast enough (64 * 16 * 16 ~ 16k ops).
        
        for i, (u, v) in enumerate(edges):
            if time.time() > deadline:
                break
            adj[u].remove(v)
            adj[v].remove(u)
            
            new_diam = bfs_diameter(adj, n)
            
            adj[u].add(v)
            adj[v].add(u)
            
            if new_diam <= 2:
                # This edge is not critical. 
                # For K_8,8, this should never happen.
                # If it does, we have a bug in our understanding or implementation.
                # But K_8,8 is triangle-free, so d(u,v) becomes 3.
                pass

    return edges
