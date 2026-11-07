import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def construct():
    """
    Find minimum number of edges to add to the path on 24 vertices
    such that the result is triangle-free, connected, and has diameter <= 4.
    
    Strategy:
    1. The path already has diameter 23. We need to reduce it to <= 4.
    2. Triangle-free constraint: no three mutually adjacent vertices.
    3. Key insight: In a triangle-free graph, if u-v and v-w are edges, then u-w is NOT an edge.
    
    For diameter 4 on 24 vertices, we need a structure where every pair of vertices is within
    distance 4. 
    
    A promising approach: Use a "star-like" structure with a few hub vertices.
    But hubs must be triangle-free, so a vertex can have at most... actually, a vertex can have
    many neighbors as long as no two neighbors are adjacent to each other.
    
    Let's think about the structure. The path 0-1-2-...-23 has diameter 23.
    We need to add edges to bring the diameter down to 4.
    
    One idea: Connect vertices that are far apart but maintain triangle-free property.
    If we add edge (i, j), we must ensure that for any k, not both (i,k) and (j,k) are edges.
    
    Let's try a different approach: Greedy with backtracking, or use the fact that
    the minimum might be small (like 4-6 edges).
    
    Actually, let's think about what's the minimum number of edges needed.
    With diameter 4, the graph must be "compact". 
    For a path of length 23, we need shortcuts.
    
    Consider adding edges that jump across the path:
    - (0, 6), (6, 12), (12, 18), (18, 23) - but these might create triangles
    - Actually, if we add (0,6), then 0-1-2-3-4-5-6 and 0-6 creates a cycle of length 7, no triangle.
    - If we add (0,6) and (6,12), then 0-6-12 and 0-1-...-12, no triangle since 0-12 is not added.
    
    Let me try: add edges to create a structure where the diameter is small.
    
    Alternative: Use BFS to check diameter, and greedy add edges that reduce the most distances.
    
    For a 24-vertex path, let's try adding edges strategically:
    - The farthest pairs are (0,23), (0,22), etc.
    - If we add (0,12), (12,23), that helps but diameter is still large.
    
    Let me try a systematic approach:
    - Start with the path
    - Repeatedly find the pair with maximum distance
    - Add an edge that reduces many distances while maintaining triangle-free
    
    But we need to be careful about triangle-free constraint.
    
    Actually, for this specific problem, let's try a few candidate solutions:
    
    Candidate 1: Add edges (0,5), (5,10), (10,15), (15,20), (20,23)
    - Check: 0-5-10-15-20-23, each step is 5 apart except last
    - Diameter: max(5,5,5,5,3) = 5, too large
    
    Candidate 2: Add edges (0,6), (6,12), (12,18), (18,23)
    - Diameter: max(6,6,6,5) = 6, too large
    
    Candidate 3: Need better shortcuts.
    
    Let me think differently. The path has 23 edges. To get diameter 4, we need
    the graph to be "4-hops" from any vertex to any other.
    
    For a line graph, if we add edges that skip 7 vertices:
    (0,7), (7,14), (14,21), (21,23) - diameter would be max(7,7,7,2)=7, still too large.
    
    We need more aggressive shortcuts or a different structure.
    
    What if we use a "bipartite" approach? The path is already bipartite (even/odd).
    Adding edges between same-parity vertices would break bipartiteness but not necessarily create triangles.
    Actually, in a bipartite graph, all cycles are even-length, so no triangles.
    If we add edges only between same-parity vertices, we might create triangles.
    Wait: if u and v have same parity, and we add (u,v), then for a triangle we'd need a w adjacent to both.
    If w has opposite parity, w is adjacent to u and v only if... in the path, w is adjacent to u only if |w-u|=1.
    
    Let me try a computational approach: use a greedy algorithm with triangle-free check.
    """
    n = P['n']
    deadline = time.time() + 80.0  # Leave 10s margin
    
    # Build initial graph
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    def bfs_distances(adj, start):
        """BFS from start, return distance array."""
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist
    
    def all_pair_distances(adj):
        """Compute all-pair shortest path distances."""
        dists = []
        for i in range(n):
            dists.append(bfs_distances(adj, i))
        return dists
    
    def get_diameter(dists):
        """Return the maximum distance."""
        max_d = 0
        for d in dists:
            for x in d:
                if x > max_d:
                    max_d = x
        return max_d
    
    def is_triangle_free(adj, u, v):
        """Check if adding edge (u,v) would create a triangle."""
        # Triangle would be u-v-w-u where w is common neighbor
        for w in adj[u]:
            if w != v and w in adj[v]:
                return False
        return True
    
    # Greedy: repeatedly add the edge that most reduces the diameter
    added_edges = []
    
    def try_add_edge(u, v, adj, edge_set, added_edges):
        """Try to add edge (u,v), return True if successful."""
        if (u,v) in edge_set or (v,u) in edge_set:
            return False
        if not is_triangle_free(adj, u, v):
            return False
        # Add the edge
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
        added_edges.append([u, v])
        return True
    
    def remove_edge(u, v, adj, edge_set, added_edges):
        """Remove edge (u,v)."""
        adj[u].remove(v)
        adj[v].remove(u)
        edge_set.discard((min(u,v), max(u,v)))
        if added_edges:
            for i, e in enumerate(added_edges):
                if (e[0], e[1]) == (u, v) or (e[0], e[1]) == (v, u):
                    added_edges.pop(i)
                    break
    
    # Initial diameter
    dists = all_pair_distances(adj)
    current_diam = get_diameter(dists)
    
    # Greedy approach: while diameter > 4, find best edge to add
    max_iterations = 100
    iteration = 0
    
    while current_diam > 4 and iteration < max_iterations:
        if time.time() > deadline:
            break
        iteration += 1
        
        # Find candidate edges to add
        best_edge = None
        best_diam = current_diam + 1
        best_score = -1
        
        # Consider all non-adjacent pairs
        for u in range(n):
            for v in range(u+1, n):
                if (u,v) in edge_set:
                    continue
                if not is_triangle_free(adj, u, v):
                    continue
                
                # Temporarily add edge
                adj[u].add(v)
                adj[v].add(u)
                
                # Compute new diameter
                new_dists = all_pair_distances(adj)
                new_diam = get_diameter(new_dists)
                
                # Remove edge
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = (u, v)
                    best_score = current_diam - new_diam
        
        if best_edge is None:
            break
        
        u, v = best_edge
        try_add_edge(u, v, adj, edge_set, added_edges)
        
        dists = all_pair_distances(adj)
        current_diam = get_diameter(dists)
    
    # Verify the result
    if get_diameter(all_pair_distances(adj)) > 4:
        # Fallback: return a known valid solution
        # For now, return empty if we can't find a solution
        pass
    
    return added_edges
