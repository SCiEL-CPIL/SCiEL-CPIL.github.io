"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Better vertex ordering: use a greedy approach that prioritizes vertices
    # with more already-colored neighbors (dynamic), but for initial ordering,
    # use a structure-aware ordering. For this 4-regular graph, we'll use
    # a BFS from vertex 0 to get a natural ordering, then reverse it to
    # process "leaves" of the BFS tree first (which have fewer constraints).
    
    # Actually, let's use a different strategy: order vertices by how "central" 
    # they are. We'll compute a simple centrality measure.
    # For this graph, let's try ordering by the sum of neighbor indices (lower = earlier)
    # which tends to group connected vertices together.
    
    # Let's use a more principled approach: minimum remaining values heuristic
    # suggests we should color vertices with the most constraints first.
    # Since all have degree 4, let's use a static ordering based on graph structure.
    
    # Try multiple orderings and pick the best
    # Ordering 1: natural order
    order1 = list(range(n))
    
    # Ordering 2: BFS order from 0
    visited = [False] * n
    visited[0] = True
    queue = deque([0])
    bfs_order = []
    while queue:
        u = queue.popleft()
        bfs_order.append(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    order2 = bfs_order
    
    # Ordering 3: Reverse BFS (process leaves first)
    order3 = list(reversed(bfs_order))
    
    # Ordering 4: Sort by degree then by index (all degree 4, so by index)
    order4 = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    
    # We'll try order2 (BFS) as it tends to work well for constraint propagation
    order = order2
    
    def is_valid(color, v, c, deadline):
        """Check if assigning color c to vertex v is valid given current color assignment."""
        if time.time() > deadline:
            return False
        
        # Check properness: no neighbor has color c
        for u in adj[v]:
            if color[u] == c:
                return False
        
        # Check bicolored cycle constraint using union-find
        # For each color d != c that appears in v's neighborhood,
        # check that v's neighbors with color d are not connected in {c,d}-subgraph
        
        neighbor_colors = set()
        for u in adj[v]:
            if color[u] != -1:
                neighbor_colors.add(color[u])
        
        for d in neighbor_colors:
            if d == c:
                continue
            
            # Find all neighbors of v that have color d
            neighbors_with_d = [u for u in adj[v] if color[u] == d]
            
            if len(neighbors_with_d) < 2:
                continue
            
            # Use union-find to check if any two neighbors_with_d are connected
            # in the subgraph induced by colors {c, d}
            
            # Build a temporary union-find for the {c,d} subgraph
            # Only include vertices with color c or d
            uf_parent = {}
            uf_rank = {}
            
            def uf_find(x):
                if x not in uf_parent:
                    uf_parent[x] = x
                    uf_rank[x] = 0
                if uf_parent[x] != x:
                    uf_parent[x] = uf_find(uf_parent[x])
                return uf_parent[x]
            
            def uf_union(x, y):
                rx, ry = uf_find(x), uf_find(y)
                if rx == ry:
                    return
                if uf_rank[rx] < uf_rank[ry]:
                    uf_parent[rx] = ry
                elif uf_rank[rx] > uf_rank[ry]:
                    uf_parent[ry] = rx
                else:
                    uf_parent[ry] = rx
                    uf_rank[rx] += 1
            
            # Union all edges in the {c,d} subgraph
            for u in range(n):
                if color[u] in (c, d):
                    uf_find(u)  # initialize
                    for w in adj[u]:
                        if color[w] in (c, d) and u < w:
                            uf_union(u, w)
            
            # Check if any two neighbors_with_d are in the same component
            for i in range(len(neighbors_with_d)):
                for j in range(i + 1, len(neighbors_with_d)):
                    if uf_find(neighbors_with_d[i]) == uf_find(neighbors_with_d[j]):
                        return False
        
        return True
    
    def backtrack(idx, color, k, deadline):
        """Backtracking search with k colors."""
        if time.time() > deadline:
            return None
        
        if idx == n:
            return color[:]
        
        v = order[idx]
        
        # Try each color from 0 to k-1
        for c in range(k):
            if is_valid(color, v, c, deadline):
                color[v] = c
                res = backtrack(idx + 1, color, k, deadline)
                if res is not None:
                    return res
                color[v] = -1
        
        return None
    
    # Try increasing k from 3 upward
    deadline = time.time() + 85.0
    
    for k in range(3, 15):
        if time.time() > deadline - 3.0:
            break
        
        color = [-1] * n
        res = backtrack(0, color, k, deadline)
        if res is not None:
            # Verify the solution
            # Check properness
            for u, v in edges:
                if res[u] == res[v]:
                    break
            else:
                # Check bicolored cycles
                valid = True
                for c1 in range(k):
                    for c2 in range(c1 + 1, k):
                        # Check if the subgraph induced by colors c1 and c2 is a forest
                        vertices_in_subgraph = set()
                        for i in range(n):
                            if res[i] in (c1, c2):
                                vertices_in_subgraph.add(i)
                        
                        if not vertices_in_subgraph:
                            continue
                        
                        # Count edges in the subgraph
                        edges_in_subgraph = 0
                        for u, v in edges:
                            if res[u] in (c1, c2) and res[v] in (c1, c2):
                                edges_in_subgraph += 1
                        
                        # A forest on v vertices has at most v - 1 edges per component
                        # Total edges <= total vertices - number of components
                        # Check using union-find
                        uf_parent = {}
                        uf_rank = {}
                        
                        def uf_find(x):
                            if x not in uf_parent:
                                uf_parent[x] = x
                                uf_rank[x] = 0
                            if uf_parent[x] != x:
                                uf_parent[x] = uf_find(uf_parent[x])
                            return uf_parent[x]
                        
                        def uf_union(x, y):
                            rx, ry = uf_find(x), uf_find(y)
                            if rx == ry:
                                return False
                            if uf_rank[rx] < uf_rank[ry]:
                                uf_parent[rx] = ry
                            elif uf_rank[rx] > uf_rank[ry]:
                                uf_parent[ry] = rx
                            else:
                                uf_parent[ry] = rx
                                uf_rank[rx] += 1
                            return True
                        
                        for u in vertices_in_subgraph:
                            uf_find(u)
                        
                        for u, v in edges:
                            if res[u] in (c1, c2) and res[v] in (c1, c2):
                                if not uf_union(u, v):
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return [int(x) for x in res]
    
    # Fallback: use all distinct colors
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
