import sys
import time
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strategy 1: Try to construct a smaller graph using LUW(q=4, d=5) + Matching
    # This is valid for k=5.
    if k == 5:
        try:
            n, edges = construct_luw_plus_matching(4, 5, 5, g)
            if n is not None:
                # Verify
                if verify(n, edges, k, g):
                    return n, edges
        except Exception:
            pass
    
    # Strategy 2: Fallback to standard LUW
    # The seed program logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def construct_luw_plus_matching(q, d, target_k, g):
    # Build LUW graph
    # q=4, d=5
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    adj = [[] for _ in range(10000)] # Pre-allocate large enough
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                adj[index[vertex]].append(index[other])
                adj[index[other]].append(index[vertex])
                
    n = len(order)
    
    # Compute distances
    # n=512, so n BFS is fast
    dist = [[-1]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
                    
    # Build bipartite matching graph
    # Points are side 0, Lines are side 1
    points = [i for i in range(n) if order[i][0] == 0]
    lines = [i for i in range(n) if order[i][0] == 1]
    
    # We need a perfect matching between Points and Lines
    # such that dist(p, l) >= 8
    # Cycle length = dist + 1 >= 9
    
    # Hopcroft-Karp
    # Left: points, Right: lines
    # Edge if dist[p][l] >= 8
    
    n_p = len(points)
    n_l = len(lines)
    if n_p != n_l:
        return None, None
        
    # Map original index to local index
    p_idx = {p: i for i, p in enumerate(points)}
    l_idx = {l: i for i, l in enumerate(lines)}
    
    # Adjacency for matching
    match_adj = [[] for _ in range(n_p)]
    for p in points:
        for l in lines:
            if dist[p][l] >= 8:
                match_adj[p_idx[p]].append(l_idx[l])
                
    # Hopcroft-Karp implementation
    def hopcroft_karp(adj_list, n_left, n_right):
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist_hk = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist_hk[u] = 0
                    queue.append(u)
                else:
                    dist_hk[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj_list[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist_hk[pair_v[v]] == float('inf'):
                        dist_hk[pair_v[v]] = dist_hk[u] + 1
                        queue.append(pair_v[v])
            return found
            
        def dfs(u):
            for v in adj_list[u]:
                if pair_v[v] == -1 or (dist_hk[pair_v[v]] == dist_hk[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist_hk[u] = float('inf')
            return False
            
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching, pair_u, pair_v
        
    matching, pair_u, pair_v = hopcroft_karp(match_adj, n_p, n_l)
    
    if matching == n_p:
        new_edges = list(edges)
        for u in range(n_p):
            if pair_u[u] != -1:
                p = points[u]
                l = lines[pair_u[u]]
                new_edges.append((p, l))
        return n, new_edges
        
    return None, None

def verify(n, edges, k, g):
    # Check degree
    degree = [0] * n
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v: return False
        degree[u] += 1
        degree[v] += 1
        adj[u].append(v)
        adj[v].append(u)
        
    if any(d != k for d in degree):
        return False
        
    # Check girth using BFS from each vertex
    # This is O(n * (n+k))
    # For n=512, it's fine.
    for i in range(n):
        dist = [-1] * n
        dist[i] = 0
        queue = deque([i])
        parent = [-1] * n
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    # But this is not necessarily the shortest cycle through i
                    # We need to be careful.
                    pass
                    
    # Proper girth check:
    # For each vertex, do BFS. The first time we see a visited node (not parent),
    # we found a cycle. The length is dist[u] + 1 + dist[v]? No.
    # Standard method:
    # For each edge (u,v), if u and v are in the same BFS tree from some root?
    # Simpler: For each vertex s, run BFS.
    # When exploring u, if neighbor v is visited and v != parent[u],
    # then we have a cycle of length dist[u] + dist[v] + 1?
    # No, dist is from s.
    # If v is visited, and v is not parent[u], then there is a cycle.
    # The length of the cycle is dist[u] + 1 + (dist[v] - dist[u])? No.
    # The path s->...->u->v->...->s.
    # Length = dist[u] + 1 + (dist[v] - dist[u])? No.
    # The distance from s to v is dist[v].
    # The path s->...->v has length dist[v].
    # The path s->...->u->v has length dist[u]+1.
    # The cycle length is dist[u] + 1 + dist[v] - 2*dist[lca]?
    # This is getting complicated.
    
    # Use the property: Girth is the minimum cycle length.
    # We can just check all pairs? No.
    # For n=512, we can do BFS from each node and track the shortest cycle found.
    
    min_girth = float('inf')
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = deque([s])
        parent = [-1] * n
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        # Cycle found
                        # Length is dist[u] + dist[v] + 1?
                        # No, this formula works if the tree is a BFS tree from s.
                        # The cycle is s->...->u-v->...->s.
                        # The length is dist[u] + 1 + dist[v].
                        # Wait, is this correct?
                        # If u and v are both at depth d, cycle is 2d+1.
                        # If u is at depth d, v is at depth d+1.
                        # The edge u-v connects depth d and d+1.
                        # This is a tree edge? No, if it were a tree edge, v would be parent of u or vice versa.
                        # If v is already visited, and v is not parent of u, then u-v is a cross edge.
                        # The cycle length is dist[u] + dist[v] + 1.
                        # Let's verify.
                        # Path s->u length dist[u].
                        # Path s->v length dist[v].
                        # Edge u-v.
                        # Cycle s->...->u->v->...->s.
                        # Length = dist[u] + 1 + dist[v].
                        # This is correct for BFS trees.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            
    return min_girth >= g
