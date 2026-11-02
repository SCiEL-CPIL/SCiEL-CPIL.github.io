import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # If the task is specifically for the (4, 10)-cage, try to construct it
    if k == 4 and g == 10:
        # Use Cube graph Q3 (8 vertices, 12 edges, 4-regular, girth 4)
        # Lift by Z_40 to get 320 vertices.
        base_edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7)
        ]
        n_base = 8
        n_lift = 320
        group_size = 40
        
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Assign random voltages to base edges
            # v[e_index]
            v = [random.randint(0, group_size - 1) for _ in base_edges]
            
            # Build lift edges
            edges = []
            for idx, (u, v_base) in enumerate(base_edges):
                w = v[idx]
                # Edge (u, g) -- (v_base, g + w)
                # We add edges for all g in 0..39
                # To ensure uniqueness and simplicity, we just add them.
                # Since base is simple, lift is simple.
                for x in range(group_size):
                    u_id = u * group_size + x
                    v_id = v_base * group_size + (x + w) % group_size
                    edges.append((u_id, v_id))
            
            # Check girth
            if _check_girth(n_lift, edges, g):
                return n_lift, edges
        
        # If random search fails, fallback
        return _luw_construct(k, g)
    
    # Fallback for other parameters
    return _luw_construct(k, g)

def _check_girth(n, edges, min_girth):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check for cycles of length < min_girth
    # BFS from each vertex up to depth min_girth // 2 - 1
    max_depth = min_girth // 2 - 1
    
    for start in range(n):
        visited = [-1] * n
        # parent is not strictly needed if we just track depth and check back edges
        # But we need to ensure we don't go back to the immediate parent.
        # BFS tree: parent[v] = u.
        # If we see v already visited and v != parent[u], it's a back edge.
        # Cycle length = depth[u] + 1 + depth[v]? No.
        # If u and v are both in the BFS tree, and edge (u,v) exists.
        # If v is parent of u, skip.
        # If u is parent of v, skip (handled when processing v? No, we process u).
        # If neither is parent of the other, it's a cross edge or back edge.
        # Cycle length = depth[u] + depth[v] + 1 - 2*depth[lca(u,v)].
        # This is hard to compute quickly.
        
        # Simpler check:
        # If we find a back edge (u, v) where v is an ancestor of u (in BFS tree),
        # cycle length = depth[u] - depth[v] + 1.
        # If we find a cross edge (u, v) where neither is ancestor,
        # cycle length is even? No.
        
        # Actually, for girth checking, BFS from each node to depth D is sufficient.
        # If there is a cycle of length L <= 2D, BFS from a node on the cycle
        # will detect a "conflict" or back edge.
        
        # Let's use a simpler heuristic:
        # If there is a cycle of length < min_girth, there exists a vertex with
        # two distinct paths of length <= min_girth//2 - 1 to some other vertex?
        
        # Standard algorithm:
        # For each start:
        #   BFS up to max_depth.
        #   If we encounter a visited node v (v != parent[u]),
        #   we need to check the cycle length.
        #   However, BFS tree paths are shortest.
        #   If we see v visited, the cycle formed is depth[u] + 1 + depth[v] ONLY IF
        #   the path to v is disjoint from path to u except at start? No.
        
        # Let's stick to the robust method:
        # If we find an edge (u, v) such that dist(u, start) + dist(v, start) + 1 < min_girth
        # AND the paths are disjoint? No, they share the start.
        # The cycle is start -> ... -> u -> v -> ... -> start.
        # Length = dist(u, start) + 1 + dist(v, start).
        # This is a valid cycle length.
        # If this length < min_girth, we return False.
        
        visited = [-1] * n
        queue = deque([(start, -1, 0)])
        visited[start] = 0
        
        while queue:
            u, p, d = queue.popleft()
            if d >= max_depth:
                continue
            for v in adj[u]:
                if v == p:
                    continue
                if visited[v] == -1:
                    visited[v] = d + 1
                    queue.append((v, u, d + 1))
                else:
                    # Found a connection to a visited node.
                    # The cycle length is d + 1 + visited[v].
                    # This is the length of the cycle formed by the BFS tree paths to u and v
                    # plus the edge (u, v).
                    # This cycle might not be simple if the paths overlap, but the length
                    # d + 1 + visited[v] is an upper bound on the cycle length?
                    # Actually, if the paths overlap, the cycle is shorter.
                    # So if d + 1 + visited[v] < min_girth, there is definitely a short cycle.
                    if d + 1 + visited[v] < min_girth:
                        return False
                    
    return True

def _luw_construct(k, g):
    # Original LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
