import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # If the task is specifically for the (4, 10)-cage, try to construct it
    if k == 4 and g == 10:
        # Base graph: K_{5,5} minus perfect matching
        # Vertices 0..4 (top), 5..9 (bottom)
        # Edges (i, 5+j) for i != j
        
        base_edges = []
        for i in range(5):
            for j in range(5):
                if i != j:
                    base_edges.append((i, 5 + j))
        
        n_base = 10
        n_lift = 320
        group_size = 32
        
        # Pre-calculate base edge indices for voltage assignment
        # We need voltages for each directed edge.
        # Let's assign voltage v[i][j] for edge (i, 5+j) where i != j.
        # The reverse edge (5+j, i) has voltage -v[i][j].
        
        max_attempts = 2000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Assign random voltages
            # v[i][j] for i in 0..4, j in 0..4, i != j
            v = [[0]*5 for _ in range(5)]
            for i in range(5):
                for j in range(5):
                    if i != j:
                        v[i][j] = random.randint(0, 31)
            
            # Build lift edges
            edges = []
            # For each base edge (u, v) with voltage w
            # u is 0..4, v is 5..9
            # w = v[u][v-5]
            for u, v_base in base_edges:
                i = u
                j = v_base - 5
                w = v[i][j]
                # Add edges (u*32 + x, v*32 + (x+w)%32)
                # And the reverse is handled by the other direction?
                # No, base_edges contains (i, 5+j).
                # The lift is undirected.
                # Edge in lift: (u_id, v_id).
                # We add it once.
                for x in range(32):
                    u_id = u * 32 + x
                    v_id = v_base * 32 + (x + w) % 32
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
        parent = [-1] * n
        queue = [(start, -1, 0)]
        visited[start] = 0
        
        while queue:
            u, p, d = queue.pop(0)
            if d >= max_depth:
                continue
            for v in adj[u]:
                if v == p:
                    continue
                if visited[v] == -1:
                    visited[v] = d + 1
                    parent[v] = u
                    queue.append((v, u, d + 1))
                else:
                    # Found a back edge
                    # Cycle length is d + 1 + visited[v]
                    # But this is the length of the cycle formed by the tree path
                    # and the edge (u, v).
                    # We need to check if this cycle is shorter than min_girth.
                    # However, the tree path might not be the shortest path.
                    # But if there is a cycle of length L < min_girth,
                    # BFS will find a back edge with d + 1 + visited[v] <= L?
                    # Actually, BFS finds the shortest path.
                    # If a cycle of length L exists, there is a back edge
                    # such that the cycle length formed is <= L.
                    # So if d + 1 + visited[v] < min_girth, we have a short cycle.
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
