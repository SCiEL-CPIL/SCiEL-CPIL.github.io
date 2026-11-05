import time
import random
from collections import deque
import itertools

def _moore_bound(k, g):
    if g % 2 == 0:
        depth = g // 2 - 1
        return 2 * (1 + k * (k**depth - 1) // (k - 1))
    else:
        depth = (g - 1) // 2
        return 1 + k * (k**depth - 1) // (k - 1)

def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        parent[start] = -2
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _luw_construct(k, g):
    """Construct the Lazebnik-Ustimenko-Woldar graph."""
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 5:
        add = lambda x, y: (x + y) % 5
        sub = lambda x, y: (x - y) % 5
        mul = lambda x, y: x * y % 5
    elif q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 3:
        add = lambda x, y: (x + y) % 3
        sub = lambda x, y: (x - y) % 3
        mul = lambda x, y: x * y % 3
    elif q == 2:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: x & y
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
            
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _try_cayley_search(k, g, deadline):
    """
    Try to find a smaller graph using Cayley graphs on specific groups.
    Returns (n, edges) if found, else None.
    """
    # Candidate groups of size in [428, 625]
    # We'll use A5 x Z8 (size 480) as a primary candidate.
    # A5 elements: 60 permutations.
    # We represent A5 elements as tuples of 5 integers (permutations of 0..4).
    # Z8 elements: 0..7.
    
    # Generate A5 elements
    # A5 is the group of even permutations of 5 elements.
    # We can generate all 120 permutations and filter for even.
    perms = list(itertools.permutations(range(5)))
    a5_elems = []
    for p in perms:
        # Calculate parity
        inv_count = 0
        for i in range(5):
            for j in range(i+1, 5):
                if p[i] > p[j]:
                    inv_count += 1
        if inv_count % 2 == 0:
            a5_elems.append(p)
    
    # Map A5 elements to indices 0..59
    a5_index = {p: i for i, p in enumerate(a5_elems)}
    
    # Precompute A5 multiplication table
    # mult_a5[i][j] = index of a5_elems[i] * a5_elems[j]
    mult_a5 = [[0]*60 for _ in range(60)]
    for i, p1 in enumerate(a5_elems):
        for j, p2 in enumerate(a5_elems):
            # Compose p1 then p2? Or p2 then p1?
            # Convention: (p1 * p2)(x) = p1(p2(x))
            composed = tuple(p1[p2[x]] for x in range(5))
            mult_a5[i][j] = a5_index[composed]
    
    # Group: A5 x Z8
    # Element: (a5_idx, z)
    # Multiplication: (i1, z1) * (i2, z2) = (mult_a5[i1][i2], (z1+z2)%8)
    
    n_group = 60 * 8
    # n_group = 480
    
    # We want a 5-regular graph.
    # Pick 5 generators S = {(s1_a, s1_z), ..., (s5_a, s5_z)}
    # Ensure distinct neighbors.
    
    def get_neighbors(g_idx, S):
        a5_idx, z = divmod(g_idx, 8)
        neighbors = []
        for sa, sz in S:
            na = mult_a5[a5_idx][sa]
            nz = (z + sz) % 8
            neighbors.append(na * 8 + nz)
        return neighbors

    def check_validity(S):
        # Check degree 5
        # Check girth >= g
        # Build adj list
        adj = [[] for _ in range(n_group)]
        for u in range(n_group):
            nbrs = get_neighbors(u, S)
            # Check distinct
            if len(set(nbrs)) != 5:
                return False
            adj[u] = nbrs
        
        # Check simple graph (no loops)
        for u in range(n_group):
            if u in adj[u]:
                return False
        
        # Check girth
        # BFS from each vertex
        for start in range(n_group):
            dist = [-1] * n_group
            parent = [-1] * n_group
            dist[start] = 0
            parent[start] = -2
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    # Random search
    # We try to find S such that check_validity(S) is True.
    # To speed up, we can prune early in BFS.
    
    # We'll try a few random sets.
    # Limit iterations based on time.
    start_time = time.time()
    max_iters = 2000 # Adjust based on speed
    
    for _ in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        # Generate random S
        # Pick 5 random elements from the group.
        # Avoid identity (0,0).
        # Avoid inverses to ensure distinct neighbors?
        # Inverse of (a, z) is (a_inv, -z mod 8).
        # We just check distinctness in check_validity.
        
        S = []
        while len(S) < 5:
            a = random.randint(0, 59)
            z = random.randint(0, 7)
            if a == 0 and z == 0:
                continue
            # Check if (a,z) is already in S
            if (a, z) in S:
                continue
            S.append((a, z))
        
        # Quick check: distinct neighbors for identity
        id_nbrs = get_neighbors(0, S)
        if len(set(id_nbrs)) != 5:
            continue
        if 0 in id_nbrs: # Loop
            continue
            
        # Full check
        if check_validity(S):
            # Construct edges
            edges = []
            for u in range(n_group):
                for v in get_neighbors(u, S):
                    if u < v:
                        edges.append((u, v))
            return n_group, edges
            
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Get LUW baseline
    n_luw, edges_luw = _luw_construct(k, g)
    
    # 2. Try to find a smaller graph
    # Only try if k=5, g=9 and n_luw > Moore Bound
    if k == 5 and g == 9:
        mb = _moore_bound(k, g)
        if n_luw > mb:
            result = _try_cayley_search(k, g, deadline)
            if result:
                n_found, edges_found = result
                # Verify just in case
                if _verify_girth(n_found, edges_found, g):
                    return n_found, edges_found
    
    # 3. Return LUW
    return n_luw, edges_luw
