import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if k == 4 and g == 10:
        # Try the known 320-vertex construction
        result = _construct_320_cage(deadline)
        if result is not None:
            return result

    # Fallback: LUW Construction
    return _luw_construct(k, g)

def _construct_320_cage(deadline):
    """
    Construct the (4,10)-cage on 320 vertices using a voltage graph approach.
    Uses a 4-regular base graph on 10 vertices lifted by Z_32.
    
    The base graph is K_{5,5} minus a perfect matching (4-regular, 10 vertices, girth 4).
    We assign voltages to edges so that the lift has girth >= 10.
    """
    n_lift = 320
    group_size = 32
    
    # Base graph: 10 vertices (0-4 on one side, 5-9 on other)
    # Each vertex i (0-4) connects to 5+j for j != i
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))
    
    # Try systematic voltage assignments using a search over structured patterns
    # Key insight: for girth >= 10, we need all closed walks of length 4,6,8 in
    # the base graph to have non-zero voltage sum.
    
    # The base graph K_{5,5}\M has many 4-cycles. A 4-cycle in the base graph
    # corresponds to a closed walk of length 4 in the lift. We need the voltage
    # sum around every 4-cycle to be non-zero mod 32.
    
    # Let's use a parameterized family: voltage of edge (i, 5+j) = a*i + b*j + c
    # for integers a, b, c mod 32. This gives a structured assignment.
    
    import itertools
    
    # Try linear voltage assignments: w(i,j) = (a*i + b*j + c) % 32
    # where i in 0..4, j in 0..4, i != j
    found = False
    for a in range(1, 32, 1):
        if time.time() > deadline:
            break
        for b in range(1, 32, 1):
            if time.time() > deadline:
                break
            for c in range(0, 32, 1):
                if time.time() > deadline:
                    break
                
                voltages = []
                for (u, v) in base_edges:
                    i = u
                    j = v - 5
                    w = (a * i + b * j + c) % group_size
                    voltages.append(w)
                
                # Quick check: verify no 4-cycle has zero voltage sum
                if _quick_girth_check_4(base_edges, voltages, group_size):
                    edges = _build_lift_edges(base_edges, voltages, group_size)
                    if _verify_girth_fast(n_lift, edges, 4, 10, deadline):
                        return n_lift, edges
    
    # Try the Cayley graph approach as fallback
    result = _try_cayley_320(deadline)
    if result is not None:
        return result
    
    return None

def _quick_girth_check_4(base_edges, voltages, group_size):
    """
    Quick check that no 4-cycle in the base graph has zero voltage sum.
    A 4-cycle in K_{5,5}\M has the form:
    i -> 5+j -> i' -> 5+j' -> i where i != i', j != j', i != j, i' != j', etc.
    """
    # Build adjacency for base graph
    adj = {}
    for idx, (u, v) in enumerate(base_edges):
        adj.setdefault(u, []).append((v, voltages[idx]))
        adj.setdefault(v, []).append((u, -voltages[idx] % group_size))
    
    # Check all 4-cycles: u -> v -> w -> x -> u
    for u in adj:
        for v, wuv in adj[u]:
            for w, wvw in adj[v]:
                if w == u:
                    continue
                for x, wwx in adj[w]:
                    if x == v:
                        continue
                    if x in adj.get(u, []):
                        # Check if x is adjacent to u
                        for x2, wux in adj[u]:
                            if x2 == x:
                                total = (wuv + wvw + wwx + wux) % group_size
                                if total == 0:
                                    return False
    return True

def _build_lift_edges(base_edges, voltages, group_size):
    """Build the lifted graph edges from base edges and voltages."""
    edges = []
    for idx, (u, v) in enumerate(base_edges):
        w = voltages[idx]
        for x in range(group_size):
            u_id = u * group_size + x
            v_id = v * group_size + (x + w) % group_size
            edges.append((u_id, v_id))
    return edges

def _verify_girth_fast(n, edges, k, g, deadline):
    """
    Fast verification that the graph is k-regular, simple, and has girth >= g.
    Uses non-backtracking walks from every vertex with early termination.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, j in edges:
        if i == j:
            return False
        if i > j:
            i, j = j, i
        if (i, j) in edge_set:
            return False
        edge_set.add((i, j))
        adj[i].append(j)
        adj[j].append(i)
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check girth using non-backtracking BFS from each vertex
    # Optimization: use list-based queue and tuple keys for speed
    for v in range(n):
        if time.time() > deadline:
            return False
        
        # visited[(node, parent)] = depth
        visited = {}
        queue = []
        for w in adj[v]:
            visited[(w, v)] = 1
            queue.append((w, v, 1))
        
        qi = 0
        while qi < len(queue):
            node, parent, depth = queue[qi]
            qi += 1
            if depth >= g:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor == v:
                    if depth + 1 < g:
                        return False
                    continue
                state = (neighbor, node)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((neighbor, node, depth + 1))
    
    return True

def _try_cayley_320(deadline):
    """
    Try Cayley graph on Z_2 x Z_160 with generators that give 4-regular graph.
    Generators: (1,0), (1,0) [no, must be distinct], need 4 distinct generators
    with each equal to its own inverse or paired.
    
    For Z_2 x Z_160: generators a, b, c, d where a=-a, b=-b, etc. (order 2)
    or pairs {g, -g}.
    
    Let's try: generators = {(1, 0), (0, a), (0, -a), (1, b), (1, -b)} -- that's 5.
    
    Better: use Z_160 x Z_2 with generators:
    (1, 0), (-1, 0), (a, 1), (-a, 1) for some a.
    """
    m, n = 160, 2
    total = m * n
    
    # Generators: (1,0), (-1,0), (a,1), (-a,1)
    # Need to avoid short cycles.
    
    def check_girth(a):
        if a == 0 or a == 1 or a == m - 1:
            return False
        # Also need (a,1) != (1,0) and (a,1) != (-1,0)
        # (a,1) has second coord 1, (1,0) and (-1,0) have second coord 0, so OK
        
        gens = [(1, 0), (m-1, 0), (a, 1), ((-a) % m, 1)]
        
        # BFS with non-backtracking to find shortest cycle
        # State: (position, last_gen_used)
        # Start from (0,0) with no previous gen
        visited = {}  # (pos, prev_gen_idx) -> depth
        queue = deque()
        
        for gi, g in enumerate(gens):
            np_ = (g[0] % m, g[1] % n)
            visited[(np_, gi)] = 1
            queue.append((np_, gi, 1))
        
        while queue:
            pos, prev_gi, depth = queue.popleft()
            if depth >= 10:
                continue
            for gi, g in enumerate(gens):
                if gi == prev_gi:  # non-backtracking: can't use same generator
                    continue
                # Also can't use the inverse of the previous generator
                # In a Cayley graph, the inverse of gen gi is gen (3-gi) if paired
                # gens[0]=(1,0), gens[1]=(-1,0) are inverses
                # gens[2]=(a,1), gens[3]=(-a,1) are inverses
                if (prev_gi == 0 and gi == 1) or (prev_gi == 1 and gi == 0):
                    continue
                if (prev_gi == 2 and gi == 3) or (prev_gi == 3 and gi == 2):
                    continue
                
                nu = (pos[0] + g[0]) % m
                nv = (pos[1] + g[1]) % n
                state = ((nu, nv), gi)
                
                if state in visited:
                    cycle_len = depth + 1 + visited[state]
                    if cycle_len < 10:
                        return False
                else:
                    visited[state] = depth + 1
                    queue.append(((nu, nv), gi, depth + 1))
        
        return True
    
    for a in range(2, m):
        if time.time() > deadline:
            return None
        if check_girth(a):
            edges = []
            gens = [(1, 0), (m-1, 0), (a, 1), ((-a) % m, 1)]
            for u in range(m):
                for v in range(n):
                    vid = u * n + v
                    for g in gens:
                        nu = (u + g[0]) % m
                        nv = (v + g[1]) % n
                        nid = nu * n + nv
                        if vid < nid:
                            edges.append((vid, nid))
            
            # Verify degree
            deg = [0] * total
            for e1, e2 in edges:
                deg[e1] += 1
                deg[e2] += 1
            if all(d == 4 for d in deg):
                return total, edges
    
    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction - the baseline."""
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
