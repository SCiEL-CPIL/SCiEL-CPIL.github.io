import time
import random
from collections import deque, defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Calculate Moore Bound
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))
    
    # 2. Fallback: LUW Construction
    best_n, best_edges = construct_luw(k, g)
    
    # 3. Try Voltage Graph / Lift Construction
    # This can potentially find smaller graphs than LUW by using
    # a small base graph with voltage assignments in Z_m
    try:
        n_lift, edges_lift = construct_voltage_lift(k, g, deadline)
        if n_lift is not None and n_lift < best_n:
            best_n = n_lift
            best_edges = edges_lift
    except Exception:
        pass
    
    # 4. Try Greedy Tree Construction (Moore bound)
    n_target = moore_bound
    if k % 2 == 1 and n_target % 2 != 0:
        n_target += 1
    
    try:
        n_greedy, edges_greedy = construct_greedy_tree(k, g, n_target)
        if n_greedy is not None and n_greedy < best_n:
            best_n = n_greedy
            best_edges = edges_greedy
    except Exception:
        pass
    
    return best_n, best_edges

def construct_luw(k, g):
    # Original LUW logic, kept as fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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

def construct_voltage_lift(k, g, deadline):
    """
    Voltage graph / lift construction.
    
    Idea: Take a small base graph B with k-regular structure, assign voltages 
    (elements of Z_m) to edges, then the m-fold cover is k-regular with 
    n = m * |V(B)| vertices.
    
    For k=5, g=9: We need a base graph that is "girth 9 in the voltage sense".
    The girth of the cover is determined by the shortest cycle in B whose 
    voltage sum is 0 (or divides m appropriately).
    
    Strategy: Use a small base graph (e.g., 5-regular on 10-20 vertices) and
    search for voltage assignments that give large girth in the cover.
    """
    r = (g - 1) // 2
    
    # Try different base graph sizes and group sizes
    # The cover has n = m * base_n vertices
    # We want n < best_n from LUW
    
    # LUW for k=5, g=9 gives some n_luw. Let's try to beat it.
    # First, let's try a few candidate base graphs.
    
    # Base graph 1: The 5-regular graph on 10 vertices (complement of 5 disjoint edges? No)
    # Actually, let's use known small k-regular graphs.
    
    # For k=5, small 5-regular graphs:
    # - 10 vertices: K_10 minus a perfect matching (5-regular, 10 vertices)
    # - 12 vertices: various
    # - etc.
    
    # Let's try base_n = 10, 12, 14, ... and various m values.
    
    # The girth constraint: In the voltage graph, a cycle of length L in the base
    # with voltage sum v gives a cycle of length L in the cover if v = 0 mod m.
    # For girth >= g, we need: no cycle of length < g has voltage sum 0 mod m.
    
    # Actually, more precisely: the cover graph has a cycle of length L iff there's
    # a closed walk of length L in the base whose voltage sum is 0 mod m.
    # For girth >= g, we need no closed walk of length < g with voltage sum 0 mod m.
    
    # This is a constraint on the voltage assignment.
    
    # Let's try a systematic search:
    # 1. Pick a base graph (k-regular, small)
    # 2. Try random voltage assignments in Z_m for various m
    # 3. Check if the cover has girth >= g
    # 4. If yes, return the cover
    
    # Candidate base graphs for k=5:
    # We'll use a few known small 5-regular graphs.
    
    # Generate candidate base graphs
    base_graphs = []
    
    # Base graph: 10 vertices, 5-regular (K_10 minus perfect matching)
    # Vertices 0-9, edges between i and j if |i-j| not in {0, 5} and i != j
    # Actually, K_10 is 9-regular. Remove a perfect matching to get 8-regular.
    # That's not 5-regular. Let me think again.
    
    # For 5-regular on 10 vertices: Each vertex connects to 5 others.
    # Total edges = 10*5/2 = 25.
    # One construction: circulant C_10(1,2,3,4,5)? No, that's 9-regular.
    # Circulant C_10(1,2,3,4): 8-regular.
    # Let's just use a random 5-regular graph on 10 vertices.
    
    def make_random_k_reg(n, k):
        """Generate a random k-regular graph on n vertices."""
        if n * k % 2 != 0:
            return None
        # Configuration model
        stubs = []
        for v in range(n):
            stubs.extend([v] * k)
        random.shuffle(stubs)
        
        edges = set()
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None  # Loop, try again
            edge = (min(u,v), max(u,v))
            if edge in edges:
                return None  # Multi-edge, try again
            edges.add(edge)
        return list(edges)
    
    # Try different base sizes
    for base_n in [10, 12, 14, 16, 18, 20]:
        if time.time() > deadline:
            break
            
        # Try to generate a valid k-regular base graph
        base_edges = None
        for attempt in range(50):
            base_edges = make_random_k_reg(base_n, k)
            if base_edges is not None:
                break
        
        if base_edges is None:
            continue
        
        # Now try different group sizes m
        # The cover will have m * base_n vertices
        # We want m * base_n < current best
        
        # Try m values that give reasonable total sizes
        for m in range(2, 50):
            if time.time() > deadline:
                break
            
            total_n = m * base_n
            # We'll check girth later, but we can skip if total_n is too large
            # (we don't know best_n here, but we can bound it)
            
            # Try random voltage assignments
            for volt_attempt in range(20):
                if time.time() > deadline:
                    break
                    
                # Assign random voltages to edges
                voltages = {}
                for (u, v) in base_edges:
                    voltages[(u, v)] = random.randint(0, m - 1)
                    voltages[(v, u)] = (-voltages[(u, v)]) % m
                
                # Check girth of the cover
                # The cover has vertices (v, t) where v in 0..base_n-1, t in 0..m-1
                # Edge from (u, t) to (v, t + voltages[(u,v)] mod m)
                
                # To check girth >= g, we need no cycle of length < g
                # A cycle in the cover corresponds to a closed walk in the base
                # with voltage sum 0 mod m.
                
                # We can check by BFS in the cover, but that's expensive for large m.
                # Instead, check all closed walks of length < g in the base.
                
                # For g=9, we need to check lengths 3, 4, 5, 6, 7, 8.
                # This is expensive but doable for small base graphs.
                
                # Alternative: build the cover and check girth with BFS
                # But m * base_n can be up to 50 * 20 = 1000, which is fine.
                
                # Build the cover
                cover_adj = defaultdict(list)
                for (u, v) in base_edges:
                    vu = voltages[(u, v)]
                    vv = voltages[(v, u)]
                    for t in range(m):
                        u_cover = u * m + t
                        v_cover = v * m + (t + vu) % m
                        cover_adj[u_cover].append(v_cover)
                        cover_adj[v_cover].append(u_cover)
                
                # Check degree
                for v in range(total_n):
                    if len(cover_adj[v]) != k:
                        break
                else:
                    # Check girth using non-backtracking walks
                    girth_ok = True
                    # BFS from each vertex to find shortest cycle
                    # For efficiency, only check if total_n is reasonable
                    if total_n <= 500:
                        for start_v in range(total_n):
                            if time.time() > deadline:
                                girth_ok = False
                                break
                            # BFS to find shortest cycle back to start_v
                            # Use non-backtracking: track (current, prev)
                            visited = {}
                            queue = deque()
                            for nb in cover_adj[start_v]:
                                queue.append((nb, start_v, 1))
                                visited[(nb, start_v)] = 1
                            
                            while queue and girth_ok:
                                curr, prev, dist = queue.popleft()
                                if curr == start_v:
                                    if dist < g:
                                        girth_ok = False
                                        break
                                    else:
                                        break
                                for nb in cover_adj[curr]:
                                    if nb == prev:
                                        continue
                                    if (nb, curr) not in visited:
                                        visited[(nb, curr)] = dist + 1
                                        queue.append((nb, curr, dist + 1))
                        
                        if girth_ok:
                            # Valid! Return the cover
                            cover_edges = []
                            for (u, v) in base_edges:
                                vu = voltages[(u, v)]
                                for t in range(m):
                                    u_cover = u * m + t
                                    v_cover = v * m + (t + vu) % m
                                    cover_edges.append((u_cover, v_cover))
                            return total_n, cover_edges
    
    return None, None

def construct_greedy_tree(k, g, n_target):
    r = (g - 1) // 2
    n_vertices = 1
    for i in range(1, r + 1):
        n_vertices += k * ((k - 1) ** (i - 1))
        
    if n_vertices != n_target:
        return None, None

    parent = [-1] * n_vertices
    level = [0] * n_vertices
    branch = [0] * n_vertices
    
    parent[0] = -1
    level[0] = 0
    
    next_id = 1
    for i in range(k):
        parent[next_id] = 0
        level[next_id] = 1
        branch[next_id] = i
        next_id += 1
        
    for l in range(2, r + 1):
        nodes_at_prev_level = [v for v in range(next_id) if level[v] == l-1]
        for v in nodes_at_prev_level:
            for j in range(k - 1):
                parent[next_id] = v
                level[next_id] = l
                branch[next_id] = branch[v]
                next_id += 1
                
    leaves = [v for v in range(n_vertices) if level[v] == r]
    
    branches = defaultdict(list)
    for v in leaves:
        branches[branch[v]].append(v)
        
    adj = defaultdict(list)
    for v in range(1, n_vertices):
        u = parent[v]
        adj[u].append(v)
        adj[v].append(u)
        
    deg_needed = {v: k - len(adj[v]) for v in range(n_vertices)}
    
    random.shuffle(leaves)
    
    for attempt in range(100):
        edges_to_add = []
        used_edges = set()
        success = True
        
        stubs = []
        for v in leaves:
            stubs.extend([v] * deg_needed[v])
            
        random.shuffle(stubs)
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                success = False
                break
            if branch[u] == branch[v]:
                success = False
                break
            if (min(u,v), max(u,v)) in used_edges:
                success = False
                break
            used_edges.add((min(u,v), max(u,v)))
            edges_to_add.append((u, v))
            
        if not success:
            continue
            
        final_edges = []
        for v in range(1, n_vertices):
            final_edges.append((parent[v], v))
        final_edges.extend(edges_to_add)
        
        deg_check = defaultdict(int)
        for u, v in final_edges:
            deg_check[u] += 1
            deg_check[v] += 1
        if all(deg_check[v] == k for v in range(n_vertices)):
            return n_vertices, final_edges
            
    return None, None
