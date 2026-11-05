import time
import random
from collections import defaultdict, deque
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # 1. Calculate Moore Bound
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))

    # 2. Fallback: LUW Construction (deterministic, fast)
    best_n, best_edges = construct_luw(k, g)

    # 3. Try Tensor Product Construction
    try:
        n_tensor, edges_tensor = construct_tensor_product(k, g, deadline)
        if n_tensor is not None and n_tensor < best_n:
            best_n = n_tensor
            best_edges = edges_tensor
    except Exception:
        pass

    # 4. Try Voltage Graph / Lift Construction with optimized DP check
    try:
        n_lift, edges_lift = construct_voltage_lift_optimized(k, g, deadline)
        if n_lift is not None and n_lift < best_n:
            best_n = n_lift
            best_edges = edges_lift
    except Exception:
        pass

    # 5. Try Greedy Tree Construction (Moore bound) - likely to fail for g=9, k=5
    # but kept for completeness if it ever works
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

def construct_tensor_product(k, g, deadline):
    """
    Use tensor product (Kronecker product) of two graphs to construct a k-regular graph.
    For k=5, factor as 5*1: tensor of 5-regular graph with a matching (1-regular on 2 vertices).
    This duplicates the 5-regular graph. Girth is preserved if base graph is non-bipartite.
    
    Alternatively, try Cartesian product: deg(G) + deg(H) = k.
    For k=5: (3,2) or (4,1) or (1,4) or (2,3).
    Girth of Cartesian product G□H: any cycle projects to closed walks in G and H.
    The girth is at least min(girth(G), girth(H)) but often larger.
    """
    best_result = None
    
    # Try Cartesian product with degree split (3,2)
    # G: 3-regular, H: 2-regular (cycle)
    # Girth of G□H >= girth(G) + girth(H) - 1 is not guaranteed, but let's try
    
    for deg_g, deg_h in [(3, 2), (4, 1), (2, 3), (1, 4)]:
        if deg_g + deg_h != k:
            continue
        
        # Try small cycle sizes for the 2-regular factor
        if deg_h == 2:
            # H is a cycle of length m
            for m in range(3, 50):
                if time.time() > deadline:
                    break
                # G needs to be deg_g-regular with enough vertices
                # G□H has n_G * m vertices, deg_g + deg_h = k regular
                # Girth: at least min(girth(G), m) but can be larger
                # We need girth(G□H) >= g
                # A cycle in G□H of length L projects to closed walks of length L in G and H
                # The H-projection is a closed walk on a cycle, so it must have length divisible by... 
                # Actually on a cycle of length m, a closed walk of length L exists iff L >= m and L has the same parity as... 
                # For a cycle C_m, there's a closed walk of length L iff L >= m (for L >= m, you can go around)
                # Actually for C_m, closed walks of length L exist for all L >= m except when m is even and L is odd and L < 2m
                # This is getting complex. Let's just try and verify.
                
                # For the 3-regular factor, try small graphs
                # Smallest 3-regular graphs: K4 (4 vertices, girth 3), K3,3 (6 vertices, girth 4), etc.
                # We need girth(G) large enough
                for n_g in range(deg_g + 1, 30):
                    if time.time() > deadline:
                        break
                    if n_g * deg_g % 2 != 0:
                        continue
                    
                    # Try to find a deg_g-regular graph on n_g vertices with large girth
                    # Use random generation
                    for attempt in range(10):
                        if time.time() > deadline:
                            break
                        g_edges, g_adj = make_random_k_reg(n_g, deg_g)
                        if g_edges is None:
                            continue
                        
                        # Compute girth of G
                        girth_g = compute_girth(g_adj, n_g)
                        
                        # Build Cartesian product G□H where H is cycle C_m
                        # Vertices: (u, v) where u in G, v in C_m
                        # Edges: ((u,v),(u',v)) if (u,u') in E(G) OR ((u,v),(u,v')) if (v,v') in E(H)
                        n_total = n_g * m
                        prod_edges = []
                        prod_adj = defaultdict(set)
                        
                        for u in range(n_g):
                            for v in range(m):
                                idx1 = u * m + v
                                # G-edges
                                for u2 in g_adj[u]:
                                    idx2 = u2 * m + v
                                    prod_adj[idx1].add(idx2)
                                    prod_adj[idx2].add(idx1)
                                # H-edges (cycle)
                                v_next = (v + 1) % m
                                v_prev = (v - 1) % m
                                idx_next = u * m + v_next
                                idx_prev = u * m + v_prev
                                prod_adj[idx1].add(idx_next)
                                prod_adj[idx_next].add(idx1)
                                prod_adj[idx1].add(idx_prev)
                                prod_adj[idx_prev].add(idx1)
                        
                        # Check degree
                        if not all(len(prod_adj[v]) == k for v in range(n_total)):
                            continue
                        
                        # Compute girth of product
                        girth_prod = compute_girth(prod_adj, n_total)
                        
                        if girth_prod >= g:
                            # Build edge list
                            edge_set = set()
                            for u in range(n_g):
                                for v in range(m):
                                    idx1 = u * m + v
                                    for u2 in g_adj[u]:
                                        idx2 = u2 * m + v
                                        edge = (min(idx1, idx2), max(idx1, idx2))
                                        if edge not in edge_set:
                                            edge_set.add(edge)
                                            prod_edges.append(edge)
                                    v_next = (v + 1) % m
                                    idx_next = u * m + v_next
                                    edge = (min(idx1, idx_next), max(idx1, idx_next))
                                    if edge not in edge_set:
                                        edge_set.add(edge)
                                        prod_edges.append(edge)
                            
                            if best_result is None or n_total < best_result[0]:
                                best_result = (n_total, prod_edges)
    
    return best_result

def compute_girth(adj, n):
    """Compute the girth (shortest cycle) of a graph using BFS from each vertex."""
    min_girth = float('inf')
    for start in range(n):
        # BFS tracking parent to avoid backtracking
        dist = [-1] * n
        parent = [-1] * n
        queue = deque()
        dist[start] = 0
        queue.append(start)
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth <= 3:
                        return min_girth
    
    return min_girth if min_girth < float('inf') else n

def make_random_k_reg(n, k):
    """Generate a random k-regular graph on n vertices."""
    if n * k % 2 != 0:
        return None, None
    for _ in range(50):
        stubs = []
        for v in range(n):
            stubs.extend([v] * k)
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u,v), max(u,v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        if valid:
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            return list(edges), adj
    return None, None

def construct_voltage_lift_optimized(k, g, deadline):
    """
    Optimized voltage graph lift search using DP for girth check.
    """

    def make_circulant_k_reg(n, k):
        """Generate a circulant k-regular graph on n vertices."""
        if k % 2 != 0:
            if n % 2 == 0:
                if (k - 1) // 2 + 1 > n // 2:
                    return None, None
                diffs = list(range(1, (k - 1) // 2 + 1)) + [n // 2]
            else:
                if k >= n:
                    return None, None
                diffs = list(range(1, (k + 1) // 2 + 1))
                if len(diffs) * 2 > n - 1:
                    return None, None
        else:
            if k >= n:
                return None, None
            diffs = list(range(1, k // 2 + 1))
            if len(diffs) * 2 > n - 1:
                return None, None

        edges = set()
        for d in diffs:
            for v in range(n):
                u = (v + d) % n
                edges.add((min(v, u), max(v, u)))

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        if all(len(adj[v]) == k for v in range(n)):
            return list(edges), adj
        return None, None

    def make_random_k_reg(n, k):
        """Generate a random k-regular graph on n vertices."""
        if n * k % 2 != 0:
            return None, None
        for _ in range(20):
            stubs = []
            for v in range(n):
                stubs.extend([v] * k)
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                edge = (min(u,v), max(u,v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
            if valid:
                adj = defaultdict(set)
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                return list(edges), adj
        return None, None

    def check_girth_dp(base_adj, base_edges, voltages, m, g):
        """
        Check if cover has girth >= g using DP on non-backtracking walks.
        Returns True if no closed walk of length < g has voltage sum 0.
        """
        n_base = len(base_adj)
        neighbors = [list(base_adj[v]) for v in range(n_base)]

        # Precompute voltage lookup for speed
        vol_matrix = [[0] * n_base for _ in range(n_base)]
        for (u, v) in base_edges:
            val = voltages[(u, v)]
            vol_matrix[u][v] = val
            vol_matrix[v][u] = (-val) % m

        for L in range(3, g):
            S = {}
            for s in range(n_base):
                for v in neighbors[s]:
                    vol = vol_matrix[s][v]
                    key = (v, s, s)
                    if key in S:
                        S[key].add(vol)
                    else:
                        S[key] = {vol}

            for step in range(2, L + 1):
                S_next = {}
                for (u, v, s), sums in S.items():
                    for w in neighbors[u]:
                        if w == v:
                            continue
                        vol = vol_matrix[u][w]
                        new_sums = set()
                        for x in sums:
                            new_sums.add((x + vol) % m)
                        key = (w, u, s)
                        if key in S_next:
                            S_next[key].update(new_sums)
                        else:
                            S_next[key] = new_sums
                S = S_next

            for (u, v, s), sums in S.items():
                if u == s:
                    if 0 in sums:
                        return False

        return True

    best_result = None

    for base_n in range(10, 40):
        if time.time() > deadline:
            break

        base_edges, base_adj = make_circulant_k_reg(base_n, k)
        if base_edges is None:
            base_edges, base_adj = make_random_k_reg(base_n, k)
        if base_edges is None:
            continue

        if not all(len(base_adj[v]) == k for v in range(base_n)):
            continue

        for m in range(2, 100):
            if time.time() > deadline:
                break

            total_n = m * base_n

            if best_result is not None and total_n >= best_result[0]:
                break

            found_valid = False
            for volt_attempt in range(20):
                if time.time() > deadline:
                    break

                voltages = {}
                for (u, v) in base_edges:
                    val = random.randint(0, m - 1)
                    voltages[(u, v)] = val
                    voltages[(v, u)] = (-val) % m

                if check_girth_dp(base_adj, base_edges, voltages, m, g):
                    cover_edges = []
                    for (u, v) in base_edges:
                        vu = voltages[(u, v)]
                        for t in range(m):
                            u_cover = u * m + t
                            v_cover = v * m + (t + vu) % m
                            cover_edges.append((u_cover, v_cover))

                    if best_result is None or total_n < best_result[0]:
                        best_result = (total_n, cover_edges)
                    found_valid = True
                    break

            if found_valid:
                pass

    return best_result

def construct_greedy_tree(k, g, n_target):
    """
    Try to construct a k-regular graph of girth g on exactly n_target vertices.
    """
    r = (g - 1) // 2
    moore = 1
    for i in range(1, r + 1):
        moore += k * ((k - 1) ** (i - 1))

    if n_target != moore:
        return None, None

    parent = [-1] * n_target
    level = [0] * n_target
    branch = [0] * n_target

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

    if next_id != n_target:
        return None, None

    leaves = [v for v in range(n_target) if level[v] == r]

    adj = defaultdict(set)
    for v in range(1, n_target):
        u = parent[v]
        adj[u].add(v)
        adj[v].add(u)

    deg_needed = {v: k - len(adj[v]) for v in range(n_target)}

    random.shuffle(leaves)

    for attempt in range(50):
        edges_to_add = []
        used_edges = set()
        success = True

        stubs = []
        for v in leaves:
            for _ in range(deg_needed[v]):
                stubs.append(v)

        random.shuffle(stubs)

        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                success = False
                break
            if branch[u] == branch[v]:
                success = False
                break
            if v in adj[u]:
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
        for v in range(1, n_target):
            final_edges.append((parent[v], v))
        final_edges.extend(edges_to_add)

        deg_check = defaultdict(int)
        for u, v in final_edges:
            deg_check[u] += 1
            deg_check[v] += 1
        if all(deg_check[v] == k for v in range(n_target)):
            return n_target, final_edges

    return None, None
