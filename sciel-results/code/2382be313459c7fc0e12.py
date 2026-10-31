import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118.0

    def seed_construct(k, g):
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

    def check_girth(adj, n, g):
        """Verify no cycle shorter than g via non-backtracking walks from every vertex."""
        half = g // 2
        for s in range(n):
            seen = {}
            dq = deque()
            for nb in adj[s]:
                key = nb * n + s
                seen[key] = 1
                dq.append((nb, s, 1))
            while dq:
                u, pu, dist = dq.popleft()
                if dist >= half:
                    continue
                nd = dist + 1
                for v in adj[u]:
                    if v == pu:
                        continue
                    if v == s:
                        if nd < g:
                            return False
                    else:
                        key = v * n + u
                        if key in seen:
                            if nd < g:
                                return False
                        else:
                            seen[key] = nd
                            dq.append((v, u, nd))
        return True

    def build_tree_3reg(g, seed):
        """Build a 3-regular graph by growing a tree of depth h from a root,
        then adding edges among leaves to complete 3-regularity."""
        rng = random.Random(seed)
        
        # Need: for any two leaves connected by an added edge, tree_dist >= g-1
        # If we only connect leaves from different root-subtrees, min tree_dist = 2*h
        # Need 2*h >= g-1 => h >= ceil((g-1)/2)
        h = (g - 1 + 1) // 2  # ceil((g-1)/2)
        if 2 * h < g - 1:
            h += 1
        
        # Build the tree
        adj = []
        parent = []
        node_depth = []
        children_list = []
        
        # Add root
        adj.append([])
        parent.append(-1)
        node_depth.append(0)
        children_list.append([])
        
        # Root's children: 3 nodes
        for i in range(3):
            new_id = len(adj)
            adj.append([])
            parent.append(0)
            node_depth.append(1)
            children_list.append([])
            children_list[0].append(new_id)
            adj[0].append(new_id)
            adj[new_id].append(0)
        
        # Levels 2 through h
        for d in range(2, h + 1):
            level_nodes = [i for i in range(len(adj)) if node_depth[i] == d - 1]
            for node in level_nodes:
                for _ in range(2):
                    new_id = len(adj)
                    adj.append([])
                    parent.append(node)
                    node_depth.append(d)
                    children_list.append([])
                    children_list[node].append(new_id)
                    adj[node].append(new_id)
                    adj[new_id].append(node)
        
        n_tree = len(adj)
        leaves = [i for i in range(n_tree) if node_depth[i] == h]
        num_leaves = len(leaves)
        
        # Group leaves by which root-child's subtree they're in
        root_children = children_list[0]
        
        leaf_groups = {rc: [] for rc in root_children}
        for leaf in leaves:
            node = leaf
            while node_depth[node] > 1:
                node = parent[node]
            leaf_groups[node].append(leaf)
        
        # Shuffle leaves within each group
        for rc in root_children:
            rng.shuffle(leaf_groups[rc])
        
        # Create a cycle: interleave the three groups
        s = num_leaves // 3
        if s * 3 != num_leaves:
            return None
        
        cycle_order = []
        for i in range(s):
            cycle_order.append(leaf_groups[root_children[0]][i])
            cycle_order.append(leaf_groups[root_children[1]][i])
            cycle_order.append(leaf_groups[root_children[2]][i])
        
        # Add edges along the cycle
        for i in range(len(cycle_order)):
            u = cycle_order[i]
            v = cycle_order[(i + 1) % len(cycle_order)]
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify 3-regular
        for i in range(n_tree):
            if len(adj[i]) != 3:
                return None
        
        # Check for multi-edges
        for i in range(n_tree):
            if len(adj[i]) != len(set(adj[i])):
                return None
        
        # Build edge list
        edges = []
        for u in range(n_tree):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        return n_tree, edges, adj

    def build_petersen_lift(m, seed):
        """Build a voltage graph lift of the Petersen graph over Z_m.
        Returns (n, adj, edges) or None if invalid."""
        rng = random.Random(seed)
        
        # Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        n = 10 * m
        adj = [set() for _ in range(n)]
        edges = []
        
        # Assign random voltages to each edge
        voltages = {}
        for u, v in petersen_edges:
            vol = rng.randint(0, m - 1)
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m
        
        for u, v in petersen_edges:
            vol = voltages[(u, v)]
            for i in range(m):
                a = u * m + i
                b = v * m + (i + vol) % m
                if b not in adj[a]:
                    adj[a].add(b)
                    adj[b].add(a)
                    edges.append((a, b))
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        
        # Convert sets to lists for consistent access
        adj_list = [list(s) for s in adj]
        
        # Check no multi-edges
        for i in range(n):
            if len(adj_list[i]) != len(set(adj_list[i])):
                return None
        
        return n, adj_list, edges

    def build_lift_from_base(base_adj, m, seed):
        """Build a voltage graph lift of an arbitrary 3-regular base graph over Z_m."""
        rng = random.Random(seed)
        n_base = len(base_adj)
        n = n_base * m
        adj = [set() for _ in range(n)]
        edges = []
        
        # Get unique edges from base
        base_edges = set()
        for u in range(n_base):
            for v in base_adj[u]:
                if u < v:
                    base_edges.add((u, v))
        
        # Assign random voltages
        voltages = {}
        for u, v in base_edges:
            vol = rng.randint(0, m - 1)
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m
        
        for u, v in base_edges:
            vol = voltages[(u, v)]
            for i in range(m):
                a = u * m + i
                b = v * m + (i + vol) % m
                if b not in adj[a]:
                    adj[a].add(b)
                    adj[b].add(a)
                    edges.append((a, b))
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        
        adj_list = [list(s) for s in adj]
        
        # Check no multi-edges
        for i in range(n):
            if len(adj_list[i]) != len(set(adj_list[i])):
                return None
        
        return n, adj_list, edges

    # Strategy:
    # 1. Try tree-based construction
    # 2. Try Petersen graph lifts with various m
    # 3. Try random 3-regular graphs
    # 4. Fallback to seed construction

    # For g=16: Moore bound = 2*(2^8 - 1) = 510
    # Tree of depth 8 gives 766 vertices
    
    # Try tree-based construction
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        result = build_tree_3reg(g, seed)
        if result is not None:
            n, edges, adj = result
            if check_girth(adj, n, g):
                return n, edges

    # Try Petersen graph lifts
    # n = 10*m, need n >= 510, so m >= 51
    if g == 16:
        m_min = 51
    else:
        # Moore bound for k=3, girth g
        if g % 2 == 0:
            moore = 2 * (2**(g//2) - 1)
        else:
            moore = 2 * (2**(g//2) - 1) + 2
        m_min = max(1, (moore + 9) // 10)
    
    m_candidates = list(range(m_min, m_min + 50))
    
    for m in m_candidates:
        if time.time() > deadline - 5:
            break
        n = 10 * m
        # Try multiple random voltage assignments
        for attempt in range(20):
            if time.time() > deadline - 3:
                break
            result = build_petersen_lift(m, seed=attempt)
            if result is None:
                continue
            n, adj, edges = result
            if check_girth(adj, n, g):
                return n, edges

    # Try random 3-regular graphs with n close to Moore bound
    if g == 16:
        moore = 510
    else:
        if g % 2 == 0:
            moore = 2 * (2**(g//2) - 1)
        else:
            moore = 2 * (2**(g//2) - 1) + 2
    
    n_try = moore
    if n_try % 2 != 0:
        n_try += 1
    
    max_n = min(n_try + 200, 1000)
    
    while n_try <= max_n and time.time() < deadline - 10:
        if n_try % 2 != 0:
            n_try += 2
            continue
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            # Generate random 3-regular graph
            rng = random.Random(attempt * 1000 + n_try)
            stubs = []
            for v in range(n_try):
                for _ in range(3):
                    stubs.append(v)
            rng.shuffle(stubs)
            adj = [set() for _ in range(n_try)]
            edges = []
            valid = True
            for i in range(0, n_try * 3, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
            if not valid:
                continue
            # Check connectedness
            visited = set([0])
            stack = [0]
            while stack:
                v = stack.pop()
                for w in adj[v]:
                    if w not in visited:
                        visited.add(w)
                        stack.append(w)
            if len(visited) != n_try:
                continue
            adj_list = [list(s) for s in adj]
            if check_girth(adj_list, n_try, g):
                return n_try, edges
        n_try += 64

    # Fallback to seed construction
    return seed_construct(k, g)
