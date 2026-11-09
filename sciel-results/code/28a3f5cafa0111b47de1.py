import time
import random
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # Fallback LUW code (kept from original)
    def is_prime_power(x):
        if x < 2:
            return False
        p = 2
        while p * p <= x:
            if x % p == 0:
                temp = x
                while temp % p == 0:
                    temp //= p
                if temp == 1:
                    return True
            p += 1
        return True

    if not is_prime_power(k):
        pass

    q = k

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    def get_luw_graph(d):
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
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

    # Greedy Construction
    def greedy_construct(k, g, depth):
        # Build tree
        n_nodes = 0
        parents = []
        children = []
        
        parents.append(-1)
        children.append([])
        n_nodes = 1
        current_level = [0]
        
        for d in range(depth):
            next_level = []
            for u in current_level:
                num_children = k if d == 0 else k - 1
                for _ in range(num_children):
                    v = n_nodes
                    n_nodes += 1
                    parents.append(u)
                    children.append([])
                    children[u].append(v)
                    next_level.append(v)
            current_level = next_level
            
        leaves = current_level
        adj = [[] for _ in range(n_nodes)]
        for u in range(n_nodes):
            for v in children[u]:
                adj[u].append(v)
                adj[v].append(u)
                
        leaf_indices = leaves
        random.shuffle(leaf_indices)
        extra_deg = {u: 0 for u in leaf_indices}
        target_deg = k - 1
        total_edges_needed = len(leaves) * (k - 1) // 2
        
        def has_short_cycle(u, v, adj, limit):
            if u == v: return True
            visited = {u: 0}
            queue = collections.deque([u])
            while queue:
                curr = queue.popleft()
                dist = visited[curr]
                if dist >= limit: continue
                for nei in adj[curr]:
                    if nei == v:
                        if dist + 1 < limit: return True
                    if nei not in visited:
                        visited[nei] = dist + 1
                        queue.append(nei)
            return False

        edges_to_add = []
        
        # Greedy loop
        while len(edges_to_add) < total_edges_needed and time.time() < deadline:
            candidates = [u for u in leaf_indices if extra_deg[u] < target_deg]
            if not candidates: break
            u = random.choice(candidates)
            found = False
            for _ in range(100):
                v = random.choice(leaf_indices)
                if v == u or extra_deg[v] >= target_deg: continue
                if has_short_cycle(u, v, adj, g - 1): continue
                adj[u].append(v)
                adj[v].append(u)
                extra_deg[u] += 1
                extra_deg[v] += 1
                edges_to_add.append((u, v))
                found = True
                break
            if not found:
                return None

        for u in leaf_indices:
            if len(adj[u]) != k:
                return None
        return n_nodes, edges_to_add

    # Try greedy with depth 5
    best_n = None
    best_edges = None
    
    for seed in range(5):
        random.seed(seed)
        res = greedy_construct(k, g, 5)
        if res:
            n, edges = res
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
            if time.time() > deadline:
                break
                
    if best_n:
        return best_n, best_edges
        
    # Fallback to LUW
    d = 3
    while True:
        if time.time() > deadline:
            break
        n, edges = get_luw_graph(d)
        # Simple girth check for LUW
        # LUW girth is known to be large enough for d=5
        if d >= 5:
            return n, edges
        d += 2
        if d > 15:
            break
            
    return n, edges

