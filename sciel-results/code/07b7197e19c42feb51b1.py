import random
import time
import sys
from collections import deque

def construct(k, g):
    # Fallback: The Lazebnik-Ustimenko-Woldar construction from the seed program
    def l_u_w_construct(k, g):
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

    # New approach: Random regular graph generation with girth verification
    def generate_random_regular(n, k, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if seed is not None:
            random.seed(seed)
        
        # Check feasibility: n*k must be even
        if n * k % 2 != 0:
            return None
        
        # Create stubs
        stubs = []
        for v in range(n):
            for _ in range(k):
                stubs.append(v)
        
        random.shuffle(stubs)
        
        # Pair up stubs to form edges
        adj = [[] for _ in range(n)]
        edges = []
        edge_set = set()
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                # Self-loop, reject this pairing (retry with different shuffle)
                return None
            if (u, v) in edge_set or (v, u) in edge_set:
                # Multi-edge, reject
                return None
            edge_set.add((min(u, v), max(u, v)))
            adj[u].append(v)
            adj[v].append(u)
            edges.append((u, v))
        
        # Verify all vertices have degree exactly k
        for v in range(n):
            if len(adj[v]) != k:
                return None
        
        return adj, edges

    def check_girth(adj, g):
        """Check if graph has girth >= g using non-backtracking walks from every vertex."""
        n = len(adj)
        for start in range(n):
            # Non-backtracking walk enumeration
            # State: (current, previous, depth)
            # We want to detect any cycle of length < g
            # A cycle of length L < g corresponds to a non-backtracking walk
            # of length L that returns to start
            
            visited = {}  # (node, prev) -> min_depth reached
            queue = deque()
            
            for nei in adj[start]:
                queue.append((nei, start, 1))
                visited[(nei, start)] = 1
            
            while queue:
                curr, prev, depth = queue.popleft()
                
                # If we've reached start again with depth < g, we found a short cycle
                if curr == start and depth < g:
                    return False
                
                # If depth >= g, no need to continue
                if depth >= g:
                    continue
                
                for nei in adj[curr]:
                    if nei == prev:
                        continue  # Non-backtracking
                    key = (nei, curr)
                    if key in visited:
                        if visited[key] < depth + 1:
                            continue  # Already visited with shorter depth
                        # Otherwise, we found a cycle
                        # The cycle length is visited[key] + depth + 1
                        # Actually, if we revisit (nei, curr) at a later depth,
                        # it means there's a cycle. But we need to check if it's short.
                        # Simpler: if we reach (start, some_prev) with depth < g, fail.
                        # But we already check curr == start above.
                        # For non-backtracking, if we revisit a state, it implies a cycle.
                        # The cycle length is (depth + 1) - visited[key] + 1? No.
                        # Let's just track if we can reach start with depth < g.
                        continue
                    visited[key] = depth + 1
                    queue.append((nei, curr, depth + 1))
        
        return True

    def try_random_regular(n, k, g, timeout=10):
        """Try to find a random k-regular graph on n vertices with girth >= g."""
        start_time = time.time()
        attempts = 0
        max_attempts = 50000
        
        while time.time() - start_time < timeout:
            attempts += 1
            if attempts > max_attempts:
                break
            
            result = generate_random_regular(n, k)
            if result is None:
                continue
            
            adj, edges = result
            
            # Check girth
            if check_girth(adj, g):
                return n, edges
        
        return None

    # If k and g are such that the problem is very hard, we might just use the fallback.
    # But for k=5, g=9, we can try random regular graphs.
    
    # Time budget management
    total_time_limit = 110  # Leave 10s for fallback
    
    if k == 5 and g == 9:
        # Known best is 1116. Try values around there.
        n_trials = [1116, 1120, 1130, 1140, 1150]
        for n in n_trials:
            res = try_random_regular(n, k, g, timeout=15)
            if res:
                return res
    else:
        # For other parameters, compute a reasonable starting n
        moore_bound = 0
        if g % 2 == 1:
            r = (g - 3) // 2
            for i in range(r):
                moore_bound += (k - 1) ** i
            moore_bound *= k
        else:
            r = g // 2 - 1
            for i in range(r + 1):
                moore_bound += (k - 1) ** i
            moore_bound *= 2
        
        n_start = max(10, moore_bound)
        for n in [n_start, n_start + 50, n_start + 100]:
            res = try_random_regular(n, k, g, timeout=10)
            if res:
                return res

    # Fallback
    return l_u_w_construct(k, g)
