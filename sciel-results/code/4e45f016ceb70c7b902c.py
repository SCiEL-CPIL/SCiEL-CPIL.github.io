import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2 + 1)
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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
        
        n = len(order)
        return n, edges

    def check_girth_fast(adj, target_girth):
        """
        Check if graph has girth >= target_girth.
        Uses BFS from each node up to depth max_depth = (target_girth - 1) // 2.
        If a node is reached by two different paths of length <= max_depth, 
        we might have a short cycle.
        Specifically, we track the parent in BFS. If we encounter a visited node 
        that is not the parent, we have a cycle.
        Cycle length = dist[u] + dist[v] + 1.
        We check if this length < target_girth.
        """
        n = len(adj)
        max_depth = (target_girth - 1) // 2
        
        for start in range(n):
            # BFS from start
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                if dist[u] > max_depth:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] <= max_depth:
                            q.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # The cycle goes start -> ... -> u -> v -> ... -> start
                        # Length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    def generate_random_k_regular(n, k):
        """
        Generate a random simple k-regular graph on n vertices using configuration model.
        Returns list of edges or None if fails (e.g., multi-edge/loop).
        """
        if n * k % 2 != 0:
            return None
        
        # Create stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        
        random.shuffle(stubs)
        
        # Pair stubs
        edges_set = set()
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                return None # Loop
            edge = (min(u, v), max(u, v))
            if edge in edges_set:
                return None # Multi-edge
            edges_set.add(edge)
            
        return list(edges_set)

    # Main logic
    if k == 4 and g == 10:
        # Moore bound for (4, 10)
        # n >= 1 + k * sum_{i=0}^{g/2 - 1} (k-1)^i
        # For g=10, g/2 = 5. Sum i=0 to 4.
        # 1 + 4 * (1 + 3 + 9 + 27) = 1 + 4 * 40 = 161?
        # Wait, Moore bound formula:
        # If g is even: n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        # g=10, g/2=5. Sum i=0 to 4: 1 + 3 + 9 + 27 + 81 = 121.
        # n >= 2 * 121 = 242.
        # So n_min = 242.
        
        n_min = 2 * sum((k-1)**i for i in range(g//2))
        
        # We know 320 is possible. Let's try to find something between n_min and 320.
        # Random search is probabilistic.
        # For n=320, probability of girth >= 10 is low but non-zero.
        # However, generating and checking is fast.
        
        best_n = 1024
        best_edges = None
        
        # Try a few random trials for n=320 first, as it's the known best.
        # If we find one, we return it.
        # If not, we might try n=330, etc., but 320 is the target.
        
        # Actually, just trying n=320 might not find a valid one quickly.
        # The space is huge.
        # Let's try a "Greedy" local search or just many random trials.
        
        trials = 0
        start_time = time.time()
        
        # Try n=320
        target_n = 320
        while time.time() < start_time + 50: # Give 50s to n=320
            edges = generate_random_k_regular(target_n, k)
            if edges:
                # Build adjacency
                adj = [[] for _ in range(target_n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                if check_girth_fast(adj, g):
                    return target_n, edges
            trials += 1
            if trials % 1000 == 0:
                pass # Debug print if needed
        
        # If not found in 50s, try other n values?
        # Or just fall back to LUW.
        # Trying n=300, 310 etc might not be better if 320 is the cage.
        # Actually, if 320 is the cage, no graph with n < 320 exists.
        # So searching n < 320 is futile.
        # Searching n > 320 is worse for score.
        
        # So if we didn't find a 320-vertex graph, we return LUW (1024).
        
        n_luw, edges_luw = luw_construct(k, g)
        return n_luw, edges_luw

    # General case: LUW
    return luw_construct(k, g)
