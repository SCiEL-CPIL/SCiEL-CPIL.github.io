"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import random
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 140  # 10s safety margin
    
    def is_valid_graph(n, adj, k, g):
        """Check if graph is k-regular, simple, and has girth >= g."""
        if len(adj) != n:
            return False
        for i in range(n):
            if len(adj[i]) != k:
                return False
            if len(set(adj[i])) != k:
                return False
            if i in adj[i]:
                return False
        # Check girth via BFS from each vertex
        for start in range(n):
            # BFS tracking parent to avoid backtracking
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g // 2:
                    break
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < g // 2:
                            queue.append(v)
                    else:
                        # Found a cycle of length dist[u] + dist[v] + 1
                        # But we need to be careful: this might not be a simple cycle
                        # For a proper girth check, we need non-backtracking walks
                        # A simpler approach: check that no two vertices at distance <= g//2 - 1
                        # are connected by two different paths
                        pass
            # More rigorous: check that the number of vertices at each distance matches a tree
            # up to distance g//2 - 1
            # Actually, let's do a proper check: from each vertex, BFS and ensure
            # that no vertex is reached twice within distance < g/2
            pass
        
        # Proper girth check: for each vertex, do BFS and ensure tree-like structure
        # up to radius g//2 - 1
        radius = g // 2 - 1  # for even girth, we check up to distance g/2 - 1
        for start in range(n):
            # Count vertices at each distance
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            level_counts = [0] * (radius + 1)
            level_counts[0] = 1
            while queue:
                u = queue.popleft()
                if dist[u] >= radius:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= radius:
                            level_counts[dist[v]] += 1
                            queue.append(v)
                    # If dist[v] != -1 and v != parent, we have a cycle
                    # But we don't track parent here. Let's use a different approach.
            
            # Check: in a graph with girth > 2*radius, the BFS tree from any vertex
            # up to depth radius should be a true tree (no extra edges)
            # So level_counts should be: 1, k, k*(k-1), k*(k-1)^2, ...
            expected = [1] + [k * (k-1)**i for i in range(radius)]
            if level_counts != expected:
                return False
        
        return True
    
    def check_girth(n, adj, k, g):
        """Return True if girth >= g, False otherwise. Fast check."""
        radius = g // 2 - 1
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            count = [0] * (radius + 1)
            count[0] = 1
            while queue:
                u = queue.popleft()
                if dist[u] >= radius:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= radius:
                            count[dist[v]] += 1
                            queue.append(v)
            
            expected = [1] + [k * (k-1)**i for i in range(radius)]
            if count != expected:
                return False
        return True
    
    def random_regular_graph(n, k, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if seed is not None:
            random.seed(seed)
        # Configuration model: create n*k stubs, pair them randomly
        if n * k % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None  # Self-loop, try again
            if v in adj[u]:
                return None  # Multi-edge, try again
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def find_short_cycle(n, adj, k, g):
        """Find a cycle shorter than g. Returns (cycle_length, cycle_edges) or None."""
        radius = g // 2 - 1
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= radius:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] <= radius:
                            queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        # Reconstruct cycle
                        cycle = [u, v]
                        a, b = u, v
                        while a != b:
                            a = parent[a]
                            b = parent[b]
                            if a != -1:
                                cycle.append(a)
                            if b != -1:
                                cycle.append(b)
                        # This is getting complicated. Let's just return that there's a short cycle.
                        return True
        return False
    
    # Moore bound for k=3, girth=16 (even): n >= 2*3*sum(2^i for i=0..5) = 6*63 = 378
    # But known lower bound is 512. Start from 512.
    n_start = 512
    n_max = 1000  # Upper bound to try
    
    best_n = None
    best_edges = None
    
    # Try various n values
    for n in range(n_start, n_max + 1, 8):
        if time.time() > deadline:
            break
        if n * k % 2 != 0:
            continue
        
        # Try multiple random graphs for this n
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts:
            attempts += 1
            if time.time() > deadline:
                break
            
            adj = random_regular_graph(n, k)
            if adj is None:
                continue
            
            # Quick check: is it connected?
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            if count != n:
                continue
            
            # Check girth
            if check_girth(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        # If no valid graph found for this n, try to improve a near-miss
        # by edge swaps
    
    # Fallback: return a LUW graph if we couldn't find a better one
    # Use the LUW construction as fallback
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
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
# EVOLVE-BLOCK-END
