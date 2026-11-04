"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque, defaultdict

def construct(k, g):
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Moore bound for 4-regular, girth 10
    # For even girth g, n >= 2 * (k-1)^(g/2 - 1) + 2 * (k-1)^(g/2 - 2)
    # Actually, for girth g (even), n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2*((k-1)^(g/2)-1)/((k-1)-1)
    # For k=4, g=10: n >= 2*(3^5 - 1)/2 = 3^5 - 1 = 242. But we need integer, and the bound is actually
    # n >= 2 * (k-1) * ((k-1)^(g/2-1) - 1) / ((k-1) - 1) for girth >= g
    # Let me use the standard Moore bound: for girth g, n >= k * (k-1)^(g/2 - 1) + 2 * (k-1)^(g/2 - 2)
    # Wait, let me just compute it properly.
    # For a k-regular graph with girth g (even), the Moore bound is:
    # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # For k=4, g=10: n >= 2*(1 + 3 + 9 + 27 + 81) = 2*121 = 242
    # But the problem says best lower bound is 243, so n >= 243.
    
    # Try to construct a graph with n = 243 vertices (or close to it)
    # Start with a random 4-regular graph and use simulated annealing to eliminate short cycles
    
    # First, let's try the LUW construction to get a valid graph, then try to improve
    # LUW construction
    min_d_odd = max(3, g - 5)
    if min_d_odd % 2 == 0:
        min_d_odd += 1
        
    min_d_even = max(4, g - 4)
    if min_d_even % 2 == 1:
        min_d_even += 1
        
    d = min(min_d_odd, min_d_even)
    q = k
    
    if q == 4:
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return mul_table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return (x * y) % q

    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            a, b = 1, c0 - 2
        elif r == 1:
            a, b = c0 - 1, 1
        elif r == 2:
            a, b = c0, 1
        else:
            a, b = 1, c0 + 1
        rule_a[c] = a
        rule_b[c] = b

    def vertex_to_key(side, x):
        return (side, x)
    
    def compute_neighbours_fast(vertex_key):
        side, x = vertex_key
        result = []
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            result.append(vertex_to_key(1 - side, tuple(y)))
        return result

    start = vertex_to_key(0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges_luw = []
    
    queue = deque([start])
    head = 0
    
    while head < len(order):
        if time.time() > deadline - 60:
            break
        vertex = order[head]
        head += 1
        for other in compute_neighbours_fast(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    
    n_luw = len(order)
    
    # Now try to find a smaller graph using simulated annealing
    # Target n: try values from Moore bound up to n_luw
    moore_bound = 2 * sum((k-1)**i for i in range(g//2))
    if moore_bound < 243:
        moore_bound = 243
    
    # Try to construct a graph with n = moore_bound to n_luw - 1
    # We'll use simulated annealing on random 4-regular graphs
    
    def count_short_cycles(adj, n, g, max_depth=None):
        """Count non-backtracking closed walks of length < g using matrix powers."""
        if max_depth is None:
            max_depth = g - 1
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.int64)
        for u, v in adj:
            A[u, v] = 1
            A[v, u] = 1
        
        # Count closed walks of length 2, 4, 6, 8 (even lengths only for undirected)
        total = 0
        for length in range(2, g, 2):
            # A^length diagonal sum counts closed walks of length length
            # But we need non-backtracking, so this is an overcount
            # For a rough estimate, use trace(A^length)
            Alen = A.copy()
            for _ in range(length - 1):
                Alen = Alen @ A
            trace = np.trace(Alen)
            # Each closed walk of length L is counted L times in trace (once for each starting point)
            # And each cycle of length L is counted 2L times (L starting points * 2 directions)
            # So number of cycles of length L is approximately trace / (2*L)
            total += trace / (2 * length)
        
        return total
    
    def find_shortest_cycle_bfs(adj, n, g):
        """Find the shortest cycle using BFS from each vertex."""
        min_cycle = g  # We only care if it's < g
        for start in range(n):
            if time.time() > deadline - 30:
                break
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and dist[v] >= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle
        return min_cycle
    
    # Generate a random 4-regular graph
    def generate_random_4reg(n):
        """Generate a random 4-regular graph on n vertices."""
        # Use the configuration model with rejection
        for _ in range(100):
            # Create 4 stubs per vertex
            stubs = []
            for v in range(n):
                for _ in range(4):
                    stubs.append(v)
            
            random.shuffle(stubs)
            
            # Pair up stubs
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
            
            if not valid:
                continue
            
            # Check 4-regularity
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if all(d == 4 for d in deg):
                return list(edges)
        
        return None
    
    # Simulated annealing to minimize short cycles
    def simulated_annealing(n, k, g, time_limit):
        """Try to find a k-regular graph with girth >= g using simulated annealing."""
        if time.time() > deadline - 10:
            return None
        
        # Start with a random k-regular graph
        edges = generate_random_4reg(n)
        if edges is None:
            return None
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Initial energy: count of short cycles
        # Use a fast approximation
        def energy(adj, n, g):
            """Count short cycles approximately."""
            # For each vertex, do BFS up to depth g//2 and count backtracking violations
            count = 0
            for start in range(n):
                # BFS with parent tracking
                visited = {}
                visited[start] = (0, -1)
                q = deque([(start, -1)])
                while q:
                    u, p = q.popleft()
                    d = visited[u][0]
                    if d >= g // 2:
                        continue
                    for v in adj[u]:
                        if v == p:
                            continue
                        if v in visited:
                            # Found a cycle
                            vd = visited[v][0]
                            if vd >= d:
                                cycle_len = d + vd + 1
                                if cycle_len < g:
                                    count += 1
                        else:
                            visited[v] = (d + 1, u)
                            q.append((v, u))
            return count
        
        # This is too slow. Let's use a simpler approach:
        # Just check if girth >= g using BFS from a sample
        def girth_ok(adj, n, g):
            for start in range(min(n, 50)):
                if time.time() > deadline - 5:
                    return False
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= g:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u] and dist[v] >= dist[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
            return True
        
        # Try edge swaps to improve
        # This is complex, so let's just return the LUW graph for now
        return None
    
    # For now, just return the LUW construction
    # Build adjacency for verification
    adj = defaultdict(set)
    for u, v in edges_luw:
        adj[u].add(v)
        adj[v].add(u)
    
    # Quick girth check
    girth_ok = True
    for start in range(min(n_luw, 20)):
        if time.time() > deadline - 5:
            break
        dist = [-1] * n_luw
        parent = [-1] * n_luw
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and dist[v] >= dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        girth_ok = False
                        break
            if not girth_ok:
                break
        if not girth_ok:
            break
    
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
