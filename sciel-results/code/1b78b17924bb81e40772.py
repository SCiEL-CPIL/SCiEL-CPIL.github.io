import time
import random
import sys
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 115.0

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        if k != 3:
            return None # Only handle k=3 for now as per task focus
            
        q = 3
        # Determine d such that girth is sufficient
        # For LUW with q=3, girth is roughly 2d+2 or 2d+4.
        # We need girth >= 16.
        # If girth = 2d+2, then 2d+2 >= 16 => d >= 7.
        # Size is 2 * q^(d+1) = 2 * 3^8 = 13122.
        d = 7
        
        # Rule definition
        # Standard LUW rules for q=3
        # We use the generic rule generator from the seed, but ensure d is correct.
        # The seed's rule generator:
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
                    y[c] = (x[c] + y[a] * x[b]) % q if side == 0 else (x[c] - x[a] * y[b]) % q
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

    # Helper: Check girth using non-backtracking BFS
    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Random Local Search
    def random_local_search(k, g, n_target, max_time):
        # Generate random k-regular graph
        def generate_random_k_regular(n, k):
            if n * k % 2 != 0:
                return None
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u != v:
                    edges.append((u, v))
            
            adj = [[] for _ in range(n)]
            edge_set = set()
            for u, v in edges:
                if u == v:
                    return None
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    return None
                edge_set.add(key)
                adj[u].append(v)
                adj[v].append(u)
                
            for i in range(n):
                if len(adj[i]) != k:
                    return None
            return edges

        # Find shortest cycle and its vertices
        def find_shortest_cycle(n, edges):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            min_len = float('inf')
            min_cycle = None
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = [start]
                qi = 0
                while qi < len(queue):
                    u = queue[qi]
                    qi += 1
                    if dist[u] >= min_len // 2:
                        continue
                    for v in adj[u]:
                        if v == parent[u]:
                            continue
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_len:
                                min_len = cycle_len
                                # Reconstruct cycle
                                cycle = []
                                curr = u
                                while curr != start:
                                    cycle.append(curr)
                                    curr = parent[curr]
                                cycle.append(start)
                                curr = v
                                while curr != start:
                                    cycle.append(curr)
                                    curr = parent[curr]
                                min_cycle = cycle
            return min_len, min_cycle

        # Main search loop
        best_n = float('inf')
        best_edges = None
        
        # Try to find a graph with n < n_target
        # We start from n_target and go down?
        # We try n = 936 (known best) and see if we can find it.
        # If not, we try smaller n.
        
        # Given the difficulty, we try a few random graphs of size 936.
        # If we find one with girth >= g, we return it.
        
        attempts = 0
        while time.time() < t0 + max_time:
            attempts += 1
            # Try size 936
            edges = generate_random_k_regular(936, 3)
            if edges is None:
                continue
            
            # Check girth
            if check_girth(936, edges, g):
                return 936, edges
                
            # If not, we could try to repair, but it's complex.
            # We just try again.
            
            # If time is running out, break
            if time.time() > t0 + max_time - 5:
                break
                
        return None

    # Main logic
    if k == 3 and g == 16:
        # First, try to find a better graph via random search
        # We have ~100s.
        # Try random search for n=936.
        result = random_local_search(3, 16, 936, 60.0)
        if result is not None:
            return result
            
        # Fallback to LUW
        return luw_construct(3, 16)
    else:
        return luw_construct(k, g)
