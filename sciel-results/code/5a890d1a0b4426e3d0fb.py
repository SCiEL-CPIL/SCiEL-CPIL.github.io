import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # 5s margin before 120s kill

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))

    def seed_construct(k, g):
        """LUW construction as fallback"""
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
            rc = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]

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
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Compute seed once for fallback
    seed_n, seed_edges = seed_construct(k, g)

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """
        Non-backtracking walk girth check from every vertex.
        Returns True if no cycle shorter than g exists.
        Uses state encoding for speed.
        """
        for s in range(n):
            # State: (current_vertex, previous_vertex) -> encoded as int
            # Root state s is special (no previous)
            visited = set()
            visited.add(0)  # root state
            q = deque([(s, -1, 0)])
            while q:
                u, prev, d = q.popleft()
                if d >= g - 1:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s:
                        return False  # Found a cycle of length d+1 < g
                    state = v * (n + 1) + (u + 1)
                    if state not in visited:
                        visited.add(state)
                        q.append((v, u, d + 1))
        return True

    def heuristic_girth_check(n, adj, g):
        """
        Fast rejection check using BFS. If it returns False, girth < g for sure.
        If it returns True, we still need exact check.
        """
        max_depth = (g - 1) // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Potential cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def generate_simple_regular(n, k, rng):
        """
        Generate a simple k-regular graph using configuration model with repair.
        Returns (edges, adj) or (None, None) if failed.
        """
        max_attempts = 200
        for attempt in range(max_attempts):
            # Create stubs
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            
            rng.shuffle(stubs)
            
            edges_set = set()
            adj = [[] for _ in range(n)]
            valid = True
            
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges_set:
                    valid = False
                    break
                edges_set.add(key)
                adj[u].append(v)
                adj[v].append(u)
            
            if valid:
                return list(edges_set), adj
        
        return None, None

    def try_n(n, k, g, rng, time_budget):
        """Try to find a valid k-regular graph with girth >= g on n vertices."""
        start = time.time()
        max_attempts = 500

        for attempt in range(max_attempts):
            if time.time() - start > time_budget:
                return None
            if time.time() > deadline:
                return None

            edges, adj = generate_simple_regular(n, k, rng)
            if edges is None:
                continue

            # Fast heuristic check first
            if not heuristic_girth_check(n, adj, g):
                continue

            # Exact check
            if exact_girth_check(n, adj, g):
                return edges

        return None

    # Strategy: Try random search for n from moore_bound upward
    # Use the seed construction as a guaranteed fallback
    best_n, best_edges = seed_n, seed_edges

    rng = random.Random(42)

    # Try several n values with limited time each
    # Start near the Moore bound and go up
    # For k=5, g=9: moore_bound = 426
    # We'll try a range of n values
    
    # Candidate n values to try
    # Start with values close to the Moore bound, then expand
    candidate_ns = []
    for offset in [0, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]:
        candidate_ns.append(moore_bound + offset)
    
    # Remove duplicates and sort
    candidate_ns = sorted(set(candidate_ns))
    
    # Time budget per n value - start with more time for smaller n
    time_per_n = 5.0

    for n in candidate_ns:
        if time.time() > deadline - 3.0:
            break
        if n >= best_n:
            break  # No point trying n >= best_n
        
        edges = try_n(n, k, g, rng, time_per_n)
        if edges is not None:
            best_n = n
            best_edges = edges
            # Found a smaller valid n, continue trying to find even smaller
            # But we're going in increasing order, so we can't find smaller from here
            # Actually we should try to find the smallest, so break if we found one
            # But wait, we're trying in increasing order, so the first valid one is the smallest
            break

    # If we still have time, try a few more random n values below the current best
    if best_n > moore_bound and time.time() < deadline - 5.0:
        # Try some random n values between moore_bound and best_n
        remaining_ns = list(range(moore_bound, best_n))
        rng.shuffle(remaining_ns)
        for n in remaining_ns[:50]:  # Try at most 50 more
            if time.time() > deadline - 2.0:
                break
            edges = try_n(n, k, g, rng, 1.0)
            if edges is not None:
                best_n = n
                best_edges = edges

    return best_n, best_edges
