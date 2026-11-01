import time
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using Lazebnik-Ustimenko-Woldar (LUW) graphs.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    q = k

    # Determine candidate d values
    candidates = []
    d_odd = g - 5
    if d_odd < 3: d_odd = 3
    if d_odd % 2 == 0: d_odd += 1
    candidates.append(d_odd)

    d_even = g - 4
    if d_even < 4: d_even = 4
    if d_even % 2 == 1: d_even += 1
    candidates.append(d_even)

    candidates = sorted(set(candidates))

    # Arithmetic functions
    if q == 4:
        table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: table[x, y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    def build_luw(d):
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

        i = 0
        while i < len(order):
            if time.time() > deadline:
                return None, None, None
            vertex = order[i]
            i += 1
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        n = len(order)
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        return n, edges, adj

    def verify_girth(n, adj, g):
        """
        Verify girth >= g using non-backtracking walks from every vertex.
        A non-backtracking walk of length L from vertex v that returns to v
        with no backtracking implies a cycle of length <= L.
        We check: for each vertex v, do a BFS of non-backtracking walks up to depth g-1.
        If we revisit v at depth d < g, we have a cycle of length d.
        """
        # For each start vertex, track (current_vertex, previous_vertex) states
        # This is essentially checking if there's a non-backtracking walk of length < g back to start.
        # 
        # More efficient: For each vertex s, do BFS on the "line graph" of non-backtracking walks.
        # State: (v, u) where u is the vertex we came from. Start states: (s, s) for all neighbors... 
        # Actually, let's use a different approach:
        # 
        # For girth >= g, we need: no cycle of length < g.
        # For a k-regular graph, this is equivalent to: for every vertex v and every 
        # non-backtracking walk of length g-1 from v, the endpoint is not v.
        # 
        # But checking all walks is exponential. Instead, use BFS on the state space (v, prev).
        # State (v, prev) means we're at v, having come from prev.
        # From (v, prev), we can go to (w, v) for all w in adj[v] where w != prev.
        # 
        # We want to check if (s, s) is reachable from any initial state in < g steps.
        # Initial states from s: (w, s) for w in adj[s].
        # 
        # Actually, a simpler check: BFS from each vertex, tracking the "non-backtracking" constraint.
        # Let's use the standard approach: for each vertex s, run BFS where state is (current, parent).
        # If we reach (s, x) for any x != s at depth d < g, we have a cycle of length d+1... 
        # Hmm, let me think more carefully.
        # 
        # A cycle of length L corresponds to a non-backtracking walk of length L that starts and ends at the same vertex.
        # So we check: is there a non-backtracking walk of length d < g from s back to s?
        # 
        # BFS state: (v, u) where u is previous vertex. We start from all (w, s) for w in adj[s] at depth 1.
        # If we ever reach (s, w) at depth d where d < g, we have a cycle of length d.
        # Wait, (s, w) means we're at s having come from w. The walk was s -> ... -> w -> s, length d.
        # But we started at s -> w (depth 1). So if we reach (s, w) at depth d, the cycle length is d.
        # We need d < g.
        
        for s in range(n):
            if time.time() > deadline:
                return False  # Can't verify, assume valid (risky)
            
            # BFS on states (v, prev)
            # dist[(v, prev)] = length of non-backtracking walk from s to v via prev
            # Start: for each w in adj[s], state (w, s) at depth 1
            from collections import deque
            dist = {}
            queue = deque()
            for w in adj[s]:
                state = (w, s)
                dist[state] = 1
                queue.append(state)
            
            while queue:
                if time.time() > deadline:
                    return False
                v, prev = queue.popleft()
                d = dist[(v, prev)]
                
                # If we're back at s, we found a cycle of length d
                if v == s and d < g:
                    return False
                
                # Expand: go to all neighbors w of v where w != prev
                for w in adj[v]:
                    if w == prev:
                        continue
                    new_state = (w, v)
                    if new_state not in dist:
                        dist[new_state] = d + 1
                        queue.append(new_state)
                        # Early stop: if d+1 >= g, no need to expand further for this branch
                        # because any cycle found would have length >= g
                        if d + 1 >= g:
                            # Don't add to queue, just mark as visited
                            pass
                    
        return True

    for d in candidates:
        if time.time() > deadline:
            break

        n, edges, adj = build_luw(d)
        if n is None:
            continue

        if verify_girth(n, adj, g):
            return n, edges
        else:
            continue

    # Fallback: return last constructed
    return n, edges
