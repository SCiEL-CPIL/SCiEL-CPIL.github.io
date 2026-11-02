"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def verify_graph(n, edges, k, g):
        """Verify: simple, k-regular, girth >= g using non-backtracking walks. Returns (n, edges) if valid, else None."""
        if time.time() > deadline:
            return None
        # Check simplicity and degree
        edge_set = set()
        degree = [0] * n
        for u, v in edges:
            if u == v:
                return None
            if u < 0 or u >= n or v < 0 or v >= n:
                return None
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None
            edge_set.add(key)
            degree[u] += 1
            degree[v] += 1
        for i in range(n):
            if degree[i] != k:
                return None
        
        # Check girth >= g using non-backtracking walks from every vertex
        # A cycle of length L exists iff there is a non-backtracking walk of length L that returns to start.
        # We need to ensure no non-backtracking walk of length L < g returns to any vertex.
        # Equivalently, for each start vertex s, and each neighbor v of s,
        # we walk non-backtracking from v for L-1 steps and check if we reach s.
        # For efficiency, we do BFS in the non-backtracking state space (current, prev).
        # State: (node, prev_node). Start states: (v, s) for each edge (s,v).
        # We want to see if we can reach (s, u) for some neighbor u of s, with path length L-1 >= 1.
        # Actually, simpler: 
        # Girth >= g means no cycle of length < g.
        # A cycle of length L corresponds to a closed non-backtracking walk of length L.
        # We can check for each start s, do a BFS in the directed line graph (states are directed edges).
        # State (u, v) means "we are at v, came from u".
        # Start: for each neighbor w of s, state is (s, w).
        # Transition: from (u, v), go to (v, x) for each neighbor x of v, x != u.
        # We want to see if we can reach a state (v, s) where v is a neighbor of s.
        # The length of the walk from (s, w) to (v, s) is L-1, so the cycle length is L = (steps) + 1.
        # We need to ensure no such path of length L-1 < g-1 exists.
        # So we BFS up to depth g-2.
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # For each start vertex s
        for s in range(n):
            # visited[state] = distance
            # state is (prev, curr)
            visited = {}
            # Queue of (prev, curr, dist)
            # dist is the number of steps taken in the non-backtracking walk.
            # Initial states: (s, w) for w in adj[s], dist=0
            queue = []
            for w in adj[s]:
                state = (s, w)
                visited[state] = 0
                queue.append(state)
            
            qi = 0
            while qi < len(queue):
                prev, curr = queue[qi]
                qi += 1
                d = visited[(prev, curr)]
                # If d+1 >= g-1, we don't need to expand further for this s?
                # We are looking for a return to s.
                # A return to s happens if curr == s. But we start with prev=s, curr=w.
                # If we reach a state (v, s), the cycle length is d+1.
                # We need d+1 < g to fail. So d < g-1.
                if d >= g - 1:
                    continue
                for x in adj[curr]:
                    if x == prev:
                        continue
                    new_state = (curr, x)
                    if new_state in visited:
                        continue
                    visited[new_state] = d + 1
                    # Check if we have returned to s
                    if x == s:
                        cycle_len = d + 1
                        if cycle_len < g:
                            return None
                    queue.append(new_state)
        return (n, edges)

    def build_luw(k, g):
        """LUW construction fallback."""
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

        def build_luw_d(d):
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

        # For LUW, girth is at least d+5 (odd) or d+4 (even)?
        # Standard result: girth of LUW(d, q) is at least d+5 if d is odd, d+4 if d is even?
        # Actually, the girth is 2*ceil(d/2) + 4? No.
        # Lazebnik-Ustimenko-Woldar: girth is at least d+5.
        # We need girth >= g.
        # So d+5 >= g => d >= g-5.
        # Also d must be >= 2.
        # Let's try d from max(2, g-5) upwards.
        
        min_d = max(2, g - 5)
        # Also, for bipartite graphs, girth is even.
        # If g is odd, say 9, girth must be >= 10.
        # LUW graphs are bipartite.
        # So if g=9, we need girth >= 10.
        # d+5 >= 10 => d >= 5.
        # If g=10, we need girth >= 10. d+5 >= 10 => d >= 5.
        
        # Let's just iterate d.
        for d in range(min_d, min_d + 20):
            if time.time() > deadline:
                break
            try:
                n, edges = build_luw_d(d)
                result = verify_graph(n, edges, k, g)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def build_gq24():
        """Construct GQ(2,4) incidence graph. 5-regular, girth 10, 75 vertices?"""
        # GQ(2,4) parameters:
        # s=2, t=4.
        # Points: (s+1)(st+s+t+1) = 3*(8+2+4+1) = 3*15 = 45.
        # Lines: (t+1)(s(t+1)+1) = 5*(2*5+1) = 5*11 = 55? No.
        # Number of lines = (s+1)(st+s+t+1) / (s+1) * (t+1)?
        # Number of points = (s+1)(st+s+t+1).
        # Number of lines = (t+1)(st+s+t+1).
        # For s=2, t=4:
        # Points = 3 * 15 = 45.
        # Lines = 5 * 15 = 75.
        # Wait, incidence graph vertices = Points + Lines = 45 + 75 = 120.
        # Each point is on t+1=5 lines. So degree of point-vertex is 5.
        # Each line has s+1=3 points. So degree of line-vertex is 3.
        # This is NOT 5-regular. It is (5,3)-biregular.
        # So GQ(2,4) incidence graph is not 5-regular.
        # The task requires k-regular.
        # So GQ(2,4) is not a direct solution for k=5.
        # 
        # What about the point graph of GQ(2,4)?
        # Two points are adjacent if they are collinear.
        # Each point is collinear with s*t = 2*4 = 8 other points?
        # Number of points on a line = s+1=3.
        # Number of lines through a point = t+1=5.
        # Each line has 2 other points.
        # So a point is adjacent to 5*2 = 10 other points.
        # So the point graph is 10-regular.
        # 
        # So GQ(2,4) doesn't give a 5-regular graph directly.
        # 
        # Let's stick to LUW.
        pass

    # Special case: k=5, g=9
    # LUW with d=5 should give girth >= 10.
    # Let's see what n is.
    # LUW(5, 5) -> q=5, d=5.
    # Number of vertices is roughly q^d * 2?
    # For q=5, d=5, n is around 2 * 5^5 = 6250? Too big.
    # We need smaller n.
    
    # Is there a smaller 5-regular graph with girth 9?
    # The (5,9)-cage is known to have 1116 vertices (recent result).
    # LUW(5,5) might be larger.
    # Let's check LUW(5, 4)? Girth >= 9?
    # If d=4, girth >= 4+5=9.
    # So LUW(4, 5) might have girth 9.
    # Let's try d=4 first.
    
    result = build_luw(k, g)
    if result is not None:
        return result

    # Absolute fallback: should not reach here
    raise RuntimeError("No valid graph found")
# EVOLVE-BLOCK-END
