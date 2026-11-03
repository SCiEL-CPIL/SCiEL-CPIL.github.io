import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    def check_girth_nb(n, edges, min_girth):
        """Non-backtracking walk girth check. Returns True if girth >= min_girth."""
        if min_girth <= 2:
            return True
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each directed edge (u, v), we track the shortest non-backtracking walk
        # starting with (u, v). If we find a cycle of length < min_girth, return False.
        # 
        # State: (current_vertex, previous_vertex)
        # We do BFS from each directed edge, but we can be smarter:
        # For each vertex, for each neighbor, do a non-backtracking BFS.
        # 
        # More efficient: For each starting directed edge (s, t), do BFS where
        # state is (v, u) meaning we're at v having come from u.
        # We want to find if there's a non-backtracking walk of length < min_girth
        # that returns to s.
        
        # Actually, the standard approach: for each vertex s, do a BFS where the state
        # is (v, u) = at v, came from u. Initialize with all (s, u) for u in adj[s].
        # Then expand: from (v, u), go to (w, v) for w in adj[v], w != u.
        # If we reach (s, v) at distance d, then we have a cycle of length d+1.
        # We need d+1 >= min_girth, i.e., d >= min_girth - 1.
        # So we only need to check up to distance min_girth - 2.
        
        max_dist = min_girth - 2  # we need walks of length < min_girth, so up to min_girth-1
        # Actually, a non-backtracking walk of length L from s back to s gives a cycle of length L.
        # We want no cycle of length < min_girth, so no non-backtracking walk of length L < min_girth
        # that starts and ends at the same vertex.
        
        # For each vertex s:
        for s in range(n):
            # dist[(v, u)] = shortest non-backtracking walk from s to v, having come from u
            # Initialize: for each neighbor t of s, dist[(t, s)] = 1
            dist = {}
            queue = deque()
            for t in adj[s]:
                dist[(t, s)] = 1
                queue.append((t, s))
            
            while queue:
                v, u = queue.popleft()
                d = dist[(v, u)]
                if d >= max_dist:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue  # non-backtracking
                    if w == s:
                        # Found a cycle of length d + 1
                        if d + 1 < min_girth:
                            return False
                    else:
                        key = (w, v)
                        if key not in dist:
                            dist[key] = d + 1
                            queue.append(key)
        
        return True

    def build_luw(d, q):
        """Build LUW graph with parameter d over field of size q."""
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Rule for LUW construction
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

    best_n = None
    best_edges = None
    
    # For k=3, g=16: The LUW construction with q=3 (prime) and various d values
    # The Moore bound for n(3,16) is 2 + 3 * (2^(8-1) - 1) = 2 + 3*127 = 383
    # But the actual smallest known is 936.
    # 
    # LUW graphs: n = 2 * q^d
    # For q=3: d=6 gives n=2*729=1458, d=5 gives n=2*243=486
    # For q=4: d=5 gives n=2*1024=2048, d=4 gives n=2*256=512
    # 
    # We need to find the smallest d such that the LUW graph has girth >= g.
    # The girth of LUW graph with parameter d is at least 2*d + 2 for some constructions.
    # For g=16, we need d >= 7 for girth >= 16? No, that's not right.
    # 
    # Let's just try various d values and check.
    
    # Try different (d, q) combinations
    candidates = []
    for q in [3, 4, 5, 7]:
        for d in range(3, 12):
            n = 2 * (q ** d)
            candidates.append((n, d, q))
    
    # Sort by n (smallest first)
    candidates.sort()
    
    for n_est, d, q in candidates:
        if time.time() > deadline - 10:  # Leave 10s for girth check
            break
        
        if n_est > 2000:  # Skip very large graphs
            continue
            
        n, edges = build_luw(d, q)
        
        # Verify it's k-regular
        if len(edges) != n * k // 2:
            continue
        
        # Check girth
        if check_girth_nb(n, edges, g):
            best_n = n
            best_edges = edges
            break
    
    # Fallback if no valid graph found
    if best_n is None:
        # Use the original seed's approach
        d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = build_luw(d_seed, k)
    
    return best_n, best_edges
