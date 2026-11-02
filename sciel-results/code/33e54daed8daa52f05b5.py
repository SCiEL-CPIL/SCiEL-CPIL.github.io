import random
import time
from collections import deque

def construct(k, g):
    t_start = time.time()
    DEADLINE = t_start + 135  # Safety margin for 150s limit

    def algebraic_construct(q, d, use_xor=False):
        """
        Lazebnik-Ustimenko-Woldar type construction.
        q: field size (or 4 for GF(4) with XOR)
        d: parameter controlling girth (girth >= d + 5 for odd d)
        Returns (n, edges) or None if invalid.
        """
        if use_xor:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # Build the rule table
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def build_adjacency(n, edges):
        """Build adjacency list from edge list."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def verify_girth_nonbacktracking(adj, n, required_girth):
        """
        Verify girth >= required_girth using non-backtracking walks from every vertex.
        A cycle of length L exists iff there's a non-backtracking walk of length L
        from some vertex back to itself (with the walk not immediately reversing).
        For girth >= g, we need no closed non-backtracking walks of length < g.
        
        More precisely: do BFS in the directed line graph (non-backtracking) from
        each directed edge, tracking visited states. If we return to the starting
        directed edge in fewer than g steps, we have a short cycle.
        
        Simpler approach: BFS from each vertex, tracking (node, prev_node) to avoid
        backtracking. If we reach the start vertex via a non-backtracking path of
        length < g, fail.
        """
        # For each starting vertex, do a non-backtracking BFS
        # State: (current_vertex, previous_vertex)
        # We want to detect if we can return to the start in < g steps
        # without immediately backtracking.
        
        for start in range(n):
            # dist[(u, v)] = shortest non-backtracking walk length from start to u
            # where the walk came from v (i.e., last step was v->u)
            # Actually, let's track: for each directed edge (u,v), the shortest
            # non-backtracking walk from (start, start) [virtual] to (u,v).
            # Simpler: BFS where state is (node, parent_node)
            # Start: for each neighbor v of start, state (v, start) at depth 1
            # Transition: from (u, p), go to (w, u) for w in adj[u], w != p
            
            # We detect a cycle if we reach (start, u) where u is a neighbor of start
            # and the path length + 1 < required_girth
            # Actually: reaching (start, u) means a non-backtracking walk from u to start
            # of length dist[(start,u)]. The cycle is u -> ... -> start -> u, length dist+1.
            # But we started from start, went to u (step 1), then non-backtracking to start.
            # So cycle length = 1 + dist[(start, u)] where dist is from (u, start).
            
            # Let me redo: 
            # BFS from start. dist[u] = shortest path from start to u (standard).
            # But we need non-backtracking to correctly count cycles.
            # 
            # Standard approach for girth: BFS from each vertex, when we find an edge
            # (u,v) where both are visited and v != parent[u], cycle length = dist[u]+dist[v]+1.
            # This is correct for simple graphs.
            
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= required_girth - 1:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue  # Don't backtrack
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle: u-v edge creates cycle of length dist[u]+dist[v]+1
                        # But only if this isn't the tree edge
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        return True

    def verify_girth_simple(adj, n, required_girth):
        """
        Simple girth verification: for each vertex, BFS and check for short cycles.
        A cycle of length L < g exists if during BFS from vertex s, we find an edge (u,v)
        where dist[u] + dist[v] + 1 < g and u,v are both in the BFS tree (or one is).
        """
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] + 1 >= required_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != s:
                        # Potential cycle: length = dist[u] + dist[v] + 1
                        # But we need to ensure this is a valid cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        return True

    # Try different algebraic constructions
    best_n = float('inf')
    best_edges = None
    
    # The standard construction: q=4, use XOR, d depends on g
    # For girth g (even), we need d such that girth >= d+5 >= g, so d >= g-5
    # For g=10: d >= 5, so d=5 or d=7 (odd)
    # Let's try various d values and see what n we get
    
    candidates = []
    
    # Try q=4 with XOR (GF(4))
    for d in [5, 7, 9]:  # d must be odd for the construction to work well
        if time.time() > DEADLINE:
            break
        try:
            n, edges = algebraic_construct(4, d, use_xor=True)
            if n < best_n:
                # Verify girth
                adj = build_adjacency(n, edges)
                # Check regularity
                regular = all(len(adj[i]) == k for i in range(n))
                if regular:
                    girth_ok = verify_girth_simple(adj, n, g)
                    if girth_ok:
                        best_n = n
                        best_edges = edges
                        candidates.append((n, d, 'xor4'))
        except Exception:
            pass
    
    # Try q=5 (prime)
    for d in [5, 7, 9]:
        if time.time() > DEADLINE:
            break
        try:
            n, edges = algebraic_construct(5, d, use_xor=False)
            if n < best_n:
                adj = build_adjacency(n, edges)
                regular = all(len(adj[i]) == k for i in range(n) if len(adj[i]) > 0)
                # For q=5, the graph might not be exactly k-regular, check
                # Actually the construction gives q-regular bipartite, so degree = q
                if all(len(adj[i]) == k for i in range(n)):
                    girth_ok = verify_girth_simple(adj, n, g)
                    if girth_ok:
                        best_n = n
                        best_edges = edges
                        candidates.append((n, d, 'q5'))
        except Exception:
            pass
    
    # Try q=7 (prime)
    for d in [5, 7]:
        if time.time() > DEADLINE:
            break
        try:
            n, edges = algebraic_construct(7, d, use_xor=False)
            if n < best_n:
                adj = build_adjacency(n, edges)
                if all(len(adj[i]) == k for i in range(n)):
                    girth_ok = verify_girth_simple(adj, n, g)
                    if girth_ok:
                        best_n = n
                        best_edges = edges
                        candidates.append((n, d, 'q7'))
        except Exception:
            pass

    # If no valid construction found, fall back to the original seed
    if best_edges is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if k == 4:
            n, edges = algebraic_construct(4, d, use_xor=True)
        else:
            n, edges = algebraic_construct(k, d, use_xor=False)
        best_n = n
        best_edges = edges

    # Verify the final result
    adj = build_adjacency(best_n, best_edges)
    assert all(len(adj[i]) == k for i in range(best_n)), "Not k-regular"
    assert verify_girth_simple(adj, best_n, g), "Girth check failed"
    
    # Ensure edges are simple (no duplicates, no loops)
    edge_set = set()
    clean_edges = []
    for u, v in best_edges:
        assert u != v, "Loop found"
        key = (min(u, v), max(u, v))
        if key not in edge_set:
            edge_set.add(key)
            clean_edges.append((u, v))
    best_edges = clean_edges

    return best_n, best_edges
