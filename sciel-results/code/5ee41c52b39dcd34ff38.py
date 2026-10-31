import random
import time
import sys
from collections import deque

# ---- global deadline (150s wall-clock kill; keep margin) ----
_T0 = time.time()
_DEADLINE = _T0 + 142.0


def construct(k, g):
    if time.time() > _DEADLINE:
        return _luw_fallback(k, g)
    
    # Try voltage graph construction over dodecahedral base
    result = _voltage_graph_dodeca(k, g)
    if result is not None:
        return result
    
    # Fallback to LUW
    return _luw_fallback(k, g)


def _luw_fallback(k, g):
    """Original LUW construction as guaranteed fallback."""
    def luw_construct(k, g):
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
    return luw_construct(k, g)


def _get_dodecahedron():
    """Return adjacency list of the dodecahedral graph (20 vertices, 3-regular, girth 5)."""
    # Dodecahedral graph: 20 vertices, 30 edges, girth 5
    # Using standard coordinates/adjacency
    adj = [[] for _ in range(20)]
    # The dodecahedron can be constructed from two pentagonal rings and connections
    # Standard labeling: vertices 0-4 (top pentagon), 5-9 (upper ring), 10-14 (lower ring), 15-19 (bottom pentagon)
    # Top pentagon: 0-1-2-3-4-0
    for i in range(5):
        adj[i].append((i + 1) % 5)
        adj[(i + 1) % 5].append(i)
    # Bottom pentagon: 15-16-17-18-19-15
    for i in range(5):
        v = 15 + i
        adj[v].append((15 + (i + 1)) % 5 + 15 if False else 15 + (i + 1) % 5)
        adj[15 + (i + 1) % 5].append(v)
    # Wait, let me use a cleaner construction
    adj = [[] for _ in range(20)]
    # Use the standard dodecahedron adjacency
    # Vertices 0-4: outer pentagon
    # Vertices 5-9: inner pentagon (rotated)
    # Vertices 10-14: upper connecting ring
    # Vertices 15-19: lower connecting ring
    
    # Actually, let me just hardcode the dodecahedral graph edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # top pentagon
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # bottom pentagon
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 10),  # middle ring 1
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 15),  # middle ring 2
        (0, 10), (1, 11), (2, 12), (3, 13), (4, 14),  # top to ring1
        (5, 15), (6, 16), (7, 17), (8, 18), (9, 19),  # bottom to ring2
        (10, 15), (11, 16), (12, 17), (13, 18), (14, 19)  # ring1 to ring2
    ]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Verify 3-regular
    for i in range(20):
        if len(adj[i]) != 3:
            raise ValueError(f"Vertex {i} has degree {len(adj[i])}")
    return adj


def _verify_girth_nonbacktracking(adj, g, max_check=15):
    """
    Verify no non-backtracking closed walk of length < g exists.
    Uses BFS on the state space (vertex, prev_vertex) tracking shortest
    non-backtracking paths. Returns True if girth >= g.
    
    For each starting vertex, we do BFS on states (current, prev) where
    prev is the vertex we came from (or -1 for start). We track the
    minimum distance to reach each state. A non-backtracking closed walk
    of length L from vertex v exists if we can return to v via a path
    of length L where the last step doesn't backtrack.
    
    More efficiently: we check if there's any cycle of length < g by
    doing BFS from each vertex and checking if we can return to the start
    with a non-backtracking walk.
    """
    n = len(adj)
    # For each vertex, do BFS tracking (node, parent) states
    # State: (current_vertex, previous_vertex) where previous_vertex is what we came from
    # We want to find if there's a non-backtracking walk from v back to v of length < g
    
    # More efficient: for each edge (u,v), check if there's a non-backtracking
    # walk from v to u of length < g-1 (so total cycle length < g)
    
    # Actually, simplest correct approach: BFS on directed edges (non-backtracking walks)
    # State: (current, prev) meaning we're at current, came from prev
    # Start: for each vertex s, for each neighbor t of s, state (t, s) at distance 1
    # We want to reach state (s, t) for some neighbor t, which means we've completed
    # a non-backtracking walk from s back to s.
    
    # But this is O(n * degree * g) which for n=936, g=16 is manageable.
    
    # Alternative: just check all cycles up to length g-1 via BFS from each vertex
    # tracking the parent to avoid backtracking.
    
    for start in range(n):
        # BFS: state is (vertex, prev_vertex), distance
        # prev_vertex = -1 means we're at start with no constraint
        # We initialize: for each neighbor nb of start, state (nb, start) at dist 1
        # Goal: reach (start, nb) for any neighbor nb of start
        
        visited = set()  # (vertex, prev)
        queue = deque()
        
        for nb in adj[start]:
            state = (nb, start)
            visited.add(state)
            queue.append((state, 1))
        
        while queue:
            (cur, prev), dist = queue.popleft()
            
            if dist >= g - 1:
                continue
            
            # Check if we've completed a cycle: cur == start and prev != start
            # Actually, if cur == start, we've returned to start. The walk is
            # start -> ... -> prev -> start. This is non-backtracking if prev != start
            # (which it isn't since prev is the vertex before start in the walk,
            #  and we only go to neighbors).
            # Wait: state (cur, prev) means we're at cur, came from prev.
            # If cur == start, we've reached start. The cycle is closed.
            # The cycle length is dist. We need dist < g for a violation.
            if cur == start:
                # We found a non-backtracking closed walk of length dist from start
                # But wait - we need to make sure this is actually a cycle, not just
                # going start->nb->start which would be length 2 (not a cycle in simple graph)
                # In a simple graph, start->nb->start is not a valid cycle (length 2).
                # Our BFS avoids backtracking, so start->nb->start is excluded because
                # from nb we can't go back to start (that's backtracking).
                # So if we reach (start, prev), the walk is start -> ... -> prev -> start
                # with no immediate backtracking, and length >= 3.
                if dist < g:
                    return False
            
            for nb in adj[cur]:
                if nb == prev:
                    continue  # non-backtracking constraint
                new_state = (nb, cur)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, dist + 1))
    
    return True


def _voltage_graph_dodeca(k, g):
    """
    Try to construct a k-regular graph with girth >= g using voltage graph
    over the dodecahedral graph.
    
    The dodecahedral graph has 20 vertices, 30 edges, is 3-regular, girth 5.
    We assign voltages from Z_m to each edge. The lift has 20*m vertices.
    Vertex (u, a) is connected to (v, a + w(u,v)) for each edge (u,v) with voltage w(u,v).
    
    For the lift to be k-regular with girth >= g, we need:
    1. The lift is simple (no multi-edges, no loops)
    2. No non-backtracking closed walk in the base has total voltage 0 mod m
       with length < g
    
    We scan m from the Moore bound requirement upward.
    """
    if k != 3:
        return None  # Only handle k=3 for now
    
    adj = _get_dodecahedron()
    n_base = len(adj)  # 20
    
    # Get the list of edges (undirected, each once)
    base_edges = []
    seen = set()
    for u in range(n_base):
        for v in adj[u]:
            if u < v:
                base_edges.append((u, v))
                seen.add((u, v))
    
    # Moore bound for 3-regular, girth g: n >= 2*(3^((g-1)//2) - 1) / (3-1) for even g
    # For g=16: n >= 2*(3^7 - 1)/2 = 3^7 - 1 = 2186? No wait.
    # Moore bound for k-regular, girth g (even): n >= 2*(k^((g/2)-1) + ... ) 
    # Actually for k=3, g=16 (even): n >= 2*(3^7 - 1)/(3-1) = 3^7 - 1 = 2186
    # Hmm, that seems high. Let me recalculate.
    # Moore bound: for k-regular graph of girth g, n >= k*(k-1)^((g-2)/2) + (k-1)^((g-2)/2) ... 
    # Standard: for even girth g = 2r, n >= 2 * sum_{i=0}^{r-1} (k-1)^i = 2*((k-1)^r - 1)/(k-2)
    # For k=3, g=16, r=8: n >= 2*(2^8 - 1)/(1) = 2*255 = 510
    # So n >= 510. With base of 20, m >= ceil(510/20) = 26.
    
    r = g // 2  # = 8 for g=16
    moore_bound = 2 * ((k - 1) ** r - 1) // (k - 2) if k > 2 else g
    m_min = max(1, (moore_bound + n_base - 1) // n_base)
    
    # Time budget for voltage search
    t_start = time.time()
    time_budget = 100.0  # seconds
    
    for m in range(m_min, m_min + 200):
        if time.time() - t_start > time_budget:
            break
        
        # Try multiple random voltage assignments for this m
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > _DEADLINE - 20:
                return None
            
            # Random voltage assignment
            voltages = {}
            for (u, v) in base_edges:
                voltages[(u, v)] = random.randint(0, m - 1)
                voltages[(v, u)] = (-voltages[(u, v)]) % m
            
            # Build the lift
            # Vertex (u, a) where u in [0,20), a in [0,m)
            # Total vertices: 20 * m
            n_lift = n_base * m
            
            # Check simplicity: no loops, no multi-edges
            # A loop occurs if (u, a) connects to itself: v=u and a+w(u,u)=a, but no self-loops in base
            # Multi-edge: (u,a) connects to (v, a+w(u,v)) and also to (v, a+w'(u,v)) for parallel edges
            # Since base is simple, for each pair (u,v) there's exactly one edge, so no multi-edges in lift
            # unless the same (v, b) is reached via different paths, but that's fine - it's just one edge.
            # Actually, the lift of a simple graph with voltage assignment is always simple
            # as long as the base is simple. Each edge (u,v) in base gives edge ((u,a),(v,a+w)) in lift.
            # Two different edges in base give different edges in lift. Same edge with different a gives
            # different edges. So the lift is always simple.
            
            # Build adjacency for the lift
            lift_adj = [[] for _ in range(n_lift)]
            lift_edges = set()
            
            for u in range(n_base):
                for v in adj[u]:
                    w = voltages[(u, v)]
                    for a in range(m):
                        ua = u * m + a
                        va = v * m + (a + w) % m
                        if ua < va:
                            lift_edges.add((ua, va))
                        else:
                            lift_edges.add((va, ua))
            
            # Verify girth using non-backtracking walk check
            # This is the expensive part - O(n_lift * 3 * g) states
            # For n_lift up to ~20*200=4000, g=16, this is ~200k operations per check
            
            # Build adjacency list for lift
            for u, v in lift_edges:
                lift_adj[u].append(v)
                lift_adj[v].append(u)
            
            # Verify girth
            if _verify_girth_nonbacktracking(lift_adj, g):
                edges_list = list(lift_edges)
                return n_lift, edges_list
    
    return None
