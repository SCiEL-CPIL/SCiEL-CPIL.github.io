"""Random voltage lift of K_{3,3} over Z_m to find small 3-regular graphs with large girth."""
import time
import random
from collections import deque

def _girth_via_nonbacktracking(n, edges, max_girth):
    """Return the girth (shortest cycle length) using non-backtracking walks from every vertex.
       Returns max_girth if no cycle shorter than max_girth is found."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each directed edge (u,v), do a BFS in the non-backtracking state space
    # State: (current_vertex, previous_vertex)
    # We want to find the shortest walk from (u,v) back to (u,v) without backtracking,
    # but we need to be careful: the non-backtracking walk of length L from (u,v) that
    # returns to (u,v) corresponds to a cycle of length L.
    
    # Actually, let's think more carefully. A non-backtracking walk is a sequence
    # v0, v1, v2, ..., vL where vi != vi-2 for all i >= 2.
    # A cycle of length L is a non-backtracking walk of length L that starts and ends at the same vertex.
    # 
    # The standard approach: for each directed edge (u,v), compute the shortest non-backtracking
    # walk from (u,v) to (u,v) that has length >= 3 (to avoid trivial 2-cycles which don't exist
    # in simple graphs). But actually, since we're in a simple graph, the shortest cycle through
    # directed edge (u,v) is the shortest non-backtracking walk from (u,v) back to (u,v) with
    # length >= 3.
    
    # Better approach: BFS in state space (current, prev). Start from (v, u) for each edge (u,v).
    # Wait, let me reconsider.
    
    # Standard method: For each vertex u, do BFS where state is (current, prev).
    # Initialize: for each neighbor v of u, state (v, u) at distance 1.
    # If we reach state (u, w) for some neighbor w, we found a cycle of length dist.
    # But we need to make sure we don't count the trivial path.
    
    # Let me use a cleaner formulation:
    # For each directed edge (u, v), we want the shortest non-backtracking walk
    # v -> w1 -> w2 -> ... -> w_{L-1} -> u where wi != w_{i-2}, and L >= 3.
    # This walk plus the edge (u,v) forms a cycle of length L+1... no wait.
    
    # Let me just do: for each vertex s, BFS in non-backtracking states.
    # State: (v, p) meaning we're at v, came from p.
    # Start: for each neighbor p of s, state (p, s) at depth 1.
    # Transition: from (v, p), go to (w, v) for each neighbor w of v, w != p.
    # If we ever reach state (s, q) for some q, the cycle length is depth + 1.
    # We want the minimum over all s and all such returns.
    
    # To be efficient, we can stop early if we find a cycle of length < max_girth.
    
    best = max_girth
    for s in range(n):
        # BFS from s
        # dist[(v, p)] = length of non-backtracking walk from s to v via p
        dist = {}
        queue = deque()
        for p in adj[s]:
            dist[(p, s)] = 1
            queue.append((p, s))
        
        while queue:
            v, p = queue.popleft()
            d = dist[(v, p)]
            if d + 1 >= best:
                continue
            for w in adj[v]:
                if w == p:
                    continue
                nd = d + 1
                if w == s:
                    # Found a cycle of length nd + 1
                    cycle_len = nd + 1
                    if cycle_len < best:
                        best = cycle_len
                    # Don't enqueue since we're returning to s
                    continue
                state = (w, v)
                if state not in dist or dist[state] > nd:
                    dist[state] = nd
                    queue.append(state)
        if best == 3:
            break
    
    return best

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # We need k-regular, girth >= g, minimize n.
    # Use voltage lift of K_{k,k} over Z_m.
    # K_{k,k} has vertices U = {0..k-1} and V = {0..k-1}, edges (u_i, v_j) for all i,j.
    # Voltage lift over Z_m: vertices (u_i, a) and (v_j, b) for i,j in 0..k-1, a,b in Z_m.
    # Total vertices: 2*k*m.
    # Edge from (u_i, a) to (v_j, b) exists iff b - a ≡ voltage(u_i, v_j) mod m.
    # Each vertex has degree k (since in K_{k,k}, each vertex has degree k).
    
    # For k=3, g=16: n = 6m. We want to find small m such that girth >= 16.
    
    # The girth of the lift depends on the voltage assignment.
    # Random voltages: assign random elements of Z_m to each edge of K_{k,k}.
    
    # Strategy: try increasing m, for each m try random voltage assignments,
    # check girth, return the first valid one with smallest n.
    
    # Moore bound for k=3, g=16: n >= 2 * (3*(3^7 - 1)/(3-1)) ... let me compute.
    # For bipartite k-regular graph with girth g (even), Moore bound:
    # n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i = 2 * (1 + 2 + 4 + ... + 2^{g/2-1})
    # For g=16: n >= 2 * (2^8 - 1) = 2 * 255 = 510.
    # So m >= 510/6 = 85.
    
    # Let's start from a reasonable lower bound for m.
    
    # First, let's figure out the minimum m we should try.
    # Moore bound gives n >= 510, so m >= 85.
    
    # But random voltages might need larger m to achieve girth 16.
    # Let's start from m = 85 and go up.
    
    # Actually, let's be smarter. Try a range of m values with random restarts.
    
    best_n = None
    best_edges = None
    
    rng = random.Random(42)
    
    # Try m values starting from the Moore bound minimum
    m_min = 85  # from Moore bound for k=3, g=16
    
    # We'll try m from m_min upward, with multiple random attempts per m
    max_m = 200  # upper limit for m to keep n reasonable
    
    for m in range(m_min, max_m + 1):
        if time.time() > deadline:
            break
        
        n = 2 * k * m
        
        # Try several random voltage assignments
        max_attempts = 50
        if n > 600:
            max_attempts = 20
        if n > 800:
            max_attempts = 10
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Assign random voltages to edges of K_{k,k}
            # voltage[i][j] = voltage on edge from u_i to v_j
            voltage = [[rng.randint(0, m - 1) for j in range(k)] for i in range(k)]
            
            # Build the lifted graph
            edges = []
            for i in range(k):
                for a in range(m):
                    u_vertex = i * m + a
                    for j in range(k):
                        b = (a + voltage[i][j]) % m
                        v_vertex = k * m + j * m + b
                        edges.append((u_vertex, v_vertex))
            
            # Check girth
            girth = _girth_via_nonbacktracking(n, edges, g)
            if girth >= g:
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                    return n, edges  # Return immediately since we're trying m in increasing order
        
        # If we didn't find a valid graph for this m, try next m
    
    # If we didn't find anything with random voltages, try a more structured approach
    # or fall back to the LUW construction
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction
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
