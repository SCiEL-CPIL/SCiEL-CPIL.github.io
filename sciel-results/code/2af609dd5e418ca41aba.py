"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _girth_check(adj, g):
    """Check if graph with adjacency list 'adj' has girth >= g.
    Uses non-backtracking walks from every vertex. Returns True if girth >= g."""
    n = len(adj)
    # For each vertex, do BFS with non-backtracking constraint
    # State: (vertex, prev_vertex)
    # We want to find the shortest cycle. A non-backtracking walk of length L
    # that returns to the start vertex (with the constraint that we don't immediately
    # go back) gives a cycle of length L.
    
    # More efficient: for each edge (u,v), check if there's a non-backtracking path
    # from v back to u of length < g-1. If not for all edges, girth >= g.
    
    # Alternative: BFS from each vertex with state (node, prev) tracking shortest
    # non-backtracking path back to start.
    
    # For efficiency, we check: for each directed edge (u->v), is there a non-backtracking
    # path from v to u of length at most g-2? If yes, we have a cycle of length <= g-1.
    
    # Let's do it differently: for each vertex s, run a modified BFS where state is
    # (current, previous) and we track the shortest distance from s. If we reach (s, prev)
    # with distance < g, we found a cycle of length < g.
    
    # Actually, the standard approach: a graph has girth >= g iff for every vertex v,
    # the number of non-backtracking walks of length < g/2 from v is exactly the Moore bound
    # count (i.e., no cycles in the ball of radius floor((g-1)/2)).
    
    # Simpler: check that the ball of radius r = (g-1)//2 around every vertex is a tree
    # (no repeated vertices in BFS tree). For girth g, the ball of radius (g-1)//2 
    # around any vertex must be a tree (no cycles).
    
    r = (g - 1) // 2
    for start in range(n):
        # BFS from start, tracking parent to detect cycles
        visited = [-1] * n  # -1 means unvisited, else parent
        visited[start] = -2  # mark start as visited with special parent
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist == r:
                continue
            for nb in adj[node]:
                if visited[nb] == -1:
                    visited[nb] = node
                    queue.append((nb, dist + 1))
                elif visited[nb] != node and node != visited[nb]:
                    # Found a back edge - but need to check if it forms a short cycle
                    # Actually in BFS, if we find an edge to an already-visited node
                    # that isn't our parent, it forms a cycle. The cycle length is
                    # dist + 1 + (distance from start to nb via BFS tree)
                    # But this is complex. Let's use a cleaner approach.
                    pass
        # If we found a cycle of length < g, return False
        # The above BFS doesn't cleanly detect cycle lengths. Let me redo.
    
    # Better approach: for each vertex, do BFS and if we encounter a node already
    # visited via a different path at the same or lower level, we have a cycle.
    # The cycle length = dist[node] + 1 + dist[nb] where both are at the same level
    # or we can detect it when exploring edges.
    
    # Cleanest: non-backtracking BFS from each vertex
    for start in range(n):
        # dist[node][prev] = shortest non-backtracking walk length from start to node
        # where prev is the node we came from
        # Initialize: dist[start][None] = 0
        # We use a dict or 2D array. Since n can be up to ~600, n*k states is fine.
        dist = [[-1] * n for _ in range(n)]
        dist[start][start] = 0  # sentinel: at start, came from "start" (self)
        queue = deque([(start, start)])  # (current, prev)
        while queue:
            node, prev = queue.popleft()
            d = dist[node][prev]
            if d >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue  # non-backtracking
                if dist[nb][node] == -1:
                    dist[nb][node] = d + 1
                    if nb == start:
                        # Found a cycle of length d+1
                        if d + 1 < g:
                            return False
                    queue.append((nb, node))
                # If dist[nb][node] != -1, we already have a shorter or equal path, skip
    return True


def _random_3reg(n):
    """Generate a random 3-regular simple graph on n vertices. Returns adjacency list or None."""
    if n < 4 or n % 2 != 0:
        return None
    # Configuration model: create 3 stubs per vertex, random matching
    random.seed()  # Use default random
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    random.shuffle(stubs)
    adj = [set() for _ in range(n)]
    for i in range(0, n * 3, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # loop, retry
        if v in adj[u]:
            return None  # multi-edge, retry
        adj[u].add(v)
        adj[v].add(u)
    # Convert to list
    adj_list = [list(s) for s in adj]
    # Verify degree
    for v in range(n):
        if len(adj_list[v]) != 3:
            return None
    return adj_list


def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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
    rule_list = [(rule[c][0], rule[c][1]) for c in range(2, d + 1)]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for idx, c in enumerate(range(2, d + 1)):
                a, b = rule_list[idx]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    i = 0
    while i < len(order):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    return len(order), edges


def construct(k, g):
    t0 = time.time()
    deadline = t0 + 125.0  # safety margin before 150s kill
    
    # Moore bound for (k, g): for even g, n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # For odd g, n >= k * sum_{i=0}^{(g-3)/2} (k-1)^i
    if g % 2 == 0:
        moore = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        moore = k * sum((k - 1) ** i for i in range((g - 1) // 2))
    
    # Start searching from near the Moore bound
    # For k=3, g=16: moore = 2*(1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 2*255 = 510
    # So start around 512
    
    best_n = None
    best_edges = None
    
    # Try random 3-regular graphs starting from Moore bound upward
    n_start = moore
    if n_start % 2 != 0:
        n_start += 1
    
    # Try a range of n values
    for n_try in range(n_start, n_start + 50, 2):  # even numbers only for 3-regular
        if time.time() > deadline:
            break
        # Try multiple random graphs at this size
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            adj = _random_3reg(n_try)
            if adj is None:
                continue
            if _girth_check(adj, g):
                # Found a valid graph!
                edges = []
                for u in range(n_try):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                if best_n is None or n_try < best_n:
                    best_n = n_try
                    best_edges = edges
                    # Try to find even smaller? We're going up, so this is our best at this n
                    # But maybe a smaller n works? We started from moore, so this is minimal
                    return n_try, edges
        # If we didn't find one at this n, try larger
    
    # Fallback to LUW construction
    if time.time() < deadline:
        n_luw, edges_luw = _luw_construct(k, g)
        if best_n is None or n_luw < best_n:
            return n_luw, edges_luw
        else:
            return best_n, best_edges
    
    if best_n is not None:
        return best_n, best_edges
    
    # Last resort: LUW
    return _luw_construct(k, g)
# EVOLVE-BLOCK-END
