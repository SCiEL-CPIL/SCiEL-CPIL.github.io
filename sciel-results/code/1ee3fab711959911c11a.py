"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using non-backtracking BFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Build non-backtracking walk representation
    # State: (current_vertex, prev_vertex)
    # BFS from each state
    for start in range(n):
        # dist[v][u] = shortest non-backtracking walk from start to u arriving via v
        # We want: no non-backtracking walk of length < g that returns to start
        # Actually, girth check: no cycle of length < g
        # Non-backtracking BFS from each vertex:
        # dist[u][v] = length of shortest non-backtracking walk from (start, -1) to (u, v)
        # A cycle of length L exists iff there's a non-backtracking walk from (start, -1)
        # that reaches (start, u) for some neighbor u of start, with total length L-1+1 = L
        
        # Simpler: BFS in the state space (current, prev) starting from (start, -1)
        # with distance 0. Then for each neighbor v of start, check if (start, v) is reached
        # at distance L-1, giving cycle of length L. If L < g, girth violated.
        
        # dist[prev][curr] = shortest distance to reach state (curr, prev) 
        # i.e., we are at curr, came from prev
        dist = {}
        # Start: at 'start', came from -1 (no previous)
        # We can go to any neighbor of start
        queue = deque()
        for nb in adj[start]:
            dist[(nb, start)] = 1
            queue.append((nb, start))
        
        while queue:
            curr, prev = queue.popleft()
            d_cur = dist[(curr, prev)]
            # From (curr, prev), we can go to any neighbor of curr except prev
            if d_cur + 1 >= g:
                continue  # No need to explore further
            for nb in adj[curr]:
                if nb == prev:
                    continue
                new_state = (nb, curr)
                if new_state not in dist:
                    dist[new_state] = d_cur + 1
                    # If nb == start, we have a cycle of length d_cur + 1
                    if nb == start:
                        return False
                    queue.append(new_state)
    
    return True


def _try_construct(k, g, deadline):
    """Try to construct a 4-regular graph of girth g with as few vertices as possible."""
    # Moore bound for k=4, g=10 (even):
    # n >= 2 * k * (k-1)^(g/2 - 1) = 2 * 4 * 3^4 = 2 * 4 * 81 = 648? 
    # Wait, for even girth g, Moore bound is 2*k*(k-1)^(g/2 - 1)
    # g=10: 2*4*3^4 = 2*4*81 = 648? That seems high.
    # Actually the Moore bound for even girth g = 2m:
    # n >= 2 * sum_{i=0}^{m-1} (k-1)^i = 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(m-1))
    # For k=4, m=5: 2 * (1 + 3 + 9 + 27 + 81) = 2 * 121 = 242
    # So n >= 242, and best known is 320.
    
    # Let's try a random greedy approach to build a 4-regular graph of girth 10
    # on n vertices, starting from n=320 and going down.
    
    def greedy_construct(n, k, g, seed=42):
        rng = np.random.RandomState(seed)
        # Build adjacency list
        adj = [[] for _ in range(n)]
        # We need to add k/2 = 2 edges per vertex (for k=4)
        # Total edges = n*k/2 = 2n
        
        # Randomly permute vertices and try to add edges
        # Use a backtracking approach with random restarts
        
        def check_girth(adj_list, n, g):
            if not _verify_girth(n, adj_list, g):
                return False
            return True
        
        # Try multiple random seeds
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            seed = 42 + attempt
            rng = np.random.RandomState(seed)
            adj = [[] for _ in range(n)]
            
            # Randomly add edges
            # First, create a random permutation of all possible edges
            possible_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    possible_edges.append((i, j))
            rng.shuffle(possible_edges)
            
            # Try to add edges in random order, checking degree constraint
            # This is a simple greedy that won't find girth 10 likely, but let's try
            # Actually, let's use a different approach: start from a random 4-regular graph
            # and try to fix short cycles
            
            # Generate random 4-regular graph
            # Each vertex needs degree 4
            # Use configuration model
            half_edges = k * n // 2  # = 2n
            # Create stubs
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            rng.shuffle(stubs)
            
            # Pair stubs to form edges
            edges_set = set()
            adj = [[] for _ in range(n)]
            i = 0
            valid = True
            while i < len(stubs):
                u, v = stubs[i], stubs[i+1]
                i += 2
                if u == v:
                    valid = False
                    break
                if (min(u,v), max(u,v)) in edges_set:
                    valid = False
                    break
                edges_set.add((min(u,v), max(u,v)))
                adj[u].append(v)
                adj[v].append(u)
            
            if not valid:
                continue
            
            # Check girth
            if check_girth(adj, n, g):
                return n, [(u, v) for (u, v) in sorted(edges_set)]
        
        return None
    
    # Try various n values
    # Moore bound is 242, known best is 320
    for n in [320, 300, 280, 260, 242]:
        if time.time() > deadline - 10:
            break
        result = greedy_construct(n, k, g)
        if result is not None:
            return result
    
    return None


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin before 120s limit
    
    # Try to find a better construction
    try:
        result = _try_construct(k, g, deadline)
        if result is not None:
            n, edges = result
            # Verify it's valid
            if _verify_girth(n, edges, g):
                return n, edges
    except Exception:
        pass
    
    # Fallback: LUW construction
    if time.time() > deadline:
        # Even LUW might be slow for large d, but it's the fallback
        n, edges = _luw_construct(k, g)
        return n, edges
    
    n, edges = _luw_construct(k, g)
    return n, edges
# EVOLVE-BLOCK-END
