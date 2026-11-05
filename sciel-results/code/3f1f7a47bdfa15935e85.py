"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def construct(k, g):
    # Cycle-avoidant random 3-regular graph construction via the configuration model.
    # We build a simple k-regular graph on n vertices by sequentially pairing stubs,
    # using rejection sampling to forbid creating any non-backtracking cycle of length < g.
    #
    # Strategy:
    # 1. Determine a reasonable n (start from Moore bound, try increasing n).
    # 2. Use the configuration model: create n*k stubs, randomly pair them.
    # 3. During pairing, check if creating an edge would form a non-backtracking cycle of length < g.
    #    If so, reject and try another pairing.
    # 4. Return the resulting graph.
    
    # Moore bound for (k, g): minimum possible vertices
    # For even girth g = 2r: n >= k + (k-1) * sum_{i=0}^{r-1} (k-1)^i
    # For odd girth g = 2r+1: n >= 2 * (k-1) * sum_{i=0}^{r-1} (k-1)^i + ... 
    # Actually for (3,16), girth 16 is even, r=8
    # Moore bound: n >= 3 + 2 * sum_{i=1}^{7} 2^i = 3 + 2*(2^8 - 2) = 3 + 2*254 = 511
    # So n >= 512 (must be even for 3-regular)
    
    def moore_bound(k, g):
        if g % 2 == 0:
            r = g // 2
            # n >= k + (k-1) * ((k-1)^(r-1) - 1) / (k-2) for k>2
            if k == 2:
                return 2 * r
            else:
                s = 0
                for i in range(r - 1):
                    s += (k - 1) ** i
                return k + (k - 1) * s
        else:
            r = (g - 1) // 2
            if k == 2:
                return 2 * r + 1
            else:
                s = 0
                for i in range(r):
                    s += (k - 1) ** i
                return 2 * (k - 1) * s + k
        # Round up to even
        mb = int(moore_bound(k, g))
        if mb % 2 != 0:
            mb += 1
        return mb
    
    # Try to construct with various n values starting from a reasonable lower bound
    # For (3,16), the known best is 936. Let's try from around 600 upward.
    
    max_vertices = 2000  # Upper bound to try
    
    # Start from a value that's likely to work
    # For k=3, g=16: try n values that are even and >= moore bound
    mb = moore_bound(k, g)
    
    # We'll try several candidate n values
    candidate_ns = []
    # Start from max(mb, some heuristic) up to max_vertices
    start_n = max(mb, 512)
    # Make sure start_n is even
    if start_n % 2 != 0:
        start_n += 1
    
    # Try a range of n values
    for n in range(start_n, max_vertices + 1, 2):
        if n % 2 != 0:  # 3-regular requires even n
            continue
        result = try_construct(n, k, g)
        if result is not None:
            return result
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction
    return _lw_graph(k, g)

def try_construct(n, k, g):
    """Try to construct a k-regular graph on n vertices with girth >= g using configuration model."""
    
    # Create stubs: vertex i has k stubs
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    
    total_stubs = len(stubs)  # n * k
    assert total_stubs % 2 == 0
    
    # We'll pair stubs sequentially
    # Maintain adjacency list
    adj = [[] for _ in range(n)]
    
    # To check for non-backtracking cycles efficiently, we maintain the graph as we build it
    # When adding edge (u, v), we need to check if there's a non-backtracking path from u to v
    # of length < g-1 that doesn't use the edge (u,v) itself.
    
    # For efficiency, we'll use BFS/DFS with non-backtracking constraint
    # But this could be slow. Let's use a smarter approach.
    
    # Actually, for the configuration model, we can do:
    # 1. Randomly shuffle stubs
    # 2. Pair them up sequentially
    # 3. For each pair, check if it creates a forbidden cycle
    # 4. If yes, reject and try a different pairing
    
    # This is complex. Let's use a simpler approach:
    # Generate a random 3-regular graph, then check girth. If girth < g, retry.
    # But this is very inefficient for large g.
    
    # Better approach: Use the "girth-checking" during construction.
    # When adding edge (u,v), do a non-backtracking BFS from u up to depth g-2,
    # and check if v is reachable.
    
    def has_nonbacktracking_path(u, v, max_depth, adj):
        """Check if there's a non-backtracking path from u to v of length <= max_depth."""
        if u == v:
            return False  # No loops
        # BFS with state (current, previous) to enforce non-backtracking
        visited = set()
        queue = deque()
        # Start: from u, can go to any neighbor except "previous" (none at start)
        for neighbor in adj[u]:
            if neighbor == v:
                # Direct edge exists, but we're checking if adding (u,v) creates a cycle
                # The path would be of length 1, which means u-v is already an edge
                # But we're adding (u,v), so this shouldn't happen if we track properly
                pass
            if (u, neighbor) not in visited:
                visited.add((u, neighbor))
                queue.append((neighbor, u, 1))
        
        while queue:
            curr, prev, depth = queue.popleft()
            if curr == v and depth >= 2:
                # Found a non-backtracking path from u to v of length depth
                # Adding edge (u,v) would create a cycle of length depth + 1
                return True
            if depth >= max_depth:
                continue
            for neighbor in adj[curr]:
                if neighbor == prev:
                    continue  # Non-backtracking: don't go back
                if (curr, neighbor) not in visited:
                    visited.add((curr, neighbor))
                    queue.append((neighbor, curr, depth + 1))
        
        return False
    
    # Shuffle stubs
    random.shuffle(stubs)
    
    # Pair stubs
    edges = []
    used = [False] * total_stubs
    
    # For each stub, find a partner
    for i in range(0, total_stubs, 2):
        # Find an unused stub
        while used[i]:
            i += 1
        if i >= total_stubs:
            return None
        
        u = stubs[i]
        # Find a partner for stub i
        candidates = []
        for j in range(i + 1, total_stubs):
            if not used[j]:
                v = stubs[j]
                # Check if u == v (loop) - reject
                if u == v:
                    continue
                # Check if edge (u,v) already exists - reject
                if v in adj[u]:
                    continue
                # Check if adding (u,v) creates a non-backtracking cycle of length < g
                # A cycle of length L is formed if there's a non-backtracking path from u to v
                # of length L-1 in the current graph.
                # We want to avoid cycles of length < g, so L < g, i.e., L-1 < g-1.
                # So check for non-backtracking path of length <= g-2.
                if has_nonbacktracking_path(u, v, g - 2, adj):
                    continue
                candidates.append(j)
        
        if not candidates:
            return None  # Backtrack needed; for simplicity, fail
        
        # Pick a random candidate
        j = random.choice(candidates)
        v = stubs[j]
        
        # Add edge (u, v)
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        used[i] = True
        used[j] = True
    
    # Verify the graph is valid
    # Check degrees
    for v in range(n):
        if len(adj[v]) != k:
            return None
        # Check for duplicate edges
        if len(set(adj[v])) != k:
            return None
    
    # Check girth
    if check_girth(n, adj, g):
        return n, edges
    
    return None

def check_girth(n, adj, g):
    """Check if the graph has no cycle shorter than g."""
    # For each vertex, do a non-backtracking BFS up to depth g//2
    # If we find a cycle of length < g, return False
    
    for start in range(n):
        # Non-backtracking BFS from start
        visited = set()
        queue = deque()
        # State: (current, previous, depth)
        for neighbor in adj[start]:
            if (start, neighbor) not in visited:
                visited.add((start, neighbor))
                queue.append((neighbor, start, 1))
        
        while queue:
            curr, prev, depth = queue.popleft()
            if curr == start and depth + 1 < g:
                # Found a cycle of length depth + 1 < g
                return False
            if depth >= g - 1:
                continue
            for neighbor in adj[curr]:
                if neighbor == prev:
                    continue
                if (curr, neighbor) not in visited:
                    visited.add((curr, neighbor))
                    queue.append((neighbor, curr, depth + 1))
    
    return True

def _lw_graph(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph."""
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
# EVOLVE-BLOCK-END
