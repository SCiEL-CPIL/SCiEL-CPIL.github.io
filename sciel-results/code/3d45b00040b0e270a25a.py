"""Construct k-regular graph of girth g with minimum vertices."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # First, get the LUW construction as baseline
    baseline_n, baseline_edges = _luw_construct(k, g)
    best_n = baseline_n
    best_edges = baseline_edges
    
    # Try to improve via search (only if we have time)
    if k == 4 and g == 10:
        # Try a bounded backtracking search for a smaller 4-regular girth-10 graph
        improved = _search_improve(k, g, best_n, deadline)
        if improved is not None:
            n, edges = improved
            if n < best_n:
                best_n = n
                best_edges = edges
    
    return best_n, best_edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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


def _search_improve(k, g, current_best, deadline):
    """
    Bounded backtracking search for a smaller k-regular graph of girth >= g.
    Uses constraint propagation to prune branches early.
    Returns (n, edges) if found, else None.
    """
    if time.time() > deadline:
        return None
    
    # Moore bound for lower bound
    if k == 4 and g == 10:
        # For even girth g, Moore bound: n >= k * (k-1)^(g/2 - 1) / (k-1) ... 
        # Actually for girth g (even): n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
        # For g=10, k=4: n >= 2*(1 + 3 + 9 + 27 + 81) = 2*121 = 242
        # Best lower bound is 243
        moore_bound = 243
    else:
        # General Moore bound
        if g % 2 == 0:
            moore_bound = 2 * sum((k-1)**i for i in range(g//2))
        else:
            moore_bound = k * sum((k-1)**i for i in range((g-1)//2))
    
    # We'll try to find graphs with n from moore_bound to current_best - 1
    # This is extremely hard, so we limit our search
    # For practical purposes with 4-regular girth-10, let's try a limited approach
    
    # Given the extreme difficulty, let's try a randomized approach with verification
    # Try a few random 4-regular graphs and check girth
    
    import random
    random.seed(42)
    
    # Try a few candidate sizes
    target_sizes = []
    # Start from just above Moore bound
    for n_candidate in range(moore_bound, min(current_best, moore_bound + 50)):
        if n_candidate % 2 == 0:  # Must be even for 4-regular
            target_sizes.append(n_candidate)
    
    # Limit search to a few candidates
    target_sizes = target_sizes[:10]
    
    for n in target_sizes:
        if time.time() > deadline:
            return None
        result = _try_random_k_regular(k, g, n, deadline)
        if result is not None:
            return result
    
    return None


def _try_random_k_regular(k, g, n, deadline):
    """
    Try to construct a random k-regular graph on n vertices and verify girth >= g.
    Uses the configuration model with girth checking.
    """
    if time.time() > deadline:
        return None
    
    import random
    
    # Use configuration model: create n stubs of degree k, pair them up
    # Then check for girth violation
    
    max_attempts = 50  # Limit attempts per n
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Build random k-regular graph using configuration model
        stubs = []
        for v in range(n):
            for _ in range(k):
                stubs.append(v)
        
        # Randomly pair stubs
        edges = set()
        valid = True
        random.shuffle(stubs)
        
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
        
        if not valid:
            continue
        
        # Check girth using BFS from each vertex (limited depth)
        if _check_girth(n, list(edges), g):
            return n, list(edges)
    
    return None


def _check_girth(n, edges, g):
    """Check if graph has no cycle shorter than g using BFS from each vertex."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS up to depth (g-1)//2 and check for cycles
    # A cycle of length < g means there's a path of length < g/2 back to start
    # More precisely: for even g, check for cycles of length < g
    # Use non-backtracking walks
    
    max_depth = g // 2 - 1  # For girth g, no cycle shorter than g
    
    for start in range(n):
        # BFS with parent tracking to avoid backtracking
        # State: (vertex, parent)
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1))
        
        while queue:
            vertex, parent, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            for neighbor in adj[vertex]:
                if neighbor == parent:
                    continue
                if neighbor == start and depth + 1 < g:
                    # Found a cycle shorter than g
                    return False
                state = (neighbor, vertex)
                if state not in visited:
                    visited.add(state)
                    queue.append((neighbor, vertex, depth + 1))
    
    return True

# EVOLVE-BLOCK-END
