import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    # For odd girth g, we need girth >= g.
    # LUW CD(d,q) has girth d+5 if d is odd, d+6 if d is even.
    # We want girth >= g.
    # If we pick d such that d+5 >= g (for odd d) or d+6 >= g (for even d).
    # To minimize n = 2*q^d, we want smallest d.
    
    # Try d = g - 5. If g-5 is odd, girth is g.
    # If g-5 is even, girth is g+1.
    # In both cases, girth >= g.
    # So d = g - 5 is always valid?
    # Check d >= 3.
    d = g - 5
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q

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
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
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


def _check_girth(adj, n, g):
    """Check if graph has girth at least g. Returns True if valid."""
    # BFS from each vertex to find shortest cycle.
    # For girth g, we need to ensure no cycle of length < g.
    # We can stop BFS at depth (g-1)//2.
    max_depth = (g - 1) // 2
    
    for v in range(n):
        # dist[v][prev] = depth
        dist = {}
        queue = deque()
        for nb in adj[v]:
            queue.append((nb, v, 1))
            dist[(nb, v)] = 1
            
        while queue:
            cur, prev, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == v:
                    # Cycle of length depth + 1
                    if depth + 1 < g:
                        return False
                    continue
                if (nb, cur) in dist:
                    # Cycle of length dist[(nb, cur)] + depth + 1
                    cycle_len = dist[(nb, cur)] + depth + 1
                    if cycle_len < g:
                        return False
                else:
                    dist[(nb, cur)] = depth + 1
                    queue.append((nb, cur, depth + 1))
    return True


def _local_search_5_9(max_time=100.0):
    """Attempt to find a 5-regular graph with girth >= 9 and fewer vertices than LUW."""
    start_time = time.time()
    deadline = start_time + max_time
    
    # LUW baseline
    n_luw, edges_luw = _luw_construct(5, 9)
    best_n = n_luw
    best_edges = edges_luw
    
    # Try candidate sizes
    # Moore bound is 106.
    # Try a range of sizes.
    candidate_ns = [106, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
    candidate_ns = [n for n in candidate_ns if n % 2 == 0]
    candidate_ns = sorted(set(candidate_ns))
    
    random.seed(42)
    
    for n in candidate_ns:
        if time.time() > deadline - 5:
            break
        if n >= best_n:
            break
        
        half = n // 2
        max_restarts = 20
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Generate random 5-regular bipartite graph
            left_stubs = []
            for i in range(half):
                left_stubs.extend([i] * 5)
            right_stubs = []
            for j in range(half):
                right_stubs.extend([j + half] * 5)
            
            random.shuffle(left_stubs)
            random.shuffle(right_stubs)
            
            adj = {i: set() for i in range(n)}
            for ls, rs in zip(left_stubs, right_stubs):
                if rs not in adj[ls]:
                    adj[ls].add(rs)
                    adj[rs].add(ls)
                else:
                    # Duplicate edge, need to fix or reject
                    # For simplicity, reject if duplicates occur in initial generation
                    # But configuration model can produce duplicates.
                    # We'll just check degree later.
                    pass
            
            # Check degree
            if any(len(adj[i]) != 5 for i in range(n)):
                continue
            
            # Check girth
            if _check_girth(adj, n, 9):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if n < best_n:
                    best_n = n
                    best_edges = edges
                break # Found valid for this n
    
    return best_n, best_edges


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 115.0
    
    if k == 5 and g == 9:
        try:
            n, edges = _local_search_5_9(max_time=110.0)
            if time.time() < deadline:
                return n, edges
        except Exception:
            pass
    
    n, edges = _luw_construct(k, g)
    return n, edges

# EVOLVE-BLOCK-END
