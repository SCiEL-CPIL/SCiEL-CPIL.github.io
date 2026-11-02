import time
import random
import numpy as np

def _girth_is_at_least_g(n, edges, g):
    """Check if the girth is at least g using non-backtracking walks from every vertex."""
    if n == 0:
        return True
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS on non-backtracking walks up to depth g-1
    # If we find a cycle of length < g, return False
    for start in range(n):
        # State: (current_vertex, previous_vertex)
        # Start: (start, -1)
        visited = set()
        queue = [(start, -1)]
        visited.add((start, -1))
        depth = 0
        while queue and depth < g - 1:
            next_queue = []
            for u, prev in queue:
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == start and depth + 1 < g:
                        # Found a cycle of length depth + 1 < g
                        return False
                    if v == start:
                        # Cycle of length depth + 1 >= g, ok
                        continue
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        next_queue.append(state)
            queue = next_queue
            depth += 1
    return True

def _is_valid(n, offsets, k, g):
    """Check if the cyclic graph defined by n and offsets is valid."""
    if len(offsets) != k // 2:
        return False
    # Check no zero offset (self-loop)
    for off in offsets:
        if off % n == 0:
            return False
    
    # Build edges
    edges = []
    for i in range(n):
        for off in offsets:
            j = (i + off) % n
            if i < j:
                edges.append((i, j))
    
    # Check simple graph: no repeated edges
    edge_set = set()
    for u, v in edges:
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        edge_set.add((u, v))
    
    # Check degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return False
    
    # Check girth
    if not _girth_is_at_least_g(n, edges, g):
        return False
    
    return True

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    # Moore bound: minimum number of vertices
    if g % 2 == 0:
        moore_bound = 1 + k * sum((k-1)**i for i in range(g//2 - 1))
    else:
        moore_bound = 2 * sum((k-1)**i for i in range((g-1)//2 - 1))
    
    max_vertices = 2047 # Search up to just before the fallback size
    start_time = time.time()
    time_limit = 100.0 # Leave 20s for fallback and overhead
    
    random.seed(42)
    
    # Search for a cyclic graph
    for n in range(max(moore_bound, k+1), max_vertices + 1):
        if time.time() - start_time > time_limit:
            break
            
        attempts = 0
        max_attempts = 1000
        found = False
        
        while attempts < max_attempts and not found:
            attempts += 1
            
            # Generate k/2 random distinct non-zero offsets mod n
            # Ensure symmetry
            valid_offsets = list(range(1, n))
            if n % 2 == 0:
                valid_offsets.remove(n // 2)
            
            if len(valid_offsets) < k // 2:
                break
            
            offsets = random.sample(valid_offsets, k // 2)
            
            # Fast pre-check: linear relations
            # For k=4, offsets are a, b.
            # Check if c1*a + c2*b == 0 mod n for small c1, c2
            # This catches cycles of length < 7 quickly.
            a, b = offsets[0], offsets[1]
            valid = True
            for c1 in range(-3, 4):
                for c2 in range(-3, 4):
                    if c1 == 0 and c2 == 0:
                        continue
                    if (c1 * a + c2 * b) % n == 0:
                        # Check if this corresponds to a short non-backtracking walk
                        # If c1, c2 are small, the walk length is |c1| + |c2|
                        # But backtracking might cancel steps.
                        # For now, reject if any small linear combo is 0.
                        # This is a necessary condition for girth >= 7.
                        # For girth >= 10, we need to be stricter, but this is a fast filter.
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
                
            if _is_valid(n, offsets, k, g):
                # Build and return edges
                edges = []
                for i in range(n):
                    for off in offsets:
                        j = (i + off) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # Fallback: return a simple construction (may not be optimal but valid)
    # Use the Lazebnik-Ustimenko-Woldar approach as fallback
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
