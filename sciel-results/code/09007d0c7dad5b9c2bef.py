"""Algebraic construction: Cayley graph over product of cyclic groups."""
import random
from collections import deque

def _cayley_girth(gen_set, n1, n2, max_girth):
    """Compute girth of Cayley graph on Z_n1 x Z_n2 with given generators."""
    # BFS from identity to find shortest cycle
    # A cycle of length L exists if there's a word of length L in generators
    # that returns to identity, with no proper prefix returning to identity
    
    # For undirected Cayley graph, generators include inverses
    # gen_set should be a list of (a,b) elements
    # The graph is undirected, so if (a,b) is a generator, (-a,-b) is also
    
    start = (0, 0)
    # BFS tracking paths to detect cycles
    # State: (vertex, last_generator_index) to avoid backtracking
    # Actually, for girth computation, we need non-backtracking walks
    
    # Use BFS with state = (current_vertex, previous_vertex)
    # Start: for each generator g, state = (g, start)
    # This gives non-backtracking walks of length 1
    
    visited = {}  # (vertex, prev) -> (depth, path_start)
    queue = deque()
    
    # Initialize with all neighbors of start
    for g in gen_set:
        nv = ((start[0] + g[0]) % n1, (start[1] + g[1]) % n2)
        if nv == start:
            # Self-loop, girth = 1 (but shouldn't happen for simple graph)
            return 1
        if nv not in visited or (nv, start) not in visited:
            visited[(nv, start)] = (1, [start, nv])
            queue.append(((nv, start), [start, nv]))
    
    while queue:
        (curr, prev), path = queue.popleft()
        depth = len(path) - 1
        if depth >= max_girth:
            continue
        
        # Try each generator
        for g in gen_set:
            # New vertex
            nv = ((curr[0] + g[0]) % n1, (curr[1] + g[1]) % n2)
            # Non-backtracking: nv != prev
            if nv == prev:
                continue
            
            if nv == start:
                # Found a cycle of length depth + 1
                return depth + 1
            
            state = (nv, curr)
            if state not in visited:
                visited[state] = (depth + 1, path + [nv])
                queue.append((state, path + [nv]))
    
    return float('inf')

def _is_simple_cayley(gen_set, n1, n2):
    """Check if Cayley graph is simple (no loops, no multi-edges)."""
    start = (0, 0)
    seen = set()
    for g in gen_set:
        nv = ((start[0] + g[0]) % n1, (start[1] + g[1]) % n2)
        if nv == start:
            return False  # Loop
        key = (min(nv, (n1 - g[0]) % n1, (n2 - g[1]) % n2) if False else (nv[0], nv[1]))
        # For undirected, we consider g and -g as same edge
        # Check for multi-edges: two different generators giving same neighbor
        if nv in seen:
            return False
        seen.add(nv)
    return True

def construct(k, g):
    """Construct a k-regular graph with girth >= g using algebraic methods."""
    if k != 3:
        return _construct_luw(k, g)
    
    girth_target = g
    rng = random.Random(42)
    
    # For 3-regular undirected graph, we need a generating set S such that
    # |S| = 3, S = -S (closed under inverses), and 0 not in S
    # Over Z_n1 x Z_n2, we pick 3 generators
    
    # Try various group sizes
    # Moore bound for (3, 16): ~382 vertices
    # Known best: 936 vertices
    
    # Strategy: search over group sizes and generator sets
    # The graph has n1*n2 vertices
    
    # For efficiency, we'll try to find a construction with n1*n2 close to Moore bound
    
    # Common approach: use Z_p x Z_q or Z_p x Z_p
    # Let's try Z_p x Z_q where p and q are primes
    
    best_result = None
    
    # Try various factorizations
    # n1 * n2 should be even for 3-regular (actually not required, but helpful)
    # Let's try n1 in [10..50], n2 in [10..100]
    
    for n1 in range(8, 40):
        for n2 in range(8, 80):
            total = n1 * n2
            if total < 380 or total > 1000:
                continue
            if total % 2 != 0:
                continue  # Prefer even for 3-regular
            
            # Try random generator sets
            for trial in range(20):
                # Pick 3 random generators
                gens = []
                for _ in range(3):
                    a = rng.randint(1, n1 - 1)
                    b = rng.randint(1, n2 - 1)
                    gens.append((a, b))
                
                # Check simplicity
                if not _is_simple_cayley(gens, n1, n2):
                    continue
                
                # Compute girth
                girth = _cayley_girth(gens, n1, n2, girth_target)
                if girth >= girth_target:
                    # Valid! Build the edge list
                    edges = []
                    for i in range(n1):
                        for j in range(n2):
                            v = i * n2 + j
                            for g_ in gens:
                                ni = (i + g_[0]) % n1
                                nj = (j + g_[1]) % n2
                                w = ni * n2 + nj
                                if v < w:
                                    edges.append((v, w))
                    
                    if best_result is None or total < best_result[0]:
                        best_result = (total, edges)
                    
                    # Move to next size
                    break
        
        # Early exit if we found a good result
        if best_result and best_result[0] <= 500:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback
    return _construct_luw(k, g)

def _construct_luw(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
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
