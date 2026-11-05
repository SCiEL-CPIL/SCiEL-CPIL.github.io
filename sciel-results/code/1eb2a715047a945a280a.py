"""Construct k-regular graph of girth g via random Cayley graphs on non-abelian groups."""
# EVOLVE-BLOCK-START
import random
import itertools
import time
from collections import deque

def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using non-backtracking walks from every vertex."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Non-backtracking BFS from each vertex up to depth (g-1)//2
    max_depth = (g - 1) // 2
    
    for start in range(n):
        # State: (node, prev_node, depth)
        # Track visited states to detect cycles
        visited = set()
        queue = deque()
        # Start with all neighbors of start (non-backtracking: no prev)
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start and depth + 1 < g:
                    return False  # Found a cycle shorter than g
                state = (nb, node)
                if state in visited:
                    continue
                visited.add(state)
                queue.append((nb, node, depth + 1))
    return True

def _cayley_graph_from_group(group_elements, generators, index_map):
    """Build Cayley graph from group elements and generators."""
    n = len(group_elements)
    edges = []
    seen_edges = set()
    for u in group_elements:
        for gen in generators:
            v = group_elements.index(u * gen)
            edge = (min(index_map[u], index_map[v]), max(index_map[u], index_map[v]))
            if edge not in seen_edges:
                seen_edges.add(edge)
                edges.append(edge)
    return n, edges

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Generate random Cayley graphs on non-abelian groups using
    involutory generators, verify girth via non-backtracking BFS.
    Fall back to L-U-W construction if random search fails.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    random.seed(42)
    start_time = time.time()
    time_limit = 110  # seconds
    
    # Moore bound
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # Fallback: L-U-W construction
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
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
    
    # For k=5, g=9: try random Cayley graphs on (Z/2)^m x S_3 or similar
    # We need exactly k involutory generators
    
    # Approach: Use group (Z/2)^m for m large enough, pick k random involutions
    # (all elements in (Z/2)^m are involutions)
    
    # For girth g, we need the Cayley graph to have girth >= g
    # The girth of a Cayley graph on (Z/2)^m with generators S is related
    # to the shortest non-trivial word in S that equals identity
    
    # Let's try different group sizes
    best_result = None
    best_n = float('inf')
    
    # Try (Z/2)^m for various m
    # Number of vertices = 2^m
    # We need 2^m to be reasonable (not too large for verification)
    
    max_m = 12  # 2^12 = 4096 vertices, manageable
    
    for m in range(8, max_m + 1):
        if time.time() - start_time > time_limit:
            break
        
        n = 2 ** m
        if n > 100000:
            break
        
        # Try random sets of k involutory generators
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Pick k random distinct non-identity elements
            # In (Z/2)^m, every element is its own inverse (involution)
            # We need to avoid the identity and ensure no generator is sum of others
            # (to avoid short relations)
            
            gens = set()
            while len(gens) < k:
                val = random.randint(1, n - 1)
                gens.add(val)
            
            gens = list(gens)
            
            # Build adjacency: vertex v connects to v XOR g for each g in gens
            # This gives a k-regular graph on 2^m vertices
            
            # Quick check: verify degree and no loops
            # Loop check: g != 0 for all g (already ensured)
            # Degree: each vertex has exactly k neighbors (one per generator)
            
            # Check for multi-edges: need all gens distinct (ensured)
            # and no generator equals another (ensured by set)
            
            # Verify girth using non-backtracking BFS
            # For (Z/2)^m, we can do this efficiently
            
            # Build edge list
            edges = []
            for v in range(n):
                for gen in gens:
                    u = v ^ gen
                    if v < u:
                        edges.append((v, u))
            
            # Verify girth
            if _verify_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
                    # Don't break immediately - try smaller m first
                    break  # Found valid graph for this m, try smaller m
        
        if best_result is not None and best_n <= n:
            # We found a valid graph, check if we can do better with smaller m
            # But we're iterating m from 8 upward, so smaller m already tried
            pass
    
    # If we found a valid random Cayley graph, use it
    if best_result is not None:
        return best_result
    
    # Fallback to L-U-W
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
