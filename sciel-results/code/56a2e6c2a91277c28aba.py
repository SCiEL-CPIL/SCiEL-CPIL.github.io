"""
Cage graph construction: k-regular graph with girth >= g on as few vertices as possible.

Strategy:
1. Try known explicit families (LUW, incidence of generalized polygons, etc.)
2. If none produce a valid graph, fall back to a greedy/randomized construction
   that tries to build a k-regular graph with large girth.
3. Validate with non-backtracking BFS from every vertex.
"""
import time
import math
from collections import deque

def construct(k, g):
    deadline = time.time() + 110.0  # safety margin before 150s kill

    def check_deadline():
        return time.time() > deadline

    # --- Validate: simple, k-regular, girth >= g via non-backtracking BFS ---
    def validate(n, edges, k, g):
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # Non-backtracking BFS from each vertex to check girth
        for s in range(n):
            # State: (current_vertex, previous_vertex)
            dist = {}
            q = deque()
            for v in adj[s]:
                q.append((s, v))
                dist[(s, v)] = 1
            found_cycle = False
            while q and not found_cycle:
                u, v = q.popleft()
                d = dist[(u, v)]
                if d >= g:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue  # non-backtracking
                    if (v, w) in dist:
                        # Cycle found: length = dist[(v,w)] + 1
                        clen = dist[(v, w)] + 1
                        if clen < g:
                            return False
                    else:
                        dist[(v, w)] = d + 1
                        q.append((v, w))
        return True

    # --- LUW (Lazebnik-Ustimenko-Woldar) family ---
    def luw_family(k, g):
        """
        The LUW construction gives k-regular graphs with girth >= g for certain parameters.
        For k=5, g=9: use the known construction.
        """
        q = k
        # Determine depth d such that girth >= g
        # The LUW graph has girth related to the depth of the rule
        # For k=5, g=9, we need d such that 2*d+1 >= 9 -> d >= 4, but girth is actually 2*floor(d/2)+1 or similar
        # Let's try d = g - 2 or so
        # Actually, the LUW graph with parameter d has girth 2*floor(d/2)+1 for odd, 2*floor(d/2) for even? 
        # Let's just try several values of d and validate.
        
        for d in range(3, 10):
            if check_deadline():
                break
            # Build the rule
            rule = {}
            for c in range(2, d + 1):
                # Standard LUW rule
                r = (c - 5) % 4 if c >= 5 else 0
                c0 = c - (c - 5) % 4 if c >= 5 else c
                if c < 5:
                    rule[c] = (1, 1) if c == 2 else (2, 1) if c == 3 else (1, 2)
                else:
                    options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
                    rule[c] = options[r % 4]
            
            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        if side == 0:
                            y[c] = (x[c] + y[a] * x[b]) % q
                        else:
                            y[c] = (x[c] - x[a] * y[b]) % q
                    yield (1 - side, tuple(y))
            
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
            
            n = len(order)
            # Check if it's k-regular
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            if all(len(adj[u]) == k for u in range(n)):
                if validate(n, edges, k, g):
                    return n, edges
        return None

    # --- Try LUW first ---
    result = luw_family(k, g)
    if result is not None:
        return result

    # --- Fallback: Greedy construction ---
    # Try to build a k-regular graph with girth >= g using a greedy approach
    # Start with a large n and try to add edges while maintaining girth
    
    def greedy_construct(k, g, n):
        """
        Try to build a k-regular graph on n vertices with girth >= g.
        Use a greedy edge-adding approach.
        """
        import random
        random.seed(42)
        
        # Start with no edges
        adj = [set() for _ in range(n)]
        edges = []
        
        # First, create a random k-regular graph
        # Use configuration model
        half_edges = [i for i in range(n) for _ in range(k)]
        random.shuffle(half_edges)
        
        # Try to pair up half-edges to form a simple graph
        for attempt in range(100):
            if check_deadline():
                break
            shuffled = half_edges[:]
            random.shuffle(shuffled)
            adj = [set() for _ in range(n)]
            edges = []
            success = True
            for i in range(0, 2 * n * k, 2):
                u, v = shuffled[i], shuffled[i + 1]
                if u == v or v in adj[u]:
                    success = False
                    break
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
            if success:
                # Check girth
                if validate(n, edges, k, g):
                    return n, edges
        return None

    # Try a few values of n starting from Moore bound
    moore = 1 + k * sum(k ** i for i in range((g - 1) // 2))
    for n in range(moore, moore + 500, 10):
        if check_deadline():
            break
        result = greedy_construct(k, g, n)
        if result is not None:
            return result

    # Last resort: return a trivially valid but large graph (if we can find one)
    # For now, return the LUW result even if it failed validation (shouldn't happen)
    # Or construct a cycle-based graph
    n = 2 * g
    edges = [(i, (i + 1) % n) for i in range(n)]
    # This is 2-regular, not k-regular. Need to fix.
    # Just return something valid for k=2, g arbitrary
    if k == 2:
        return n, edges
    
    # For general k, this is a fallback that may not be valid
    # Return LUW-like structure
    result = luw_family(k, g)
    if result is not None:
        return result
    
    # Absolute fallback: return a small valid graph if possible
    # For k=5, g=9, this shouldn't be reached
    raise ValueError("Could not construct valid graph")

# EVOLVE-BLOCK-END
