"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(k, g):
    # Lazy-Lazebnik-Ustimenko-Woldar construction for small cases, randomized for larger.
    # The evaluator calls construct(k, g) with specific k and g.
    # The current seed uses LUW(q=k, d=...) which works for specific small cases.
    # We need a general approach.
    #
    # Strategy:
    # 1. If k is small and g is small, try known constructions (LUW, etc.)
    # 2. For general cases, use a randomized search with configuration model
    # 3. Add a deadline guard to ensure we return something
    
    t_start = time.time()
    deadline = 115  # Leave 5s margin
    
    def moore_bound(k, g):
        # Moore bound for girth g
        # If g is odd: 1 + k * (1 + k + ... + k^((g-1)/2))
        # If g is even: 2 * (1 + k + ... + k^((g-2)/2))
        if g % 2 == 1:
            half = (g - 1) // 2
            return 1 + k * sum(k**i for i in range(half))
        else:
            half = (g - 2) // 2
            return 2 * sum(k**i for i in range(half + 1))
    
    def check_girth(edges, n, min_girth):
        """Check if graph has no cycle shorter than min_girth.
        Returns True if girth >= min_girth, False otherwise."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # BFS from each vertex to find shortest cycle
        # For efficiency, we can use the fact that if there's a cycle of length < min_girth,
        # we'll find it during BFS
        visited = [None] * n  # visited[v] = (dist, parent)
        
        for start in range(n):
            # BFS from start
            visited = [None] * n
            visited[start] = (0, -1)
            queue = [start]
            head = 0
            
            while head < len(queue):
                u = queue[head]
                head += 1
                du, pu = visited[u]
                
                if du + 1 >= min_girth:
                    # No need to explore further from this vertex for cycles
                    # shorter than min_girth (any cycle found would be longer)
                    break
                
                for v in adj[u]:
                    if v == pu:
                        continue  # Don't go back to parent
                    
                    if visited[v] is None:
                        visited[v] = (du + 1, u)
                        queue.append(v)
                    else:
                        dv, pv = visited[v]
                        # Found a cycle
                        cycle_len = du + dv + 1
                        if cycle_len < min_girth:
                            return False
        
        return True
    
    def generate_random_regular(k, n, rng):
        """Generate a random k-regular simple graph on n vertices using configuration model.
        Returns edges list or None if failed."""
        # Configuration model: create k stubs for each vertex, pair them randomly
        for _ in range(100):  # Try up to 100 times
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            rng.shuffle(stubs)
            
            # Pair stubs
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
            
            if valid:
                return list(edges)
        
        return None
    
    def check_degree(edges, n, k):
        """Check if all vertices have degree exactly k."""
        deg = [0] * n
        for i, j in edges:
            deg[i] += 1
            deg[j] += 1
        return all(d == k for d in deg)
    
    # Try LUW construction for small k, g
    # The LUW graph CD(d, q) has girth at least d+5 for odd d
    # We want girth >= g, so we need d+5 >= g, i.e., d >= g-5
    # For the construction to work, we need q = k (prime power)
    
    # First, try a few known small cases with explicit constructions
    # For k=3, g=5: Petersen graph (n=10)
    if k == 3 and g == 5:
        # Petersen graph
        edges = [(0,1),(1,4),(4,2),(2,3),(3,0),  # outer pentagon
                 (5,6),(6,7),(7,8),(8,9),(9,5),  # inner pentagram
                 (0,5),(1,6),(2,7),(3,8),(4,9)]
        return 10, edges
    
    # For k=3, g=7: Heawood graph (n=14)
    if k == 3 and g == 7:
        # Heawood graph is the incidence graph of Fano plane
        edges = []
        points = [(0,0,1),(0,1,0),(1,0,0),(1,1,1),(1,0,1),(0,1,1),(1,1,0)]
        # Actually, let me use the standard representation
        # Heawood graph vertices: 7 points + 7 lines of Fano plane
        # This is getting complex, let's just do randomized search for general case
        
    # General approach: randomized search
    moore = moore_bound(k, g)
    
    # Ensure n*k is even for regular graph
    def next_valid_n(n):
        while n * k % 2 != 0:
            n += 1
        return n
    
    best_result = None
    
    # Try from Moore bound upwards
    n = next_valid_n(moore)
    max_n = moore + 100  # Try up to 100 vertices above Moore bound
    
    rng = random.Random(42)
    
    while n <= max_n and time.time() < deadline:
        # Try multiple random seeds
        for attempt in range(50):
            if time.time() >= deadline:
                break
            
            edges = generate_random_regular(k, n, rng)
            if edges is None:
                continue
            
            if not check_degree(edges, n, k):
                continue
            
            if check_girth(edges, n, g):
                return n, edges
        
        n += 1
        if n * k % 2 != 0:
            n += 1
    
    # Fallback: LUW construction if available
    # This is a simplified version of the seed's LUW
    q = k
    if q in [2, 3, 4, 5, 7, 8, 9, 11, 13]:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
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
                    y[c] = add_op(x[c], mul_op(y[a], x[b])) if side == 0 else sub_op(x[c], mul_op(x[a], y[b]))
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
        
        n_luw = len(order)
        if check_girth(edges, n_luw, g):
            return n_luw, edges
    
    # Last resort: return a trivial result (shouldn't happen)
    raise RuntimeError("Failed to construct valid graph")

# EVOLVE-BLOCK-END
