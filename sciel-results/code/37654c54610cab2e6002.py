"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct(k, g):
    # Use the Lazebnik-Ustimenko-Woldar (LUW) construction family, but scan a
    # broader set of parameters (q values and d offsets) to find the smallest
    # valid 3-regular graph with girth >= g.
    #
    # The LUW graph CD(d, q) is a q-regular bipartite graph with girth at least
    # d+5 (for odd d) or d+4 (for even d), depending on the specific variant.
    # We will try multiple (q, d) pairs and return the smallest valid graph.
    
    def build_luw(q, d):
        """Build the LUW graph with parameters q and d. Returns (n, edges) or None if invalid."""
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            # q must be prime for standard field arithmetic
            if q < 2:
                return None
            # Check if q is prime (simple trial division)
            if q < 2:
                return None
            is_prime = True
            for p in range(2, int(q**0.5) + 1):
                if q % p == 0:
                    is_prime = False
                    break
            if not is_prime:
                return None
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Compute the rule for each coordinate c = 2..d
        # For odd d, girth is at least d+5; for even d, at least d+4
        # We want girth >= g, so:
        #   if d is odd: d + 5 >= g  =>  d >= g - 5
        #   if d is even: d + 4 >= g  =>  d >= g - 4
        # We'll just build it and verify girth later.
        
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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
        index = {start: 0}
        order = [start]
        edges = []
        
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        return n, edges
    
    def verify_girth(n, edges, g, k):
        """Verify that the graph is k-regular, simple, and has girth >= g.
        Uses non-backtracking walks from every vertex."""
        if n == 0:
            return False
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        edge_set = set()
        for i, j in edges:
            if i == j:
                return False
            if i > j:
                i, j = j, i
            if (i, j) in edge_set:
                return False
            edge_set.add((i, j))
            adj[i].add(j)
            adj[j].add(i)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check girth using non-backtracking BFS from each vertex
        # For each vertex, do BFS avoiding backtracking to find shortest cycle
        for start in range(n):
            # BFS state: (vertex, prev_vertex)
            # Distance = number of edges in the non-backtracking walk
            from collections import deque
            visited = {}  # (vertex, prev) -> distance
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited[(nb, start)] = 1
            
            found_cycle = False
            while queue:
                v, prev, dist = queue.popleft()
                for nb in adj[v]:
                    if nb == prev:
                        continue  # non-backtracking
                    if nb == start and dist + 1 >= 3:
                        # Found a cycle of length dist + 1
                        if dist + 1 < g:
                            return False
                        found_cycle = True
                    elif (nb, v) not in visited:
                        visited[(nb, v)] = dist + 1
                        queue.append((nb, v, dist + 1))
        
        return True
    
    # Try a range of q values (primes and 4) and d values
    # For k=3, q=3 is the natural choice, but we can also try q=5, 7, etc.
    # However, the LUW graph is q-regular, so for k=3 we need q=3.
    # For general k, we need q=k.
    
    # Since the LUW construction gives a q-regular graph, for degree k we set q=k.
    # But we can also consider that the connected component might not be the whole graph,
    # or we might be able to use a different construction.
    
    # For now, let's try the LUW construction with q=k and various d values.
    # We need d such that the girth is at least g.
    # For odd d: girth >= d+5, so d >= g-5
    # For even d: girth >= d+4, so d >= g-4
    
    best_result = None
    best_n = float('inf')
    
    # Try d values around the minimum needed
    min_d_odd = max(3, g - 5)
    if min_d_odd % 2 == 0:
        min_d_odd += 1
    min_d_even = max(4, g - 4)
    if min_d_even % 2 == 1:
        min_d_even += 1
    
    # Try a range of d values
    d_candidates = set()
    for d in range(min_d_odd, min_d_odd + 10):
        if d >= 3:
            d_candidates.add(d)
    for d in range(min_d_even, min_d_even + 10):
        if d >= 4:
            d_candidates.add(d)
    
    for d in sorted(d_candidates):
        result = build_luw(k, d)
        if result is None:
            continue
        n, edges = result
        if n < best_n:
            if verify_girth(n, edges, g, k):
                best_result = (n, edges)
                best_n = n
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the original LUW construction
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
