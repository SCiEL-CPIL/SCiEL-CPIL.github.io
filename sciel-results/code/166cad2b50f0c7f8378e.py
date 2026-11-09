import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    if k == 3 and g == 16:
        # Strategy 1: Cayley graph search over cyclic groups of prime order
        # A 3-regular Cayley graph on Z_p with generators {±a, ±b, ±c} where a+b+c=0
        # This structure can avoid short cycles more effectively than random graphs
        
        best_n = None
        best_edges = None
        
        # Try primes in a range that might work
        primes_to_try = _primes_in_range(500, 1200)
        
        for p in primes_to_try:
            if time.time() > deadline - 15:
                break
            
            # For each prime, try random generator sets
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break
                
                # Pick random a, b with a+b+c=0 => c = -(a+b) mod p
                # Ensure a, b, c are distinct and non-zero
                a, b = random.sample(range(1, p), 2)
                c = (-a - b) % p
                if c == 0 or c == a or c == b:
                    continue
                # Also ensure -a, -b, -c are distinct from a, b, c
                gens = [a, -a % p, b, -b % p, c, -c % p]
                if len(set(gens)) != 6:
                    continue
                
                # Build the Cayley graph
                edges = _build_cayley_graph(p, [a, b, c])
                if edges is None:
                    continue
                
                # Check girth
                if _check_girth_fast(edges, p, g):
                    return p, edges
        
        # Strategy 2: Fallback to LUW construction
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)

def _primes_in_range(start, end):
    """Generate primes in [start, end]."""
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = []
    for n in range(max(2, start), end + 1):
        if is_prime(n):
            primes.append(n)
    return primes

def _build_cayley_graph(p, gens):
    """Build a Cayley graph on Z_p with given generators. Returns edges or None if invalid."""
    # Each vertex i is connected to i+g and i-g for each g in gens
    # But we only need 3 edges per vertex, so we use 3 generators
    # Actually, for a 3-regular graph, we need exactly 3 neighbors per vertex
    
    # For a 3-regular Cayley graph, we can use 3 generators where
    # the graph is defined by: vertex i connects to i+g1, i+g2, i+g3
    # But this gives degree 3 only if we don't also connect to i-g1, etc.
    # 
    # Better: use a 3-regular structure. One way:
    # Vertex i connects to i+a, i+b, i+c where a, b, c are chosen such that
    # the resulting graph is 3-regular and simple.
    #
    # Actually, for a 3-regular Cayley graph on Z_p, we need exactly 3 generators
    # and the graph has edges (i, i+g) for each generator g and each vertex i.
    # But this gives degree 3 only if we count both directions properly.
    #
    # Let me reconsider: A Cayley graph with generator set S has edges (g, gs) for all g in G, s in S.
    # If |S| = 3, then each vertex has degree 3 (out-degree 3, but in an undirected graph,
    # we need S to be closed under inverses for the graph to be undirected).
    #
    # For an undirected 3-regular Cayley graph, we need |S| = 3 and S = S^{-1},
    # which means S contains elements that are their own inverses (order 2) or
    # we use a different construction.
    #
    # Alternative: Use a 6-generator set {±a, ±b, ±c} but identify edges properly.
    # Each vertex i connects to i±a, i±b, i±c, but this gives degree 6, not 3.
    #
    # I think the correct approach for 3-regular is to use a different structure.
    # Let me use a bipartite-like construction or a different method.
    
    # For now, let's use a simpler approach: random 3-regular graph with local search
    # But that's what the original code does. Let me think of a better Cayley approach.
    
    # Actually, for a 3-regular graph, we can use a Cayley graph with 3 generators
    # where the generators are chosen such that the graph is simple and 3-regular.
    # The key is that in an undirected Cayley graph, if g is a generator, so is g^{-1}.
    # So for 3-regular, we need 3 generators that are closed under inverses.
    # This means we can have at most 1 generator of order 2 (self-inverse) and
    # pairs of generators that are inverses of each other.
    # So 3-regular Cayley graphs on cyclic groups require special care.
    
    # Let me try a different structural approach: use a circulant graph.
    # A circulant graph C_n(a1, a2, a3) has vertices 0..n-1 and edges (i, i+aj) for j=1,2,3.
    # This is 3-regular if the aj are distinct and non-zero mod n.
    
    if len(gens) != 3:
        return None
    
    a, b, c = gens
    # Ensure all are non-zero and distinct
    if a == 0 or b == 0 or c == 0:
        return None
    if len({a, b, c}) != 3:
        return None
    # Also ensure that a, b, c and their negatives are all distinct (for simplicity)
    # Actually, for a circulant graph, we just need the differences to be distinct
    
    edges = []
    adj = [set() for _ in range(p)]
    
    for i in range(p):
        for g in [a, b, c]:
            j = (i + g) % p
            if i < j:  # Avoid duplicate edges
                if j not in adj[i]:
                    edges.append((i, j))
                    adj[i].add(j)
                    adj[j].add(i)
                else:
                    # Multi-edge, invalid
                    return None
            elif j == i:
                # Self-loop, invalid
                return None
    
    # Check degree
    for i in range(p):
        if len(adj[i]) != 3:
            return None
    
    return edges

def _check_girth_fast(edges, n, g):
    """Fast girth check using BFS with proper cycle detection."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for src in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[src] = 0
        queue = deque([src])
        
        while queue:
            u = queue.popleft()
            d = dist[u]
            if d >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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
