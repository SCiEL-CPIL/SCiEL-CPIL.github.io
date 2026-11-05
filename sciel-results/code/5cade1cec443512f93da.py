"""Construct a 3-regular graph of girth >= g using algebraic Cayley graph search."""
import math
import sys
from collections import deque

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _next_prime(n):
    if n <= 2:
        return 2
    candidate = n if n % 2 else n + 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate

def _compute_girth_circulant(n, a, b):
    """Compute exact girth of circulant graph C_n(a, b) using non-backtracking BFS.
    Vertices: 0..n-1. Edges: v connected to (v+a)%n, (v-a)%n, (v+b)%n, (v-b)%n.
    But since it's undirected and 3-regular, each vertex has neighbors:
      (v+a)%n, (v-a)%n, (v+b)%n  -- wait, that's 4 neighbors.
    Actually for 3-regular: generators {a, -a, b} where -a != a mod n (n odd).
    Neighbors of v: (v+a)%n, (v-a)%n, (v+b)%n. But (v-b)%n is also a neighbor.
    Hmm, for 3-regular we need exactly 3 distinct neighbors.
    If we use generators {a, b} with a != b, a != -b, b != -a, then neighbors are:
    (v+a), (v-a), (v+b), (v-b) -- that's 4. Not 3-regular.
    
    For 3-regular circulant: we need 3 generators where some are negatives.
    Standard: generators {a, b, c} where a = -c or similar.
    Simplest: generators {a, -a, b} where a != -a (n odd) and b != ±a.
    Then neighbors of v: (v+a), (v-a), (v+b). But we also need (v-b) to be connected?
    No -- in an undirected graph, if b is a generator, then -b is also a generator.
    So generators {a, -a, b, -b} gives 4 neighbors.
    
    For 3-regular: use generators {a, b, c} where one is the negative of another.
    E.g., {a, -a, b} but then -b must also be in the set for undirected, so {a, -a, b, -b} = 4.
    
    Alternative: {a, b, c} where c = -(a+b) or something. Or use {a, b, -a-b}.
    Then neighbors: v+a, v+b, v-a-b. And by symmetry, v-a, v-b, v+a+b are also neighbors
    of other vertices. This is 3-regular if a, b, a+b are all distinct and nonzero mod n.
    """
    # For 3-regular circulant: generators {a, b, c} with a+b+c = 0 mod n
    # Neighbors of v: (v+a)%n, (v+b)%n, (v+c)%n where c = (-a-b)%n
    # This is 3-regular if a, b, c are distinct and nonzero.
    
    c = (-a - b) % n
    if a == 0 or b == 0 or c == 0:
        return 0
    if len({a, b, c}) != 3:
        return 0
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for v in range(n):
        adj[v] = [(v + a) % n, (v + b) % n, (v + c) % n]
    
    # Non-backtracking BFS from each directed edge
    # State: (current_vertex, prev_vertex, depth)
    # Start: for each directed edge (u, v), state (v, u, 1)
    # If we reach (u, v, d) where d >= 3, we found a cycle of length d
    
    # To be efficient, we'll do BFS and track visited states
    # State is (current, prev) -- the last two vertices in the walk
    # We want the shortest cycle, so BFS by depth
    
    from collections import deque
    
    best_girth = float('inf')
    
    # For each starting directed edge
    for start_u in range(n):
        for start_v in adj[start_u]:
            # BFS from (start_v, start_u) at depth 1
            visited = set()
            queue = deque()
            queue.append((start_v, start_u, 1))
            visited.add((start_v, start_u))
            
            found = False
            while queue:
                curr, prev, depth = queue.popleft()
                
                # Check if we've returned to start_u
                if curr == start_u and depth >= 3:
                    best_girth = min(best_girth, depth)
                    found = True
                    break
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue  # no backtracking
                    state = (nxt, curr)
                    if state not in visited:
                        visited.add(state)
                        if depth + 1 >= best_girth:
                            continue
                        queue.append((nxt, curr, depth + 1))
            
            if found:
                # We can break early if we've found a small girth
                if best_girth <= 4:
                    break
        if best_girth <= 4:
            break
    
    return best_girth if best_girth != float('inf') else n  # fallback

def construct(k, g):
    """Construct a 3-regular graph with girth >= g using algebraic Cayley graph search."""
    assert k == 3, "This implementation only handles k=3"
    
    # Moore bound for 3-regular girth g (even): n >= 2*(2)^(g/2-1)
    moore_bound = 2 * (2 ** (g // 2 - 1)) if g % 2 == 0 else 2 * (2 ** ((g - 1) // 2))
    
    best_n = None
    best_edges = None
    
    # Search over odd n (not necessarily prime) and generator triples
    # For 3-regular Cayley on Z_n: need generators {a, b, c} such that
    # the set of neighbors of 0 is exactly 3 elements.
    # This means among {a, -a, b, -b, c, -c} mod n, there are exactly 3 distinct values.
    # Simplest: c = -(a+b), then neighbors are a, -a, b, -b, -(a+b), (a+b)
    # For this to be 3-regular, we need some coincidences.
    # Alternative: use generators where one is the negative of another:
    # gens = [a, b, -a] -> neighbors: a, -a, b, -b, -a, a -> {a, -a, b, -b} = 4 unless b=±a
    # That gives 4-regular.
    
    # Better approach: use non-cyclic groups. Let's use Z_p × Z_p for prime p.
    # Cayley graph on Z_p × Z_p with 3 generators.
    # n = p^2 vertices.
    
    # Search over primes p
    p = 3
    max_p = 50
    
    while p <= max_p and p * p >= moore_bound:
        if _is_prime(p):
            n = p * p
            # Search over generator triples in Z_p × Z_p
            # Generator (dx, dy) where not both zero
            # For 3-regular: need 3 generators such that the set
            # {g1, -g1, g2, -g2, g3, -g3} has exactly 3 distinct elements
            # This means one generator is the negative of another, OR
            # two generators are negatives of each other and the third is self-inverse (2g=0)
            
            # Strategy: pick g1, g2, set g3 = -(g1+g2) or g3 = g1-g2, etc.
            # Then check if we get exactly 3 neighbors.
            
            # Generate candidate generator sets
            candidates = set()
            
            # Type 1: g3 = -(g1 + g2)
            for x1 in range(p):
                for y1 in range(p):
                    if x1 == 0 and y1 == 0:
                        continue
                    for x2 in range(x1, p):
                        for y2 in range(0 if x2 > x1 else y1 + 1, p):
                            if x2 == 0 and y2 == 0:
                                continue
                            # Check g1 != g2 and g1 != -g2
                            if (x1, y1) == (x2, y2):
                                continue
                            if (x1, y1) == ((-x2) % p, (-y2) % p):
                                continue
                            g3 = ((-x1 - x2) % p, (-y1 - y2) % p)
                            # Check g3 != 0
                            if g3 == (0, 0):
                                continue
                            # Check g3 != g1, g3 != g2, g3 != -g1, g3 != -g2
                            if g3 == (x1, y1) or g3 == (x2, y2):
                                continue
                            if g3 == ((-x1) % p, (-y1) % p) or g3 == ((-x2) % p, (-y2) % p):
                                continue
                            # Check we have exactly 3 distinct neighbors
                            nbrs = set()
                            nbrs.add((x1, y1))
                            nbrs.add((-x1 % p, -y1 % p))
                            nbrs.add((x2, y2))
                            nbrs.add((-x2 % p, -y2 % p))
                            nbrs.add(g3)
                            nbrs.add((-g3[0] % p, -g3[1] % p))
                            if len(nbrs) == 3:
                                candidates.add(((x1, y1), (x2, y2), g3))
            
            # Type 2: g3 = g1 - g2 (or g2 - g1)
            for x1 in range(p):
                for y1 in range(p):
                    if x1 == 0 and y1 == 0:
                        continue
                    for x2 in range(x1, p):
                        for y2 in range(0 if x2 > x1 else y1 + 1, p):
                            if x2 == 0 and y2 == 0:
                                continue
                            if (x1, y1) == (x2, y2):
                                continue
                            if (x1, y1) == ((-x2) % p, (-y2) % p):
                                continue
                            g3 = ((x1 - x2) % p, (y1 - y2) % p)
                            if g3 == (0, 0):
                                continue
                            if g3 == (x1, y1) or g3 == (x2, y2):
                                continue
                            if g3 == ((-x1) % p, (-y1) % p) or g3 == ((-x2) % p, (-y2) % p):
                                continue
                            nbrs = set()
                            nbrs.add((x1, y1))
                            nbrs.add((-x1 % p, -y1 % p))
                            nbrs.add((x2, y2))
                            nbrs.add((-x2 % p, -y2 % p))
                            nbrs.add(g3)
                            nbrs.add((-g3[0] % p, -g3[1] % p))
                            if len(nbrs) == 3:
                                candidates.add(((x1, y1), (x2, y2), g3))
            
            # Try each candidate
            for gens in candidates:
                girth = _compute_girth_cayley(n, gens)
                if girth >= g:
                    # Build edges
                    edges = []
                    for v in range(n):
                        vx, vy = divmod(v, p)
                        for gx, gy in gens:
                            w = ((vx + gx) % p) * p + ((vy + gy) % p)
                            if v < w:
                                edges.append((v, w))
                    
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    
                    break  # Found valid for this p, move to next p
        
        p += 1
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use the original Lazebnik-Ustimenko-Woldar construction
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
