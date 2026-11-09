"""Voltage-graph (Cayley-like) construction for 5-regular girth-g cage search."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(k, g):
    """Return (n, edges) for a k-regular graph with girth >= g on as few vertices as possible."""
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin vs 150s kill

    # The task is fixed to k=5, g=9 by the contract, but keep it generic-ish.
    # Strategy: use a voltage graph over a small abelian group Z_m.
    # Base graph: a 5-regular base (here we use a complete graph on 5 vertices? No, that's 4-regular).
    # Better: use a Cayley graph of an abelian group directly, i.e. a 5-regular graph
    # that is a quotient of a larger Cayley graph. 
    #
    # Actually, the cleanest approach for a 5-regular graph with large girth:
    # Use a voltage graph / covering construction. Start with a small 5-regular base graph B,
    # assign voltages in an abelian group G to edges, and the derived graph is the covering.
    # The girth of the covering is related to the shortest non-backtracking closed walk in B
    # whose voltage product is the identity in G.
    #
    # For k=5, a natural base is the 5-vertex complete graph K5 (4-regular) plus one extra edge? 
    # No, K5 is 4-regular. We need a 5-regular base.
    # A 5-regular graph on 6 vertices: K6 minus a perfect matching (i.e., the octahedral graph).
    # That's 5-regular on 6 vertices. Its girth is 4 (has 4-cycles).
    #
    # Alternative: use a 5-regular base on more vertices with larger girth, then take a cover.
    # But the base itself must have girth at least... actually the base can have small girth;
    # the covering can have large girth if voltages are chosen well.
    #
    # Simplest strong approach: directly construct a 5-regular graph as a Cayley graph of 
    # Z_m x Z_n or Z_m, and search for generator sets that maximize girth.
    #
    # For 5-regular: Cayley graph of Z_m with 5 generators (pairs +/- a_i, so 5 distinct absolute values).
    # The graph is 5-regular on m vertices. Girth is the length of the shortest non-backtracking
    # closed walk, which for a Cayley graph of Z_m equals the shortest word in the generators
    # (with no consecutive inverse) that sums to 0 mod m, with no shorter word summing to 0.
    #
    # We'll search over m and generator sets {a1,...,a5} in Z_m (each a_i != 0, all distinct up to sign,
    # and we use both a_i and -a_i as neighbors, so 5 generators give degree 5 only if no a_i = -a_j mod m
    # for i != j and a_i != 0). Actually degree = 5 means we pick 5 generators a_1..a_5 and connect
    # x to x+a_i and x-a_i for each i, giving degree 10. That's not right.
    #
    # Correction: For a k-regular Cayley graph of an abelian group with generators S where S is 
    # closed under inverses, degree = |S|. For odd degree 5, we need a group of even order or 
    # a non-inversely-closed set. Simpler: use a directed Cayley graph? No, graph must be undirected.
    #
    # For undirected 5-regular Cayley graph: need a set S of 5 elements where S = S^{-1}.
    # This requires at least one element of order 2 (self-inverse). So group must have even order.
    # Let G = Z_m with m even. Pick s_1 of order 2 (s_1 = m/2), and two pairs {a, -a}, {b, -b}.
    # That gives |S| = 1 + 2 + 2 = 5. 
    #
    # So: G = Z_m (m even), generators: {m//2, a, -a, b, -b} with a, b distinct, not 0, not m//2.
    # The graph is 5-regular on m vertices.
    #
    # Girth: shortest non-backtracking closed walk. We'll compute it via BFS on the non-backtracking
    # state space (vertex, previous_vertex) or more efficiently, check all words of length < g.
    #
    # Plan:
    # 1. Try various m (even, from some minimum up).
    # 2. For each m, search over pairs (a, b) to maximize girth.
    # 3. Use simulated annealing / random search with a deadline.
    # 4. Return the best (n=m, edges) found with girth >= g.

    if k != 5:
        # Fallback for other k: use the original Lazebnik-Ustimenko-Woldar construction
        return _luw_construct(k, g)

    # Moore bound for 5-regular, girth 9:
    # n >= 1 + k * (1 + (k-1) + ... + (k-1)^{floor((g-1)/2)})
    # For g=9: floor(8/2)=4, so n >= 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426.
    # The task says best lower bound is 428, so we aim for m >= 428 even.

    min_m = 428
    if min_m % 2 != 0:
        min_m += 1

    best_m = None
    best_generators = None
    best_girth = 0

    # Precompute: for a given m and generators {m//2, a, -a, b, -b}, 
    # compute girth by BFS on non-backtracking walks.
    # State: (current_vertex, previous_vertex). Start from each vertex with no previous.
    # Actually, standard approach: BFS from each vertex, tracking (vertex, prev) to avoid backtracking.
    # But that's O(m^2) states. For m ~ 428, that's ~183k states, manageable.

    def compute_girth(m, a, b):
        """Compute girth of 5-regular Cayley graph of Z_m with generators {m//2, a, -a, b, -b}."""
        h = m // 2
        gens = [h, a, m - a, b, m - b]
        # Build adjacency
        adj = [[] for _ in range(m)]
        for i in range(m):
            for gen in gens:
                j = (i + gen) % m
                adj[i].append(j)
        
        # BFS for shortest non-backtracking cycle
        # State: (vertex, prev_vertex), dist
        # Start: for each vertex v, for each neighbor u of v, state (u, v) at dist 1.
        # If we reach (v, u) where u is a neighbor of v and dist >= 2, and the path is non-backtracking,
        # that's a cycle of length dist+1... 
        #
        # Better: standard non-backtracking BFS.
        # dist[v][prev] = shortest length of a non-backtracking walk from some start to v having come from prev.
        # Initialize: dist[start][None] = 0 for all start.
        # Transition: from (v, prev) to (u, v) for u in adj[v], u != prev, dist[u][v] = dist[v][prev] + 1.
        # A cycle of length L is found when we can go from (v, prev) to (u, v) where u == start_vertex and 
        # the walk length + 1 == L and the walk is non-backtracking.
        #
        # Simpler: use the standard algorithm for girth via BFS from each vertex, 
        # but that gives the shortest cycle through that vertex, not necessarily non-backtracking.
        # However, the shortest cycle in a graph IS non-backtracking (a backtracking walk isn't a simple cycle).
        # So we can just compute the girth as the minimum over all vertices of the shortest cycle through it.
        #
        # Even simpler: BFS from each vertex to find the shortest cycle. But that's O(m * (m + e)) = O(m^2 * 5).
        # For m=428, that's ~915k * 5 = ~4.5M operations, fast.
        
        # Actually, let's just do BFS from each vertex to find the shortest cycle through it.
        # For vertex s, BFS in the graph, tracking parent. If we find an edge (u, v) where both u and v 
        # are visited and v is not the parent of u (and u is not the parent of v), we have a cycle.
        # The cycle length is dist[u] + dist[v] + 1.
        
        min_cycle = float('inf')
        for s in range(m):
            dist = [-1] * m
            parent = [-1] * m
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        # Don't break, keep searching for shorter cycles through s
            if min_cycle <= g:
                break
            if min_cycle == 3:
                break
        
        return min_cycle if min_cycle < float('inf') else m * 2  # fallback

    # Search over m and generator pairs (a, b)
    # For a given m, a and b must be in {1, ..., m-1}, a != b, a != m-a, b != m-b (to avoid duplicates),
    # and a != h, b != h. Also a != 0, b != 0.
    # To ensure 5 distinct generators: h, a, m-a, b, m-b must all be distinct.
    # So: a != b, a != m-b, b != m-a, a != h, b != h, a != 0, b != 0.
    
    # We'll do random search with a deadline.
    # Try m values from min_m upward, and for each m, sample random (a, b) pairs.
    
    rng = random.Random(42)
    
    # Determine how many m values we can try and how many samples per m
    # Each girth computation is O(m^2) ~ 200k ops for m=428, so ~0.001s in Python.
    # We have ~130s, so we can do ~100k girth computations.
    # Let's try m from min_m to min_m + 200, with ~500 samples each.
    
    max_m_offset = 200
    samples_per_m = 500
    
    for m in range(min_m, min_m + max_m_offset + 1):
        if time.time() > deadline - 10:
            break
        h = m // 2
        valid_a = [x for x in range(1, m) if x != h and x != m - x]
        if len(valid_a) < 2:
            continue
        for _ in range(samples_per_m):
            if time.time() > deadline - 5:
                break
            a, b = rng.sample(valid_a, 2)
            # Check generators are distinct
            gens = [h, a, m - a, b, m - b]
            if len(set(gens)) != 5:
                continue
            girth = compute_girth(m, a, b)
            if girth >= g:
                if best_m is None or m < best_m:
                    best_m = m
                    best_generators = (a, b)
                    best_girth = girth
                    # Found a valid one, try to find smaller m? 
                    # Since we iterate m upward, the first valid m is the smallest for this m.
                    # But we should continue to see if a smaller m works.
                    # Actually we're going from min_m upward, so the first valid is the best for this m.
                    # We can break early if we find a valid one and m is close to the lower bound.
                    if m <= min_m + 10:
                        break
            # Track best girth for this m to see if we're making progress
    # If no valid graph found, return the one with largest girth (even if < g)
    # But the task requires girth >= g. If we can't find one, we should return something.
    # Let's also try a deterministic search for small m.
    
    if best_m is None:
        # Fallback: try the original LUW construction
        return _luw_construct(5, g)
    
    # Build the graph
    m = best_m
    a, b = best_generators
    h = m // 2
    gens = [h, a, m - a, b, m - b]
    edges = []
    for i in range(m):
        for gen in gens:
            j = (i + gen) % m
            if i < j:
                edges.append((i, j))
    
    return m, edges


def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction from the seed program."""
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
                aa, bb = rule[c]
                y[c] = add(x[c], mul(y[aa], x[bb])) if side == 0 else sub(x[c], mul(x[aa], y[bb]))
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
