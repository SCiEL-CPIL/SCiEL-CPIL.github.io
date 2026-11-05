"""Cage problem n(k,g): build a k-regular graph with girth >= g on few vertices.

Strategy:
  1. Known small cages (Petersen, Heawood, etc.)
  2. Generalized Petersen graphs (3-regular)
  3. Random k-regular graphs (configuration model) with girth check
  4. Cayley graph construction on symmetric group S_m with involutory generators
  5. LUW algebraic construction as fallback
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    deadline = time.time() + 110.0
    best_n = None
    best_edges = None

    def check_girth(n, edges, g):
        """Verify girth >= g using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for u in adj:
            u.sort()
        max_depth = g - 1
        for start in range(n):
            visited = {}
            queue = deque()
            for nb in adj[start]:
                visited[(nb, start)] = 1
                queue.append((nb, start))
            while queue:
                v, prev = queue.popleft()
                d = visited[(v, prev)]
                if d >= max_depth:
                    continue
                for nb in adj[v]:
                    if nb == prev:
                        continue
                    if nb == start:
                        return False
                    if (nb, v) not in visited:
                        visited[(nb, v)] = d + 1
                        queue.append((nb, v))
        return True

    def is_valid(n, edges, k, g):
        if n < 2:
            return False
        deg = [0] * n
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != k:
                return False
        if not check_girth(n, edges, g):
            return False
        return True

    def try_candidate(n, edges):
        nonlocal best_n, best_edges
        if best_n is not None and n >= best_n:
            return
        if is_valid(n, edges, k, g):
            best_n = n
            best_edges = edges

    # --- Family 1: Known small cages ---
    if k == 3 and g <= 5:
        n = 10
        edges = []
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        inner = [5, 7, 9, 6, 8]
        for i in range(5):
            edges.append((inner[i], inner[(i + 2) % 5]))
        for i in range(5):
            edges.append((i, 5 + i))
        try_candidate(n, edges)

    if k == 3 and g <= 6:
        n = 14
        edges = []
        fano_lines = [
            [0, 1, 2], [0, 3, 4], [0, 5, 6],
            [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]
        ]
        for line_idx, line in enumerate(fano_lines):
            for pt in line:
                edges.append((pt, 7 + line_idx))
        try_candidate(n, edges)

    # --- Family 2: Generalized Petersen graphs (3-regular) ---
    if k == 3:
        for n in range(5, 200):
            if time.time() > deadline:
                break
            if best_n is not None and 2 * n >= best_n:
                break
            edges = []
            for i in range(n):
                edges.append((i, (i + 1) % n))
            for i in range(n):
                edges.append((n + i, n + (i + k) % n))
            for i in range(n):
                edges.append((i, n + i))
            try_candidate(2 * n, edges)

    # --- Family 3: Random k-regular graphs ---
    def moore_bound(k, g):
        if k == 2:
            return g if g % 2 == 0 else g + 1
        if g % 2 == 0:
            r = g // 2
            if k == 2:
                return 2 * r
            s = 0
            for i in range(r - 1):
                s += (k - 1) ** i
            return 2 * (k - 1) * s // (k - 2) + 2
        else:
            r = (g - 1) // 2
            s = 0
            for i in range(r):
                s += (k - 1) ** i
            return k + (k - 1) * s

    mb = moore_bound(k, g)
    max_attempts = 300
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        n_try = mb + attempt * max(1, mb // 50)
        if n_try > 5000:
            n_try = 5000
        if best_n is not None and n_try >= best_n:
            continue

        if k * n_try % 2 != 0:
            n_try += 1
            if best_n is not None and n_try >= best_n:
                continue

        graph = None
        for inner_attempt in range(15):
            if time.time() > deadline:
                break
            stubs = []
            for v in range(n_try):
                stubs.extend([v] * k)
            random.shuffle(stubs)
            edges = []
            valid = True
            seen = set()
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append((u, v))
            if valid:
                graph = (n_try, edges)
                break

        if graph is None:
            continue
        try_candidate(graph[0], graph[1])

    # --- Family 4: Cayley graph on symmetric group S_m ---
    def build_cayley_cage(m, gen_set, k_target, g_target):
        """Build Cayley graph of S_m with given generators, check k-regular and girth."""
        # Generate all elements of S_m
        from itertools import permutations
        elements = list(permutations(range(m)))
        n = len(elements)
        
        # Map elements to indices
        elem_to_idx = {elem: i for i, elem in enumerate(elements)}
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        edges = []
        edge_set = set()
        
        for i, elem in enumerate(elements):
            for gen in gen_set:
                # Multiply elem * gen (right multiplication)
                prod = tuple(elem[gen[j]] for j in range(m))
                j = elem_to_idx.get(prod)
                if j is None:
                    return None
                if i < j:
                    key = (i, j)
                    if key not in edge_set:
                        edge_set.add(key)
                        edges.append(key)
                adj[i].append(j)
        
        # Check degree
        for a in adj:
            if len(a) != k_target:
                return None
        
        return n, edges

    def find_cayley_cage(k, g, time_limit):
        """Search for a Cayley graph on S_m with the right properties."""
        end_time = time.time() + time_limit
        best = None
        
        # Try different m values
        for m in range(3, 8):
            if time.time() > end_time:
                break
            
            # Generate random sets of k involutions in S_m
            from itertools import permutations
            elements = list(permutations(range(m)))
            elem_to_idx = {elem: i for i, elem in enumerate(elements)}
            
            # Find all involutions (elements of order 2)
            involutions = []
            for elem in elements:
                # Check if elem is an involution: elem^2 = identity
                comp = tuple(elem[elem[j]] for j in range(m))
                if comp == tuple(range(m)):
                    involutions.append(elem)
            
            if len(involutions) < k:
                continue
            
            # Try random subsets of k involutions
            num_tries = 50
            for _ in range(num_tries):
                if time.time() > end_time:
                    break
                gen_set = random.sample(involutions, k)
                
                # Build Cayley graph
                result = build_cayley_cage(m, gen_set, k, g)
                if result is None:
                    continue
                
                n, edges = result
                if best is None or n < best[0]:
                    if is_valid(n, edges, k, g):
                        best = (n, edges)
        
        return best

    if best_n is None:
        cayley_result = find_cayley_cage(k, g, 20.0)
        if cayley_result is not None:
            try_candidate(cayley_result[0], cayley_result[1])

    # --- Family 5: LUW construction as fallback ---
    def field_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def build_luw(q, d):
        add, sub, mul = field_ops(q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
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

    if best_n is None:
        q = k
        d = g - 5
        if d < 3:
            d = 3
        if d % 2 == 0:
            d += 1
        n_luw, edges_luw = build_luw(q, d)
        try_candidate(n_luw, edges_luw)

    if best_n is None:
        if k == 2:
            n = g if g % 2 == 0 else g + 1
            edges = [(i, (i + 1) % n) for i in range(n)]
            return n, edges
        return 0, []

    return best_n, best_edges

# EVOLVE-BLOCK-END
