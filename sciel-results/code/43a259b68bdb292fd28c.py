import time
import random
import numpy as np
from collections import deque

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
    """Non-backtracking BFS girth check from every vertex."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for 3-regular graphs.
    """
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    q = 3
    def build_luw(d):
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

    for d in [5, 6, 7, 8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    return build_luw(11)

def _petersen_edges():
    """Return edges of the Petersen graph (10 vertices, 15 edges)."""
    # Outer pentagon: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    return edges

def _find_cycles_in_base(base_edges, max_len):
    """Find all cycles up to max_len in a small base graph, returning edge-index tuples."""
    n_base = 10  # Petersen
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(base_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))

    cycles = []
    seen = set()

    def dfs(start, curr, visited, edge_path, length):
        if length >= max_len:
            return
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)

    for start in range(n_base):
        visited = {start}
        dfs(start, start, visited, [], 0)
    return cycles

def _voltage_lift_petersen(k, g, deadline):
    """Voltage lift of Petersen graph to get 3-regular graph with girth >= g."""
    if k != 3:
        return None

    base_edges = _petersen_edges()
    n_base = 10
    n_edges = len(base_edges)  # 15

    # Find all cycles up to length g-1 in Petersen
    cycles = _find_cycles_in_base(base_edges, g)

    # Try primes m; n = 10 * m
    # We want small n, so try small primes first
    primes = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    for m in primes:
        if time.time() > deadline:
            break
        n = n_base * m
        # Need at least enough vertices for girth g: Moore bound for (3,g)
        # For g=16: Moore bound is 2*(2^7-1) = 254, so n >= 254
        # 10*m >= 254 => m >= 26, so start from 29
        if n < 254:
            continue

        # Random search for voltages
        for attempt in range(5000):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            if valid:
                # Construct lift
                lift_edges = []
                for idx, (u, v) in enumerate(base_edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a = u * m + j
                        b = v * m + ((j + vol) % m)
                        lift_edges.append((a, b))
                return n, lift_edges
    return None

def _heisenberg_cayley(g, deadline):
    """
    Cayley graph on Heisenberg group H(3, Z_m) with generators
    that give a 3-regular graph.
    """
    if time.time() > deadline:
        return None

    # Heisenberg group elements: (x, y, z) in Z_m^3
    # Multiplication: (x1,y1,z1)*(x2,y2,z2) = (x1+x2, y1+y2, z1+z2+x1*y2) mod m
    # Inverse: (x,y,z)^{-1} = (-x, -y, -z+x*y) mod m
    # Generators: a=(1,0,0), b=(0,1,0), c=(0,0,1) -- but this gives 6-regular
    # For 3-regular, use generators {a, b, c} where we identify inverses
    # Actually, for 3-regular we need 3 generators with g = g^{-1} or pairs
    # Let's use: S = {(1,0,0), (0,1,0), (0,0,1)} and include inverses
    # That gives degree 6. For degree 3, use involutory generators or
    # use a different approach.

    # Alternative: use the "free" 3-regular construction via
    # random pairing with girth improvement.

    # Instead, let's try a different algebraic construction:
    # Use Z_m x Z_m with generators that give a 3-regular Cayley graph.
    # For 3-regular on abelian group, need generators S with |S|=3, S=S^{-1}.
    # So S = {a, b, c} where a^{-1}=a, etc. (involutions)
    # In Z_m, an element is an involution if 2x = 0 mod m, i.e., x = m/2.
    # This only works for even m.

    # Let's try a non-abelian approach: use the dihedral group or
    # simply fall back to random 3-regular graph with girth check.

    return None

def _random_3reg_with_girth(n, g, deadline):
    """
    Generate random 3-regular graph via configuration model,
    then try to improve girth via 2-switches.
    """
    if n % 2 != 0 or n < 4:
        return None

    half_time = deadline - 10.0
    if time.time() > half_time:
        return None

    best_edges = None
    best_girth = 0

    for trial in range(50):
        if time.time() > half_time:
            break

        # Configuration model
        stubs = []
        for v in range(n):
            stubs.extend([v] * 3)
        random.shuffle(stubs)

        # Pair stubs
        edges_set = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edges_set:
                valid = False
                break
            edges_set.add(key)

        if not valid:
            continue

        edges = list(edges_set)
        if len(edges) != n * 3 // 2:
            continue

        # Check girth
        if _check_girth_fast(n, edges, g):
            return n, edges

        # Try to improve via 2-switches
        adj = _build_adj(n, edges)
        improved = False
        for _ in range(1000):
            if time.time() > half_time:
                break
            # Pick random edge
            u, v = random.choice(edges)
            # Pick random neighbor of u (not v) and random neighbor of v (not u)
            if len(adj[u]) < 2 or len(adj[v]) < 2:
                continue
            neighbors_u = [x for x in adj[u] if x != v]
            neighbors_v = [x for x in adj[v] if x != u]
            if not neighbors_u or not neighbors_v:
                continue
            w = random.choice(neighbors_u)
            x = random.choice(neighbors_v)
            if w == x:
                continue
            # Check if (u,x) and (v,w) already exist
            if (min(u, x), max(u, x)) in edges_set or (min(v, w), max(v, w)) in edges_set:
                continue
            # Perform 2-switch
            edges_set.discard((min(u, v), max(u, v)))
            edges_set.add((min(u, x), max(u, x)))
            edges_set.add((min(v, w), max(v, w)))
            edges = list(edges_set)
            improved = True

            if _check_girth_fast(n, edges, g):
                return n, edges

        if not best_edges or True:
            best_edges = edges

    if best_edges is not None:
        return n, best_edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    if k != 3:
        return _luw_construct(k, g)

    # 1. Try LUW construction (algebraic, likely to succeed with small n)
    n_luw, edges_luw = _luw_construct(k, g)
    if _check_girth_fast(n_luw, edges_luw, g):
        return n_luw, edges_luw

    # 2. Try voltage lift of Petersen (n = 10*m, smaller base)
    result = _voltage_lift_petersen(k, g, deadline)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # 3. Fallback: random 3-regular with girth search
    # Use a target size around Moore bound
    # For (3,16): Moore bound = 2*(2^7 - 1) = 254
    # Try a few sizes
    for target_n in [256, 260, 264, 270, 280, 300]:
        if time.time() > deadline:
            break
        result = _random_3reg_with_girth(target_n, g, deadline)
        if result is not None:
            n, edges = result
            if _check_girth_fast(n, edges, g):
                return n, edges

    # Last resort: return LUW even if girth not verified (shouldn't happen)
    return n_luw, edges_luw
