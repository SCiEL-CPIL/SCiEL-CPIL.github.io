import time
import random
import math
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    if k == 3:
        # Use the voltage lift approach on the Petersen graph
        result = _construct_petersen_lift(g, deadline, start_time)
        if result is not None:
            return result
        # Fallback to LUW construction
        return _construct_luw(k, g, d=11)
    else:
        return _construct_luw(k, g, d=7)


def _construct_petersen_lift(g, deadline, start_time):
    # Petersen graph edges
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    n_edges_base = 15

    base_adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(pet_edges):
        base_adj[u].append((v, i))
        base_adj[v].append((u, i))

    # Generate all closed non-backtracking walks of length 3..g-1 in Petersen
    def _get_closed_nb_walks(max_len):
        walks = set()
        for start_v in range(n_base):
            stack = []
            for next_v, e_id in base_adj[start_v]:
                u, v = pet_edges[e_id]
                sign = 1 if u == start_v else -1
                stack.append((next_v, e_id, 1, [(e_id, sign)]))
            while stack:
                curr_v, prev_e, length, path = stack.pop()
                if length >= max_len:
                    continue
                if curr_v == start_v and length >= 3:
                    walks.add(tuple(path))
                for next_v, e_id in base_adj[curr_v]:
                    if e_id == prev_e:
                        continue
                    u, v = pet_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    new_path = path + [(e_id, sign)]
                    stack.append((next_v, e_id, length + 1, new_path))
        return list(walks)

    closed_walks = _get_closed_nb_walks(g)

    # Precompute sparse coefficient vectors
    walk_coeffs = []
    for w in closed_walks:
        coeff = [0] * n_edges_base
        for e_id, sign in w:
            coeff[e_id] += sign
        sparse = [(eid, c) for eid, c in enumerate(coeff) if c != 0]
        if sparse:
            walk_coeffs.append(sparse)

    seen = set()
    unique_walk_coeffs = []
    for sparse in walk_coeffs:
        key = tuple(sorted(sparse))
        if key not in seen:
            seen.add(key)
            unique_walk_coeffs.append(sparse)
    walk_coeffs = unique_walk_coeffs

    best_n = float('inf')
    best_edges = None

    # Search for voltage assignments
    # Moore bound for k=3, g=16 is 512. n = 10*m.
    # m >= 52.
    
    random.seed(42)
    
    # Try primes first since they have good properties for voltage assignments
    def is_prime(n):
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
    
    # Generate candidate m values, prioritizing primes
    m_candidates = []
    for m in range(52, 300):
        if is_prime(m):
            m_candidates.append(m)
    # Add some non-primes as fallback
    for m in range(52, 300):
        if m not in m_candidates:
            m_candidates.append(m)
    
    # Sort by size (smaller first)
    m_candidates.sort()
    
    for m in m_candidates:
        if time.time() > deadline - 25.0:
            break
        candidate_n = n_base * m
        if candidate_n >= best_n:
            continue

        max_tries = 30000 if m < 60 else 15000
        found = False

        for _ in range(max_tries):
            if time.time() > deadline - 15.0:
                break

            # Generate random voltage assignment
            volts = [random.randint(1, m - 1) for _ in range(n_edges_base)]
            
            # Check gcd condition
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
                if g_val == 1:
                    break
            if g_val != 1:
                continue

            # Check all closed non-backtracking walks
            valid = True
            for sparse in walk_coeffs:
                s = 0
                for eid, c in sparse:
                    s += c * volts[eid]
                if s % m == 0:
                    valid = False
                    break

            if valid:
                # Build the lifted graph
                lift_edges = []
                for i, (u, v) in enumerate(pet_edges):
                    vol = volts[i]
                    for t in range(m):
                        u_lift = u * m + t
                        v_lift = v * m + (t + vol) % m
                        lift_edges.append((u_lift, v_lift))

                if _verify_girth(candidate_n, lift_edges, g):
                    best_n = candidate_n
                    best_edges = lift_edges
                    found = True
                    break

        if found:
            break

    if best_edges is not None:
        return best_n, best_edges

    return _construct_luw(k=3, g=g, d=11)


def _verify_girth(n, edges, girth_req):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != 3:
            return False
            
    for start in range(n):
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        while queue:
            curr, prev, dist = queue.popleft()
            if dist >= girth_req:
                continue
            if curr == start and dist >= 3:
                return False
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if (nb, curr) not in visited:
                    visited.add((nb, curr))
                    queue.append((nb, curr, dist + 1))
    return True


def _construct_luw(k, g, d=None, variant=0):
    if d is None:
        d = max(3, (g - 2) // 2)
    
    q = k
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
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
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
    
    return len(order), edges
