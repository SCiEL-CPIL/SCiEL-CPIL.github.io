"""Rewritten: algebraic voltage lifts with improved search and verification."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _build_lift(base_edges, m, shifts):
    """Build a voltage graph lift over Z_m.
    base_edges: list of (u,v) on the base graph
    m: lift size
    shifts: list of voltage shifts, one per base edge
    Returns (n, adj_list, edge_list) or None if invalid."""
    n_base = max(max(u for u,v in base_edges), max(v for u,v in base_edges)) + 1
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    for idx, (u, v) in enumerate(base_edges):
        vol = shifts[idx] % m
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    adj_list = [list(s) for s in adj]
    return n, adj_list, edges

def _luw_construct(k, g, d_override=None):
    """LUW construction with optional d override."""
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _shifts_linear(m, count, offset=1):
    """Linear shifts: offset, 2*offset, 3*offset, ... mod m."""
    if m <= 1:
        return [0] * count
    result = []
    for i in range(count):
        result.append(((i + 1) * offset) % m)
    return result

def _shifts_powers(m, count, base=2):
    """Powers of base modulo m."""
    if m <= 1:
        return [0] * count
    result = []
    current = 1
    for _ in range(count):
        current = (current * base) % m
        result.append(current)
    return result

def _shifts_fibonacci(m, count):
    """Fibonacci-like shifts modulo m."""
    if m <= 1:
        return [0] * count
    a, b = 1, 1
    result = []
    for _ in range(count):
        result.append(a)
        a, b = b, (a + b) % m
    return result

def _shifts_cubic(m, count):
    """Cubic polynomial shifts: i^3 mod m."""
    if m <= 1:
        return [0] * count
    result = []
    for i in range(1, count + 1):
        result.append((i ** 3) % m)
    return result

def _shifts_prime(m, count):
    """First count primes modulo m."""
    if m <= 1:
        return [0] * count
    result = []
    candidate = 2
    while len(result) < count:
        is_prime = True
        for p in range(2, int(candidate ** 0.5) + 1):
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            result.append(candidate % m)
        candidate += 1
    return result

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    # k=3, target girth g (typically 16)
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2

    # Petersen graph
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # K_3,3
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]

    base_graphs = [
        ("petersen", petersen_edges, 10),
        ("k33", k33_edges, 6),
    ]

    best_result = None
    best_n = float('inf')

    # Shift generation strategies
    def get_shifts(m, count, strategy_id):
        if strategy_id == 0:
            return _shifts_linear(m, count, offset=1)
        elif strategy_id == 1:
            return _shifts_linear(m, count, offset=2)
        elif strategy_id == 2:
            return _shifts_powers(m, count, base=2)
        elif strategy_id == 3:
            return _shifts_powers(m, count, base=3)
        elif strategy_id == 4:
            return _shifts_fibonacci(m, count)
        elif strategy_id == 5:
            return _shifts_cubic(m, count)
        elif strategy_id == 6:
            return _shifts_prime(m, count)
        else:
            return [0] * count

    # Strategy:
    # 1. Try multiple shift strategies for Petersen and K3,3 with m starting from Moore bound / n_base.
    # 2. Scan m upwards with increasing random attempts.

    for name, base_edges, n_base in base_graphs:
        if _time.time() > deadline - 10:
            break
        e_count = len(base_edges)
        n_min = max(1, (moore + n_base - 1) // n_base)

        # We will scan m from n_min to n_min + 800
        # For each m, try multiple shift strategies.
        for m in range(n_min, n_min + 800):
            if _time.time() > deadline - 5:
                break
            n = n_base * m
            if n >= best_n:
                continue

            # Try deterministic strategies
            for strategy_id in range(7):
                if _time.time() > deadline - 3:
                    break
                shifts = get_shifts(m, e_count, strategy_id)
                result = _build_lift(base_edges, m, shifts)
                if result is not None:
                    n_r, adj, edges = result
                    if _verify_girth_fast(adj, n_r, g):
                        best_n = n_r
                        best_result = (n_r, edges)
                        return n_r, edges

            # Try random shifts with increasing attempts for smaller m
            if _time.time() < deadline - 2:
                # More random attempts for smaller m where search space is more constrained
                max_random = 20 if m < n_min + 100 else 5
                rng = _random.Random(hash((name, m, 0)))
                for _ in range(max_random):
                    # Avoid 0 shifts to prevent trivial structures
                    shifts_rand = [rng.randint(1, m - 1) for _ in range(e_count)]
                    result = _build_lift(base_edges, m, shifts_rand)
                    if result is not None:
                        n_r, adj, edges = result
                        if _verify_girth_fast(adj, n_r, g):
                            best_n = n_r
                            best_result = (n_r, edges)
                            return n_r, edges

    # Fallback: LUW
    if _time.time() < deadline - 5:
        for d_try in range(6, 14):
            if _time.time() > deadline - 3:
                break
            try:
                n_try, edges_try = _luw_construct(3, g, d_override=d_try)
                if n_try < best_n:
                    adj = [set() for _ in range(n_try)]
                    for u, v in edges_try:
                        adj[u].add(v)
                        adj[v].add(u)
                    adj_list = [list(s) for s in adj]
                    if _verify_girth_fast(adj_list, n_try, g):
                        best_n = n_try
                        best_result = (n_try, edges_try)
                        return n_try, edges_try
            except Exception:
                pass

    if best_result is not None:
        return best_result

    n_luw, edges_luw = _luw_construct(3, g)
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
