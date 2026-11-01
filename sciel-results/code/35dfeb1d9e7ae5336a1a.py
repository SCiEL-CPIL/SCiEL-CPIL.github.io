"""Combined program: algebraic voltage lifts + fast verification + LUW fallback."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _find_primitive_root(m):
    """Find a primitive root modulo m, or return None if it doesn't exist."""
    if m == 2:
        return 1
    if m == 4:
        return 3
    if m % 2 == 0 and m != 2 and m != 4:
        # Primitive roots only exist for p^k and 2*p^k where p is an odd prime
        # We can still try to find one if it exists, but it's rare for composite m
        pass
        
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    # Check if m has a primitive root
    # m must be 2, 4, p^k, or 2*p^k for an odd prime p
    def has_primitive_root(m):
        if m in [2, 4]:
            return True
        if m % 2 == 0:
            m_half = m // 2
            if m_half % 2 == 0:
                return False
        # Check if m or m/2 is a prime power
        def is_prime_power(n):
            if n < 2:
                return False
            for p in range(2, int(n**0.5) + 1):
                if n % p == 0:
                    temp = n
                    while temp % p == 0:
                        temp //= p
                    return temp == 1
            return True # It's prime
        
        if m % 2 != 0:
            return is_prime_power(m)
        else:
            return is_prime_power(m // 2)

    if not has_primitive_root(m):
        return None

    # Find a primitive root
    # Get the prime factors of phi(m)
    def phi(n):
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    phi_m = phi(m)
    
    # Get distinct prime factors of phi_m
    factors = set()
    temp = phi_m
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            factors.add(p)
            while temp % p == 0:
                temp //= p
        p += 1
    if temp > 1:
        factors.add(temp)

    for g in range(2, m):
        is_gen = True
        for p in factors:
            if pow(g, phi_m // p, m) == 1:
                is_gen = False
                break
        if is_gen:
            return g
    return None

def _primitive_root_shifts(m, count, base_offset=1):
    """Generate shifts using powers of a primitive root modulo m."""
    root = _find_primitive_root(m)
    if root is None:
        # Fallback to quadratic residues or linear
        return _quadratic_residues(m, count)
    
    powers = []
    current = 1
    for _ in range(m):
        powers.append(current)
        current = (current * root) % m
    
    # We want 'count' distinct shifts.
    # The powers of a primitive root generate all units modulo m.
    # If m is prime, there are m-1 units.
    # We pick the first 'count' powers, skipping 1 (which is 0 shift effectively if we consider multiplicative identity, but here shifts are additive in Z_m? No, voltage is additive).
    # Wait, the lift is over Z_m with additive voltages.
    # Using powers of a primitive root as *indices* or *values*?
    # The directive says: "assigning edge voltages as the first 15 powers of a primitive root modulo m".
    # So shift[i] = root^i mod m.
    
    result = []
    current = 1
    for i in range(count):
        # Skip 1? 1 is a valid shift.
        # But if shift is 0, it creates a loop in the base graph's lift if the base edge is a loop? No, base is simple.
        # If shift is 0, the lift edge connects (u, i) to (v, i).
        # Let's just use the powers directly.
        result.append(current)
        current = (current * root) % m
        
    return result

def _quadratic_residues(m, count):
    """Generate count distinct quadratic residues modulo m, spread out."""
    if m <= 1:
        return [0] * count
    residues = set()
    for a in range(1, m):
        residues.add((a * a) % m)
    for a in range(1, m):
        residues.add((m - (a * a) % m) % m)
    residues_list = sorted(residues)
    if len(residues_list) < count:
        residues_list = list(range(m))
    step = max(1, len(residues_list) // count)
    result = []
    for i in range(count):
        idx = (i * step) % len(residues_list)
        result.append(residues_list[idx])
    return result

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

    # Strategy:
    # 1. Try primitive root lifts for Petersen and K3,3 with m starting from Moore bound / n_base.
    # 2. If primitive root doesn't exist, fallback to other heuristics.
    # 3. Scan m upwards.

    for name, base_edges, n_base in base_graphs:
        if _time.time() > deadline - 10:
            break
        e_count = len(base_edges)
        n_min = max(1, (moore + n_base - 1) // n_base)

        # We will scan m from n_min to n_min + 500
        # For each m, try primitive root shifts first.
        for m in range(n_min, n_min + 500):
            if _time.time() > deadline - 5:
                break
            n = n_base * m
            if n >= best_n:
                continue

            # 1. Primitive Root Shifts
            shifts_pr = _primitive_root_shifts(m, e_count)
            result = _build_lift(base_edges, m, shifts_pr)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

            # 2. Quadratic Residues (Fallback/Alternative)
            shifts_qr = _quadratic_residues(m, e_count)
            result = _build_lift(base_edges, m, shifts_qr)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

            # 3. Random Shifts (Limited attempts to save time)
            # Only do random if we have time
            if _time.time() < deadline - 2:
                rng = _random.Random(hash((name, m, 0)))
                for _ in range(10):
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
