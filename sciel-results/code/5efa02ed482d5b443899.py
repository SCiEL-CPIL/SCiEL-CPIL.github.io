"""Rewritten: Algebraic LPS constructions + voltage lifts."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque
import numpy as np

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

def _lps_construct_3(q, g):
    """
    Construct LPS graph for p=2 (degree 3) over PGL(2, F_q).
    Returns (n, edges) if girth >= g, else None.
    """
    # Group size: q(q^2 - 1) / 2
    n = q * (q*q - 1) // 2
    
    # Generators in SL(2, F_q) lifted to PGL
    # A = [[1,1],[0,1]], B = [[1,0],[1,1]], C = [[0,1],[1,0]]
    # We work in PGL(2, q).
    # Represent elements as matrices [[a,b],[c,d]] with ad-bc = 1 (SL) or just PGL.
    # For PGL, we can normalize.
    
    # Enumerate PGL(2, q)
    # Elements are projective classes of invertible 2x2 matrices.
    # We can represent them by triples (a,b,c) where the matrix is [[a,b],[c,1]] (if d=1) or handle d=0.
    # Simpler: Enumerate all [[a,b],[c,d]] in F_q^4 with ad-bc != 0, modulo scalar.
    # Fix d=1 if possible? No.
    # Standard enumeration:
    # 1. Matrices with d=1: [[a,b],[c,1]]. ad-bc = a-cb != 0.
    # 2. Matrices with d=0: [[a,b],[c,0]]. ad-bc = -bc != 0 => b!=0, c!=0.
    
    elements = []
    index_map = {}
    
    # Case 1: d = 1
    for a in range(q):
        for b in range(q):
            for c in range(q):
                det = a - b*c
                if det != 0:
                    # Normalize? In PGL, [[a,b],[c,1]] is unique if we fix d=1.
                    # But [[2a,2b],[2c,2]] is the same element.
                    # So we must fix a representative.
                    # Let's fix d=1. If d=0, we can't.
                    # For d=1, the representative is unique?
                    # If [[a,b],[c,1]] = lambda [[a',b'],[c',1]], then 1=lambda*1 => lambda=1.
                    # So yes, unique.
                    mat = (a, b, c, 1)
                    elements.append(mat)
                    index_map[mat] = len(elements) - 1
                    
    # Case 2: d = 0
    # Matrix [[a,b],[c,0]]. det = -bc.
    # We need to fix a representative.
    # Let's fix b=1?
    # If b!=0, we can scale by b^-1 to make b=1.
    # Then matrix is [[a/b, 1], [c/b, 0]].
    # Let a' = a/b, c' = c/b.
    # det = -c'. So c' != 0.
    # Representative: (a', 1, c', 0).
    # Unique? If [[a',1],[c',0]] = lambda [[a'',1],[c'',0]], then 1=lambda*1 => lambda=1.
    # So yes.
    for a_prime in range(q):
        for c_prime in range(1, q): # c_prime != 0
            mat = (a_prime, 1, c_prime, 0)
            elements.append(mat)
            index_map[mat] = len(elements) - 1
            
    # Verify count
    if len(elements) != n:
        return None

    # Define multiplication in PGL
    def mul_pgl(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        a = a1*a2 + b1*c2
        b = a1*b2 + b1*d2
        c = c1*a2 + d1*c2
        d = c1*b2 + d1*d2
        # Normalize to representative form
        # If d != 0:
        if d != 0:
            # Scale by d^-1 to make d=1
            inv_d = pow(d, q-2, q)
            a = a * inv_d % q
            b = b * inv_d % q
            c = c * inv_d % q
            d = 1
            return (a, b, c, 1)
        else:
            # d = 0
            # Need b != 0 for det != 0?
            # det = a*c2... wait.
            # Original det was non-zero.
            # In PGL, if d=0, we normalized to b=1.
            # Check if b == 0?
            # If d=0, det = -b*c. Since det != 0, b!=0 and c!=0.
            # So we can scale by b^-1 to make b=1.
            if b == 0:
                # This should not happen for valid elements
                return None
            inv_b = pow(b, q-2, q)
            a = a * inv_b % q
            c = c * inv_b % q
            b = 1
            d = 0
            return (a, 1, c, 0)

    # Generators
    # A = [[1,1],[0,1]] -> (1,1,0,1)
    # B = [[1,0],[1,1]] -> (1,0,1,1)
    # C = [[0,1],[1,0]] -> (0,1,1,0)
    
    gen_A = (1, 1, 0, 1)
    gen_B = (1, 0, 1, 1)
    gen_C = (0, 1, 1, 0)
    
    # Check if generators are in the group
    if gen_A not in index_map or gen_B not in index_map or gen_C not in index_map:
        return None
        
    idx_A = index_map[gen_A]
    idx_B = index_map[gen_B]
    idx_C = index_map[gen_C]
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edges = []
    
    for i, v in enumerate(elements):
        # Neighbors: v*A, v*B, v*C
        for gen_idx, gen in [(idx_A, gen_A), (idx_B, gen_B), (idx_C, gen_C)]:
            u = mul_pgl(v, gen)
            if u is None:
                continue
            j = index_map[u]
            if j > i:
                adj[i].add(j)
                adj[j].add(i)
                edges.append((i, j))
                
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    adj_list = [list(s) for s in adj]
    
    # Verify girth
    if _verify_girth_fast(adj_list, n, g):
        return n, edges
        
    return None

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

    # Try LPS constructions first (algebraic, guaranteed high girth)
    # q must be such that LPS graph has girth >= g
    # Try small primes
    for q in [11, 13, 17, 19, 23]:
        if _time.time() > deadline - 20:
            break
        result = _lps_construct_3(q, g)
        if result is not None:
            return result

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
