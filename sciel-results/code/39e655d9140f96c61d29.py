import time
import random
from collections import deque
import math

def _luw_construct(k, g):
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

def _get_gf_tables(q):
    """
    Returns (add, mul) functions for GF(q) where q is a prime power.
    For q=16, uses GF(2^4).
    For q=9, uses GF(3^2).
    For prime q, uses standard mod arithmetic.
    """
    if q == 16:
        # GF(16) = GF(2^4)
        # Irreducible poly: x^4 + x + 1
        # Elements are 0..15
        # Add is XOR
        add = lambda x, y: x ^ y
        # Build multiplication table
        mul_table = [[0]*16 for _ in range(16)]
        # Basis: 1, x, x^2, x^3
        # x^4 = x + 1
        # We can compute powers of a primitive element
        # Let alpha = 2 (binary 0010, represents x)
        # Actually, let's just do polynomial multiplication mod (x^4+x+1)
        def poly_mul(a, b):
            res = 0
            for i in range(4):
                if a & (1 << i):
                    for j in range(4):
                        if b & (1 << j):
                            res ^= (1 << (i+j))
            # Reduce mod x^4 + x + 1
            # If bit 4 is set, x^4 = x + 1
            # x^5 = x^2 + x
            # x^6 = x^3 + x^2
            # x^7 = x^4 + x^3 = x + 1 + x^3
            # General reduction:
            for k in range(7, 3, -1):
                if res & (1 << k):
                    # x^k = x^(k-4) * x^4 = x^(k-4) * (x+1) = x^(k-3) + x^(k-4)
                    res ^= (1 << k)
                    res ^= (1 << (k-1))
                    res ^= (1 << (k-2))
            return res
        for i in range(16):
            for j in range(16):
                mul_table[i][j] = poly_mul(i, j)
        mul = lambda x, y: mul_table[x][y]
        return add, mul

    elif q == 9:
        # GF(9) = GF(3^2)
        # Irreducible poly: x^2 + 1 (since -1 is not a square in GF(3)? 1^2=1, 2^2=1. Yes x^2+1 is irreducible)
        # Elements are pairs (a, b) in GF(3)^2
        # Represent as integer 3*a + b
        add_table = [[0]*9 for _ in range(9)]
        mul_table = [[0]*9 for _ in range(9)]
        for i in range(9):
            ai, bi = i // 3, i % 3
            for j in range(9):
                aj, bj = j // 3, j % 3
                # Add
                add_table[i][j] = 3 * ((ai + aj) % 3) + ((bi + bj) % 3)
                # Mul: (ai + bi*x)(aj + bj*x) = ai*aj + (ai*bj + bi*aj)x + bi*bj*x^2
                # x^2 = -1 = 2
                real = (ai * aj + bi * bj * 2) % 3
                imag = (ai * bj + bi * aj) % 3
                mul_table[i][j] = 3 * real + imag
        add = lambda x, y: add_table[x][y]
        mul = lambda x, y: mul_table[x][y]
        return add, mul

    else:
        # Prime
        add = lambda x, y: (x + y) % q
        mul = lambda x, y: (x * y) % q
        return add, mul

def _sl2_construct_prime_power(q, g, deadline):
    """
    Construct Cayley graph on SL(2,q) for prime power q.
    """
    if q < 4:
        return None
    
    n_expected = q * (q*q - 1)
    if n_expected > 50000: # Limit size for speed
        return None
        
    add, mul = _get_gf_tables(q)
    
    # Identity
    identity = (1, 0, 0, 1)
    neg_identity = ((q-1) if q != 2 else 1, 0, 0, (q-1) if q != 2 else 1)
    # For q=2, -1=1, so neg_identity = identity. But SL(2,2) is small.
    # We assume q >= 9.
    
    # Enumerate elements
    elements = []
    elem_index = {}
    
    # a, b, c, d
    # ad - bc = 1
    # d = (1 + bc) / a
    # a must be non-zero (invertible)
    
    # Precompute inverses for non-zero elements
    inv = [0] * q
    for i in range(1, q):
        # Find inverse
        for j in range(1, q):
            if mul(i, j) == 1:
                inv[i] = j
                break
    
    for a in range(1, q):
        for b in range(q):
            for c in range(q):
                d = mul(add(1, mul(b, c)), inv[a])
                elem = (a, b, c, d)
                if elem not in elem_index:
                    elem_index[elem] = len(elements)
                    elements.append(elem)
    
    if len(elements) != n_expected:
        return None
        
    n = len(elements)
    
    # Mat mul
    def mat_mul(e1, e2):
        a1, b1, c1, d1 = e1
        a2, b2, c2, d2 = e2
        # (a1 a2 + b1 c2, a1 b2 + b1 d2, c1 a2 + d1 c2, c1 b2 + d1 d2)
        return (
            add(mul(a1, a2), mul(b1, c2)),
            add(mul(a1, b2), mul(b1, d2)),
            add(mul(c1, a2), mul(d1, c2)),
            add(mul(c1, b2), mul(d1, d2))
        )
        
    # Find involutions
    involutions = []
    for elem in elements:
        a, b, c, d = elem
        # Trace = a + d
        if add(a, d) == 0:
            if elem != identity and elem != neg_identity:
                sq = mat_mul(elem, elem)
                if sq == identity:
                    involutions.append(elem)
                    
    if len(involutions) < 3:
        return None
        
    # Check girth function
    def check_girth_and_connectivity(gens):
        # BFS from identity
        # State: (elem_idx, prev_gen_idx)
        # prev_gen_idx: 0, 1, 2 or -1 for start
        start_state = (elem_index[identity], -1)
        visited = {start_state: 0}
        queue = deque([(start_state, 0)])
        
        max_depth = g // 2
        found_cycle = False
        
        while queue:
            state, depth = queue.popleft()
            if depth >= max_depth:
                continue
            elem_idx, prev_gen = state
            elem = elements[elem_idx]
            for gi, gen in enumerate(gens):
                if gi == prev_gen:
                    continue
                new_elem = mat_mul(elem, gen)
                new_idx = elem_index[new_elem]
                new_state = (new_idx, gi)
                if new_state in visited:
                    # Cycle found
                    # Length is depth + 1 + (dist to previous visit)
                    # Actually, if we revisit, it's a cycle.
                    # We just need to know if length < g
                    # The depth of the first visit is visited[new_state]
                    # Cycle length = depth + 1 - visited[new_state] ? No.
                    # Path 1: Id -> ... -> State (depth d1)
                    # Path 2: Id -> ... -> State (depth d2)
                    # Cycle length = d2 - d1.
                    # We want d2 - d1 >= g.
                    # If d2 - d1 < g, return False.
                    if depth + 1 - visited[new_state] < g:
                        return False
                else:
                    visited[new_state] = depth + 1
                    queue.append((new_state, depth + 1))
                    
        # Check connectivity
        # The BFS visited all reachable states.
        # Number of states visited should be n * 3 (approx)
        # Actually, each vertex has 3 incoming edges from 3 generators.
        # So total states is 3*n.
        # If visited count < 3*n, graph is not connected (or generators don't generate group).
        if len(visited) < n * 3:
             return False
             
        return True

    random.seed(42)
    max_trials = 500
    for trial in range(max_trials):
        if time.time() > deadline - 5:
            break
        if len(involutions) < 3:
            break
        idxs = random.sample(range(len(involutions)), 3)
        gens = [involutions[i] for i in idxs]
        if check_girth_and_connectivity(gens):
            # Build edges
            edges = []
            for i, elem in enumerate(elements):
                for gen in gens:
                    new_elem = mat_mul(elem, gen)
                    j = elem_index[new_elem]
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 3 or g != 16:
        return _luw_construct(k, g)
    
    # Try prime powers in increasing order of n
    # q=9 (n=720), q=16 (n=4080), q=32 (n=32736)
    candidate_qs = [9, 16, 32]
    
    for q in candidate_qs:
        if time.time() > deadline - 10:
            break
        result = _sl2_construct_prime_power(q, g, deadline)
        if result is not None:
            return result
            
    # Fallback
    return _luw_construct(3, 16)
