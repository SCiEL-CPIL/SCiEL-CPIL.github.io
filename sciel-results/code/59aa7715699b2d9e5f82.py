"""Voltage-graph lift construction for small 5-regular graph of girth 9."""
import random
import time
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    if k != 5 or g != 9:
        return _luw_construct(k, g)
    
    deadline = time.time() + 110
    
    result = _cayley_psl2_13_search(deadline)
    if result is not None:
        return result
        
    return _luw_construct(k, g)

def _cayley_psl2_13_search(deadline):
    p = 13
    elems = []
    elem_to_idx = {}
    
    for a in range(p):
        for b in range(p):
            for c in range(p):
                if a == 0:
                    if b == 0:
                        continue
                    if (b * c) % p != (p - 1):
                        continue
                    d = 0
                    m = ((a, b), (c, d))
                    key = _canon_psl(m, p)
                    if key not in elem_to_idx:
                        elem_to_idx[key] = len(elems)
                        elems.append(m)
                else:
                    inv_a = pow(a, p - 2, p)
                    for c2 in range(p):
                        d = ((1 + b * c2) * inv_a) % p
                        m = ((a, b), (c2, d))
                        key = _canon_psl(m, p)
                        if key not in elem_to_idx:
                            elem_to_idx[key] = len(elems)
                            elems.append(m)
    
    n = len(elems)
    
    # Precompute multiplication table
    mult_table = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            m = _mat_mul(elems[i], elems[j], p)
            key = _canon_psl(m, p)
            mult_table[i, j] = elem_to_idx.get(key, -1)
    
    # Find involution
    inv_idx = -1
    for i in range(1, n):
        if mult_table[i, i] == 0:
            inv_idx = i
            break
    
    if inv_idx == -1:
        return None
    
    # Precompute inverse indices
    inv_table = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if mult_table[i, j] == 0:
                inv_table[i] = j
                break
    
    candidates = [i for i in range(n) if i != 0 and i != inv_idx]
    
    # Compute orders of elements to filter
    orders = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        orders[i] = _element_order(i, mult_table)
    
    # Filter candidates: avoid order 3 (creates 3-cycles)
    good_candidates = [i for i in candidates if orders[i] != 3]
    
    rng = random.Random(42)
    
    max_attempts = 20000
    for attempt in range(max_attempts):
        if time.time() > deadline - 2:
            break
            
        x_idx = rng.choice(good_candidates)
        y_idx = rng.choice(good_candidates)
        
        x_inv = int(inv_table[x_idx])
        y_inv = int(inv_table[y_idx])
        
        S_list = [inv_idx, x_idx, x_inv, y_idx, y_inv]
        
        # Check all distinct
        if len(set(S_list)) < 5:
            continue
        if 0 in S_list:
            continue
        
        # Build inv_map: for each generator index, what's its inverse index in S_list
        inv_map = {}
        for i, s in enumerate(S_list):
            inv_s = int(inv_table[s])
            inv_map[i] = S_list.index(inv_s)
        
        # Check girth using Cayley-specific method
        girth = _check_girth_cayley(mult_table, S_list, inv_map, 9)
        if girth >= 9:
            edges = []
            for u in range(n):
                for v in mult_table[u, S_list]:
                    v = int(v)
                    if u < v:
                        edges.append((u, v))
            return n, edges
            
    return None

def _element_order(i, mult_table):
    current = i
    order = 1
    while current != 0:
        current = int(mult_table[current, i])
        order += 1
        if order > 30:
            return order
    return 1

def _check_girth_cayley(mult_table, S_list, inv_map, min_girth):
    """
    Check girth of Cayley graph by finding shortest non-backtracking relation.
    """
    k = len(S_list)
    
    dist = {}
    q = deque()
    
    for gi in range(k):
        g = S_list[gi]
        state = (g, gi)
        dist[state] = 1
        q.append(state)
    
    while q:
        elem, prev_gi = q.popleft()
        d = dist[(elem, prev_gi)]
        
        if d >= min_girth - 1:
            continue
        
        forbidden = inv_map[prev_gi]
        for gi in range(k):
            if gi == forbidden:
                continue
            
            new_elem = int(mult_table[elem, S_list[gi]])
            
            if new_elem == 0:
                return d + 1
            
            new_state = (new_elem, gi)
            if new_state not in dist:
                dist[new_state] = d + 1
                q.append(new_state)
    
    return min_girth

def _canon_psl(m, p):
    a, b = m[0]
    c, d = m[1]
    
    if a != 0:
        inv_a = pow(a, p-2, p)
        return ((1, b*inv_a % p), (c*inv_a % p, d*inv_a % p))
    elif b != 0:
        inv_b = pow(b, p-2, p)
        return ((0, 1), (c*inv_b % p, d*inv_b % p))
    elif c != 0:
        inv_c = pow(c, p-2, p)
        return ((0, 0), (1, d*inv_c % p))
    else:
        return ((1, 0), (0, 1))

def _mat_mul(m1, m2, p):
    a1, b1 = m1[0]
    c1, d1 = m1[1]
    a2, b2 = m2[0]
    c2, d2 = m2[1]
    return (( (a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p ),
            ( (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p ))

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    for c in range(5, d + 1):
        r = (c - 5) % 4
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
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
