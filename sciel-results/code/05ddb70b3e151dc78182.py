import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # Leave 10s for safety
    
    if k != 5:
        return _construct_luw(k, g)
        
    # Try to find a Cayley graph in PSL(2, q)
    # For k=5, g=9, we know n(5,9) is around 1116.
    # PSL(2, 13) has 1092 vertices.
    # PSL(2, 17) has 2448 vertices.
    # We prefer smaller n.
    
    for q in [13, 17, 19, 23]:
        if time.time() > deadline:
            break
            
        n = q * (q * q - 1) // 2
        if n < 1066:
            continue
            
        elements, idx_map, inverses, orders, mul_table, ident_idx = _generate_psl2_fast(q)
        
        # Group elements by order
        # We want generators that are not involutions (order 2) to avoid short cycles
        # We need 5 generators.
        # Typically, we pick 2 pairs of inverses (4 elements) and 1 involution.
        # Or 5 elements such that the set is closed under inverse.
        # The seed uses {s1, s1^-1, s2, s2^-1, s3} where s3 is an involution.
        
        involutions = [i for i in range(n) if orders[i] == 2]
        non_involutions = [i for i in range(n) if orders[i] > 2]
        
        if not involutions or not non_involutions:
            continue
            
        # Sort non-involutions by order (descending) to prioritize high order elements
        non_involutions.sort(key=lambda x: orders[x], reverse=True)
        
        found = False
        # Limit the search to avoid timeout
        # We will iterate over a subset of candidates
        # Let's try the top 100 non-involutions and all involutions
        
        s1_candidates = non_involutions[:100]
        s2_candidates = non_involutions[:100]
        
        # To avoid duplicate checks, we can enforce s1 < s2 or similar
        # But orders might be different.
        
        for s1_idx in s1_candidates:
            if time.time() > deadline:
                break
            s1_inv_idx = inverses[s1_idx]
            
            for s2_idx in s2_candidates:
                if s2_idx == s1_idx or s2_idx == s1_inv_idx:
                    continue
                s2_inv_idx = inverses[s2_idx]
                
                # Ensure we don't check the same set multiple times
                # The set is {s1, s1_inv, s2, s2_inv, s3}
                # We can fix s1 < s2 in some order?
                # For now, just check.
                
                for s3_idx in involutions:
                    if time.time() > deadline:
                        break
                        
                    candidates = [s1_idx, s1_inv_idx, s2_idx, s2_inv_idx, s3_idx]
                    
                    if len(set(candidates)) != 5:
                        continue
                    
                    # Check girth
                    if _check_girth_cayley_optimized(n, candidates, mul_table, ident_idx, g):
                        edges = []
                        for i in range(n):
                            for s in candidates:
                                j = mul_table[i][s]
                                if i < j:
                                    edges.append((i, j))
                        
                        if len(edges) == n * 5 // 2:
                            return n, edges
                            
    return _construct_luw(k, g)

def _generate_psl2_fast(q):
    elements = []
    idx_map = {}
    
    inv = [0] * q
    for i in range(1, q):
        inv[i] = pow(i, q - 2, q)
        
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0:
                    if b == 0:
                        continue
                    if (b * c) % q != (q - 1) % q:
                        continue
                    d = 0
                else:
                    d = (1 + b * c) % q
                    d = (d * inv[a]) % q
                
                if (a * d - b * c) % q != 1:
                    continue
                
                m = (a, b, c, d)
                first_non_zero = None
                for val in m:
                    if val != 0:
                        first_non_zero = val
                        break
                        
                if first_non_zero is not None and first_non_zero > (q - 1) // 2:
                    m = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
                    
                if m not in idx_map:
                    idx_map[m] = len(elements)
                    elements.append(m)
                    
    n = len(elements)
    
    mul_table = [[0] * n for _ in range(n)]
    inverses = [0] * n
    orders = [0] * n
    
    ident = (1, 0, 0, 1)
    ident_idx = idx_map[ident]
    
    for i in range(n):
        a1, b1, c1, d1 = elements[i]
        for j in range(n):
            a2, b2, c2, d2 = elements[j]
            a = (a1 * a2 + b1 * c2) % q
            b = (a1 * b2 + b1 * d2) % q
            c = (c1 * a2 + d1 * c2) % q
            d = (c1 * b2 + d1 * d2) % q
            prod = (a, b, c, d)
            first_non_zero = None
            for val in prod:
                if val != 0:
                    first_non_zero = val
                    break
            if first_non_zero is not None and first_non_zero > (q - 1) // 2:
                prod = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
            
            mul_table[i][j] = idx_map[prod]
            
    for i in range(n):
        a, b, c, d = elements[i]
        inv_m = (d, (q - b) % q, (q - c) % q, a)
        first_non_zero = None
        for val in inv_m:
            if val != 0:
                first_non_zero = val
                break
        if first_non_zero is not None and first_non_zero > (q - 1) // 2:
            inv_m = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
        inverses[i] = idx_map[inv_m]
        
        cur = i
        order = 1
        while cur != ident_idx:
            cur = mul_table[cur][i]
            order += 1
            if order > q + 2:
                break
        orders[i] = order
        
    return elements, idx_map, inverses, orders, mul_table, ident_idx

def _check_girth_cayley_optimized(n, candidates, mul_table, ident_idx, g):
    # Check if there is a word of length < g in candidates that equals identity.
    # This is equivalent to checking if the Cayley graph has a cycle of length < g.
    # We do a BFS from ident_idx.
    # If we reach ident_idx at distance d > 0 and d < g, return False.
    
    dist = [-1] * n
    dist[ident_idx] = 0
    queue = deque([ident_idx])
    
    max_depth = g - 1
    
    while queue:
        u = queue.popleft()
        du = dist[u]
        
        if du >= max_depth:
            continue
            
        for s in candidates:
            v = mul_table[u][s]
            
            if v == ident_idx:
                # Found a cycle of length du + 1
                if du + 1 < g:
                    return False
            else:
                if dist[v] == -1:
                    dist[v] = du + 1
                    queue.append(v)
                    
    return True

def _construct_luw(k, g):
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
