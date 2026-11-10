import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 135.0
    
    if k != 5 or g != 9:
        return _construct_luw(k, g)
    
    # Try PSL(2, 13) with n = 1092
    best_n, best_edges = _try_psl2_13(deadline)
    
    if best_n < float('inf'):
        return best_n, best_edges
    
    # Fallback to LUW
    return _construct_luw(5, 9)

def _try_psl2_13(deadline):
    q = 13
    n = q * (q * q - 1) // 2  # 1092
    
    # Generate PSL(2, q) elements
    group, idx_map = _generate_psl2(q)
    if len(group) != n:
        return float('inf'), []
    
    # Precompute multiplication table
    mul_table = [[0] * n for _ in range(n)]
    for i, gi in enumerate(group):
        for j, gj in enumerate(group):
            prod = _mul_psl2(gi, gj, q)
            mul_table[i][j] = idx_map[prod]
    
    # Find involutions (elements of order 2)
    involutions = []
    for i, gi in enumerate(group):
        if i == 0:
            continue
        if mul_table[i][i] == 0:  # gi * gi = identity
            involutions.append(i)
    
    if not involutions:
        return float('inf'), []
    
    # Find elements of order > 2 (for the pairs)
    non_involutions = [i for i in range(1, n) if mul_table[i][i] != 0]
    
    # Search for valid generator sets
    max_attempts = 20000
    attempts = 0
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        # Pick two non-involution elements and one involution
        s1 = random.choice(non_involutions)
        s2 = random.choice(non_involutions)
        s3 = random.choice(involutions)
        
        # Ensure all 5 generators are distinct
        s1_inv = mul_table[s1][0]  # Wait, inverse of s1 is the element j such that mul_table[s1][j] == 0
        # Actually, I need to find the inverse properly
        s1_inv = _find_inverse(s1, mul_table, n)
        s2_inv = _find_inverse(s2, mul_table, n)
        
        if s1_inv == -1 or s2_inv == -1:
            continue
        
        gens = [s1, s1_inv, s2, s2_inv, s3]
        if len(set(gens)) != 5:
            continue
        
        # Check girth
        if _check_girth(n, gens, 9, mul_table):
            # Construct edges
            edges = []
            for i in range(n):
                for gen in gens:
                    j = mul_table[i][gen]
                    if i < j:
                        edges.append((i, j))
            
            if len(edges) == n * 5 // 2:
                return n, edges
    
    return float('inf'), []

def _find_inverse(i, mul_table, n):
    for j in range(n):
        if mul_table[i][j] == 0:
            return j
    return -1

def _check_girth(n, gens, g, mul_table):
    """Check if Cayley graph has girth at least g."""
    # BFS from each vertex up to depth (g-1)//2
    # Check if any vertex is reached twice
    
    max_depth = (g - 1) // 2  # 4 for g=9
    
    for start in range(n):
        # BFS from start
        visited = {}  # node -> depth
        queue = deque([(start, 0)])
        visited[start] = 0
        
        while queue:
            u, d = queue.popleft()
            if d >= max_depth:
                continue
            for gen in gens:
                v = mul_table[u][gen]
                if v in visited:
                    # Found a cycle of length d + 1 + visited[v]
                    # Wait, this isn't right. The cycle is from start to u (depth d),
                    # then edge u->v, then v back to start (depth visited[v]).
                    # The cycle length is d + 1 + visited[v] only if the paths are disjoint.
                    # Actually, the cycle length is at most d + 1 + visited[v].
                    if d + 1 + visited[v] < g:
                        return False
                else:
                    visited[v] = d + 1
                    queue.append((v, d + 1))
    
    return True

def _generate_psl2(q):
    """Generate all elements of PSL(2, q)."""
    elements = []
    idx_map = {}
    
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
                    a_inv = pow(a, q - 2, q)
                    d = (d * a_inv) % q
                
                if (a * d - b * c) % q != 1:
                    continue
                
                # Normalize: make the first non-zero entry in [1, (q-1)//2]
                m = (a, b, c, d)
                m = _normalize_psl2(m, q)
                
                if m not in idx_map:
                    idx_map[m] = len(elements)
                    elements.append(m)
    
    return elements, idx_map

def _normalize_psl2(m, q):
    a, b, c, d = m
    for val in [a, b, c, d]:
        if val != 0:
            if val > (q - 1) // 2:
                return ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
            break
    return m

def _mul_psl2(m1, m2, q):
    a1, b1, c1, d1 = m1
    a2, b2, c2, d2 = m2
    a = (a1 * a2 + b1 * c2) % q
    b = (a1 * b2 + b1 * d2) % q
    c = (c1 * a2 + d1 * c2) % q
    d = (c1 * b2 + d1 * d2) % q
    return _normalize_psl2((a, b, c, d), q)

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
