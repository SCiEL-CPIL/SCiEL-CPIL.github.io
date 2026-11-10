import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    if k == 5 and g == 9:
        # Try PSL(2,q) Cayley graphs for promising q values
        best_n = float('inf')
        best_edges = None
        
        for q in [13, 17, 19, 23]:
            if time.time() > deadline - 20:
                break
                
            n = q * (q * q - 1) // 2
            if n >= best_n:
                continue
                
            # Generate PSL(2,q) group
            try:
                group, idx_map, orders, involutions = _generate_psl2(q)
            except Exception:
                continue
                
            if not involutions:
                continue
                
            # Search for valid generator sets
            found = _search_psl_generators(group, idx_map, orders, involutions, q, g, deadline)
            if found:
                n_found, edges_found = found
                if n_found < best_n:
                    best_n = n_found
                    best_edges = edges_found
                    break  # Found valid graph, return it
        
        if best_edges is not None:
            return best_n, best_edges
        
        # Fallback to LUW
        return _construct_luw(5, 9)
    
    # Fallback for other k, g
    return _construct_luw(k, g)


def _generate_psl2(q):
    """Generate PSL(2,q) group elements with their properties."""
    elements = []
    idx_map = {}
    
    # Generate all SL(2,q) matrices and identify PSL(2,q)
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
                
                # Normalize to PSL: identify M and -M
                m = (a, b, c, d)
                m_norm = _normalize_psl(m, q)
                
                if m_norm not in idx_map:
                    idx_map[m_norm] = len(elements)
                    elements.append(m_norm)
    
    # Compute orders and find involutions
    orders = [1] * len(elements)
    involutions = []
    
    for i, m in enumerate(elements):
        # Compute order by repeated multiplication
        prod = (1, 0, 0, 1)  # Identity
        for _ in range(1, q * q):
            prod = _mul_psl(prod, m, q)
            if prod == (1, 0, 0, 1):
                orders[i] = _
                break
        if orders[i] == 2:
            involutions.append(i)
    
    return elements, idx_map, orders, involutions


def _normalize_psl(m, q):
    """Normalize a PSL(2,q) matrix to canonical form."""
    a, b, c, d = m
    # Find first non-zero entry
    for val in [a, b, c, d]:
        if val != 0:
            if val > (q - 1) // 2:
                return ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
            break
    return m


def _mul_psl(m1, m2, q):
    """Multiply two PSL(2,q) matrices."""
    a1, b1, c1, d1 = m1
    a2, b2, c2, d2 = m2
    a = (a1 * a2 + b1 * c2) % q
    b = (a1 * b2 + b1 * d2) % q
    c = (c1 * a2 + d1 * c2) % q
    d = (c1 * b2 + d1 * d2) % q
    # Normalize
    return _normalize_psl((a, b, c, d), q)


def _search_psl_generators(group, idx_map, orders, involutions, q, g, deadline):
    """Search for valid 5-generator sets in PSL(2,q)."""
    n = len(group)
    random.seed(42)
    
    # We need 2 pairs + 1 involution
    non_involutions = [i for i in range(n) if orders[i] != 2]
    
    if len(non_involutions) < 2:
        return None
    
    max_attempts = 5000
    attempts = 0
    
    while attempts < max_attempts:
        if time.time() > deadline - 10:
            break
        attempts += 1
        
        # Pick s1, s2 from non-involutions, s3 from involutions
        s1 = random.choice(non_involutions)
        s2 = random.choice(non_involutions)
        s3 = random.choice(involutions)
        
        # Get the actual group elements
        g1 = group[s1]
        g2 = group[s2]
        g3 = group[s3]
        
        # Compute inverses
        g1_inv = _inverse_psl(g1, q)
        g2_inv = _inverse_psl(g2, q)
        
        # Get indices
        g1_inv_idx = idx_map[g1_inv]
        g2_inv_idx = idx_map[g2_inv]
        
        # Check all 5 are distinct
        candidates = [s1, g1_inv_idx, s2, g2_inv_idx, s3]
        if len(set(candidates)) != 5:
            continue
        
        # Check girth using non-backtracking walks
        if _check_girth_nonbacktracking(n, candidates, group, idx_map, g, deadline):
            # Build edges
            edges = []
            for i in range(n):
                gi = group[i]
                for s_idx in candidates:
                    s_elem = group[s_idx]
                    neighbor = _mul_psl(gi, s_elem, q)
                    j = idx_map[neighbor]
                    if i < j:
                        edges.append((i, j))
            
            if len(edges) == n * 5 // 2:
                return (n, edges)
    
    return None


def _inverse_psl(m, q):
    """Compute inverse of PSL(2,q) matrix."""
    a, b, c, d = m
    # Inverse of [[a,b],[c,d]] is [[d,-b],[-c,a]]
    inv = (d, (q - b) % q, (q - c) % q, a)
    return _normalize_psl(inv, q)


def _check_girth_nonbacktracking(n, candidates, group, idx_map, g, deadline):
    """Check girth >= g using non-backtracking walks from each vertex."""
    start_time = time.time()
    
    # Precompute neighbors for the 5 generators
    # neighbors[i] = list of 5 neighbor indices
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        gi = group[i]
        for s_idx in candidates:
            s_elem = group[s_idx]
            neighbor = _mul_psl(gi, s_elem, q_from_group(group))
            neighbors[i].append(idx_map[neighbor])
    
    # Non-backtracking BFS from each vertex
    # State: (vertex, previous_vertex, distance)
    # We track visited states to avoid cycles
    
    for start in range(n):
        if time.time() > deadline - 5:
            return False  # Assume girth not satisfied to be safe
        
        # BFS with non-backtracking constraint
        # dist[v][prev] = shortest non-backtracking walk from start to v not coming from prev
        # Simplified: use (v, prev) as state
        
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1))
        
        while queue:
            v, prev, dist = queue.popleft()
            
            if dist >= g:
                continue
            
            for neighbor in neighbors[v]:
                if neighbor == prev:
                    continue  # Non-backtracking
                
                if (neighbor, v) in visited:
                    # Found a cycle
                    # The cycle length is at least dist + 1
                    # But we need to be more careful
                    if dist + 1 < g:
                        return False
                    continue
                
                visited.add((neighbor, v))
                queue.append((neighbor, v, dist + 1))
    
    return True


def q_from_group(group):
    """Infer q from group size."""
    # n = q(q^2-1)/2
    n = len(group)
    # Solve q^3 - q = 2n
    # Try small q values
    for q in range(5, 100):
        if q * (q * q - 1) // 2 == n:
            return q
    return 13  # Default


def _construct_luw(k, g):
    """Fallback LUW construction."""
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
