"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth >= g using deterministic algebraic lifts."""
    deadline = time.time() + 115
    
    if k != 3:
        # Fallback for k != 3: LUW construction
        return _luw_construct(k, g)
    
    # Petersen graph: 10 vertices, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),  # spokes
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)   # inner star
    ]
    
    def check_girth_bounded(adj, n, min_girth):
        """Check girth >= min_girth using BFS from each vertex, bounded depth."""
        max_depth = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True
    
    def build_algebraic_lift(m, a, b, c):
        """Build a deterministic algebraic lift of Petersen graph.
        For each edge (u,v), the permutation is i -> (a*i + b) % m for forward,
        and the inverse permutation for backward.
        Different edges use different parameters to avoid short cycles."""
        n = 10 * m
        adj = [[] for _ in range(n)]
        
        # Assign different multipliers to each edge type
        # Outer pentagon edges: (0,1), (1,2), (2,3), (3,4), (4,0)
        # Spokes: (0,5), (1,6), (2,7), (3,8), (4,9)
        # Inner star: (5,7), (7,9), (9,6), (6,8), (8,5)
        
        # Use different multipliers for different edge classes
        edge_params = {}
        edge_idx = 0
        for u, v in petersen_edges:
            # Cycle through different multiplier patterns
            if edge_idx < 5:
                # Outer pentagon
                edge_params[(u, v)] = (a, b)
            elif edge_idx < 10:
                # Spokes
                edge_params[(u, v)] = (a + 1, b + 1)
            else:
                # Inner star
                edge_params[(u, v)] = (a + 2, b + 2)
            edge_idx += 1
        
        for u, v in petersen_edges:
            mul, offset = edge_params[(u, v)]
            # Forward: i -> (mul*i + offset) % m
            # Backward: inverse permutation
            inv_mul = _mod_inverse(mul, m)
            if inv_mul is None:
                # If no inverse, skip this parameter set
                return None
            
            for i in range(m):
                u_idx = u * m + i
                v_idx = v * m + (mul * i + offset) % m
                adj[u_idx].append(v_idx)
                adj[v_idx].append(u_idx)
        
        return adj
    
    def _mod_inverse(a, m):
        """Compute modular inverse of a mod m using extended Euclidean algorithm."""
        if m == 1:
            return 0
        g, x, _ = _extended_gcd(a, m)
        if g != 1:
            return None
        return x % m
    
    def _extended_gcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x1, y1 = _extended_gcd(b, a % b)
        return g, y1, x1 - (a // b) * y1
    
    best_result = None
    
    # Try different m values and algebraic parameters
    # For girth 16, we need to search systematically
    m_candidates = []
    for m in range(50, 2000, 10):
        m_candidates.append(m)
    
    # Try different multiplier patterns
    multiplier_patterns = []
    for a in range(2, 20, 2):  # Even multipliers for better distribution
        for b in range(0, 10):
            multiplier_patterns.append((a, b))
    
    for m in m_candidates:
        if time.time() > deadline:
            break
        
        n = 10 * m
        found = False
        
        for a, b in multiplier_patterns:
            if time.time() > deadline:
                break
            
            adj = build_algebraic_lift(m, a, b, 0)
            if adj is None:
                continue
            
            if check_girth_bounded(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_result = (n, edges)
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """LUW construction as fallback."""
    d, q = _luw_params(k, g)
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

def _luw_params(k, g):
    """Compute LUW parameters."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return d, q
# EVOLVE-BLOCK-END
