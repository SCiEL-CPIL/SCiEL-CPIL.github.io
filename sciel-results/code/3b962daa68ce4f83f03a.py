"""Voltage graph construction over K_{5,5} for cage problem n(5,9)."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import itertools

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using voltage graph
    construction over K_{k,k} with voltage assignments from Z_m.
    
    The base graph is K_{k,k} (complete bipartite). We assign voltages to edges
    from Z_m. The lift has k*m vertices (m copies of each of the 2k base vertices).
    We search for the smallest m such that the lift has girth >= g.
    """
    if k != 5:
        # Fallback for other values of k
        return _generic_construct(k, g)
    
    # For k=5, g=9: search for smallest m with voltage graph over K_{5,5}
    # giving girth >= 9.
    
    # The voltage graph: base is K_{5,5}. Left vertices L_0..L_4, right R_0..R_4.
    # Edge (L_i, R_j) gets voltage v_{i,j} in Z_m.
    # In the lift, vertex (L_i, t) connects to (R_j, t + v_{i,j}) for all j.
    # Vertex (R_j, t) connects to (L_i, t + v_{i,j}) for all i.
    # (Voltage on edge is symmetric: going L->R adds voltage, R->L adds voltage too,
    #  or we can define it so that the total around a cycle is the sum of voltages.)
    
    # A cycle in the lift corresponds to a closed walk in the base where the
    # sum of voltages (with appropriate signs) equals 0 mod m.
    # For K_{k,k}, the shortest cycles in the base are of length 4:
    # L_i - R_j - L_l - R_m - L_i (i != l, j != m)
    # The voltage sum for this 4-cycle: v_{i,j} - v_{l,j} + v_{l,m} - v_{i,m}
    # (going L_i -> R_j adds v_{i,j}, R_j -> L_l adds -v_{l,j} [since we go backwards],
    #  L_l -> R_m adds v_{l,m}, R_m -> L_i adds -v_{i,m})
    # Wait, let me be more careful.
    
    # Standard voltage graph: each directed edge e = (u,v) has a voltage phi(e) in G.
    # For undirected graphs, we assign voltage to each edge, and when traversing
    # in reverse direction, we use the inverse (which for Z_m is negation).
    
    # So for edge (L_i, R_j) with voltage v_{i,j}:
    #   Traversing L_i -> R_j: adds v_{i,j}
    #   Traversing R_j -> L_i: adds -v_{i,j} (i.e., m - v_{i,j} mod m)
    
    # A 4-cycle L_i - R_j - L_l - R_m - L_i:
    #   L_i -> R_j: +v_{i,j}
    #   R_j -> L_l: -v_{l,j}
    #   L_l -> R_m: +v_{l,m}
    #   R_m -> L_i: -v_{i,m}
    # Total: v_{i,j} - v_{l,j} + v_{l,m} - v_{i,m} mod m
    # For girth >= 5, this must be != 0 for all i!=l, j!=m.
    
    # For girth >= 9, we also need no 6-cycles and no 8-cycles with total voltage 0.
    # 6-cycle: L_i - R_j - L_l - R_m - L_n - R_p - L_i (i,l,n all distinct, j,m,p all distinct)
    # Total: v_{i,j} - v_{l,j} + v_{l,m} - v_{n,m} + v_{n,p} - v_{i,p} mod m
    
    # 8-cycle: L_i - R_j - L_l - R_m - L_n - R_p - L_q - R_r - L_i
    # (i,l,n,q distinct, j,m,p,r distinct)
    # Total: v_{i,j} - v_{l,j} + v_{l,m} - v_{n,m} + v_{n,p} - v_{q,p} + v_{q,r} - v_{i,r} mod m
    
    # Strategy: Try small m values. For each m, search for voltage assignments
    # that avoid zero-sum cycles of length < 9.
    
    # The number of voltage variables is 25 (for 5x5). We can try to find
    # assignments by random search or structured approach.
    
    # Let's try a structured approach: use a Latin-square-like or arithmetic structure.
    # v_{i,j} = (a*i + b*j + c) mod m for some constants a, b, c.
    
    # For a 4-cycle: v_{i,j} - v_{l,j} + v_{l,m} - v_{i,m}
    # = (a*i + b*j + c) - (a*l + b*j + c) + (a*l + b*m + c) - (a*i + b*m + c)
    # = a*(i - l) + a*(l - i) + b*(j - j) + b*(m - m) = 0
    # So this structure always gives zero-sum 4-cycles! Not good.
    
    # Try v_{i,j} = (a*i*j + b*i + c*j + d) mod m
    # 4-cycle sum: v_{i,j} - v_{l,j} + v_{l,m} - v_{i,m}
    # = (a*i*j + b*i + c*j + d) - (a*l*j + b*l + c*j + d) + (a*l*m + b*l + c*m + d) - (a*i*m + b*i + c*m + d)
    # = a*(i*j - l*j + l*m - i*m) + b*(i - l + l - i) + c*(j - j + m - m)
    # = a*(j*(i-l) + m*(l-i)) = a*(i-l)*(j-m)
    # For i!=l and j!=m, this is nonzero iff a*(i-l)*(j-m) != 0 mod m.
    # If m is prime and a != 0, and |i-l| < m, |j-m| < m, then (i-l)*(j-m) != 0 mod m.
    # So for prime m > 4, this works for 4-cycles!
    
    # Now check 6-cycles: v_{i,j} - v_{l,j} + v_{l,m} - v_{n,m} + v_{n,p} - v_{i,p}
    # = a*(i*j - l*j + l*m - n*m + n*p - i*p) + b*(i-l+l-n+n-i) + c*(j-j+m-m+p-p)
    # = a*(j*(i-l) + m*(l-n) + p*(n-i))
    # = a*(j*(i-l) + m*(l-n) + p*(n-i))
    # For distinct i,l,n and distinct j,m,p, this can be zero.
    # E.g., if i=0, l=1, n=2, j=0, m=1, p=2:
    # = a*(0*(0-1) + 1*(1-2) + 2*(2-0)) = a*(0 - 1 + 4) = 3a
    # This is 0 mod m only if m divides 3a. For m=5, 3a != 0. Good.
    # But for other choices it might be zero.
    
    # Let me just do a computational search for small m.
    
    def check_girth(voltages, m, k, g):
        """Check if the voltage graph lift has girth >= g.
        voltages is a k x k array.
        Returns True if girth >= g."""
        # Build the lift graph
        # Vertices: (side, base_idx, coset) where side in {0,1}, base_idx in 0..k-1, coset in 0..m-1
        # Total vertices: 2*k*m
        n_vertices = 2 * k * m
        
        # Build adjacency
        adj = [[] for _ in range(n_vertices)]
        for i in range(k):
            for j in range(k):
                v = voltages[i][j]
                # L_i, t -> R_j, t+v
                for t in range(m):
                    u1 = i * m + t  # L_i, t
                    u2 = k * m + j * m + (t + v) % m  # R_j, t+v
                    adj[u1].append(u2)
                    adj[u2].append(u1)
        
        # Check girth by BFS from each vertex up to depth g-1
        # For girth >= g, no cycle of length < g exists.
        # Equivalently, in BFS from any vertex, no vertex is reached at depth < g/2
        # from two different paths... actually simpler: check that the BFS tree
        # has no cross-edges that would create a short cycle.
        
        # Simpler: for each vertex, do BFS and check that the first time we see
        # a previously visited vertex, the cycle length is >= g.
        # Actually, the standard way: girth is the minimum cycle length.
        # We can check: for each edge (u,v), if dist(u,v) < g-1 in the graph without that edge,
        # then there's a cycle of length dist+1 < g. But that's expensive.
        
        # Alternative: BFS from each vertex, track parent. If we reach a vertex already
        # in the BFS tree (not the parent), the cycle length is dist[u] + dist[v] + 1.
        # We need all such cycle lengths >= g.
        
        # For efficiency, we only need to check up to depth (g-1)//2 from each vertex.
        # If no cycle of length < g is found, girth >= g.
        
        max_depth = (g - 1) // 2
        
        for start in range(n_vertices):
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
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
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle of length dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True
    
    # Search for m and voltage assignments
    # Try m = 5, 6, 7, ... and for each m, try structured voltage assignments
    
    # First try the quadratic form: v_{i,j} = (a*i*j + b*i + c*j + d) mod m
    # with a in 1..m-1, b,c,d in 0..m-1
    
    for m in range(5, 50):
        found = False
        for a in range(1, m):
            for b in range(m):
                for c in range(m):
                    for d in range(m):
                        voltages = [[(a*i*j + b*i + c*j + d) % m for j in range(k)] for i in range(k)]
                        if check_girth(voltages, m, k, g):
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    
    if not found:
        # Try random search
        for m in range(5, 100):
            for attempt in range(1000):
                voltages = [[np.random.randint(m) for _ in range(k)] for _ in range(k)]
                if check_girth(voltages, m, k, g):
                    found = True
                    break
            if found:
                break
    
    if not found:
        # Fallback to Lazebnik-Ustimenko-Woldar
        return _luw_construct(k, g)
    
    # Build the final graph
    n_vertices = 2 * k * m
    edges = []
    for i in range(k):
        for j in range(k):
            v = voltages[i][j]
            for t in range(m):
                u1 = i * m + t
                u2 = k * m + j * m + (t + v) % m
                edges.append((u1, u2))
    
    return n_vertices, edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
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


def _generic_construct(k, g):
    """Generic fallback for k != 5."""
    return _luw_construct(k, g)
# EVOLVE-BLOCK-END
