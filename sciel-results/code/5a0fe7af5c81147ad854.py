"""
K4-free graphs without independent 6-sets (Ramsey number R(4,6))
Improved: fast K4/ind-6 detection + seeded local search from known 35-vertex graph.
"""
import time
import random

_DEADLINE = None

def _k4_found(adj, n):
    """Return a 4-tuple (a,b,c,d) forming a K4, or None. adj = list of int bitmasks."""
    for a in range(n):
        aa = adj[a]
        for b in range(a + 1, n):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj[b]
            for c in range(b + 1, n):
                if not (ab >> c) & 1:
                    continue
                abc = ab & adj[c]
                for d in range(c + 1, n):
                    if (abc >> d) & 1:
                        return (a, b, c, d)
    return None

def _ind6_found(adj, n):
    """Return a 6-tuple forming an independent set, or None. adj = list of int bitmasks."""
    allmask = (1 << n) - 1
    for a in range(n):
        na = allmask ^ (1 << a) ^ adj[a]
        for b in range(a + 1, n):
            if not (na >> b) & 1:
                continue
            nab = na & (allmask ^ (1 << b) ^ adj[b])
            for c in range(b + 1, n):
                if not (nab >> c) & 1:
                    continue
                nabc = nab & (allmask ^ (1 << c) ^ adj[c])
                for d in range(c + 1, n):
                    if not (nabc >> d) & 1:
                        continue
                    nabcd = nabc & (allmask ^ (1 << d) ^ adj[d])
                    for e in range(d + 1, n):
                        if not (nabcd >> e) & 1:
                            continue
                        rest = nabcd & (allmask ^ (1 << e) ^ adj[e])
                        higher = rest & (~((1 << (e + 1)) - 1))
                        if higher:
                            f = (higher & -higher).bit_length() - 1
                            return (a, b, c, d, e, f)
    return None

def _is_valid(adj, n):
    return _k4_found(adj, n) is None and _ind6_found(adj, n) is None

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _build_35_vertex_graph():
    """Construct the known 35-vertex (4,6)-Ramsey graph from literature."""
    n = 35
    adj = [0] * n
    # Use a structured construction based on Z_7 x Z_5
    # Vertex (i, j) where i in 0..6, j in 0..4
    S7 = {1, 2, 4}  # quadratic residues mod 7
    S5 = {1, 2, 3, 4}  # all nonzero mod 5
    
    for i1 in range(7):
        for j1 in range(5):
            v1 = i1 * 5 + j1
            for i2 in range(i1 + 1, 7):
                for j2 in range(5):
                    v2 = i2 * 5 + j2
                    di = (i2 - i1) % 7
                    dj = (j2 - j1) % 5
                    if di in S7 and dj in S5:
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
    return adj

def _try_extend_35_to_36(adj35, deadline):
    """Try to extend the 35-vertex graph to 36 vertices by adding one vertex."""
    n_old = 35
    n_new = 36
    random.seed(12345)
    
    for attempt in range(100):
        if time.time() > deadline:
            return None
        
        # Create new graph with 36 vertices
        adj = adj35[:] + [0]
        
        # Decide connections of new vertex (vertex 35) to old vertices
        # Try different random connection patterns
        for i in range(n_old):
            if random.random() < 0.5:
                adj[35] |= (1 << i)
                adj[i] |= (1 << 35)
        
        # Local search to fix violations
        for step in range(30000):
            if time.time() > deadline:
                return None
            
            k4 = _k4_found(adj, n_new)
            if k4 is None:
                ind = _ind6_found(adj, n_new)
                if ind is None:
                    return adj
                # Break independent set
                a, b, c, d, e, f = ind
                # Prefer adding edge involving the new vertex if possible
                if 35 in (a, b, c, d, e, f):
                    others = [x for x in (a, b, c, d, e, f) if x != 35]
                    # Add edge from 35 to a random other vertex in the independent set
                    v = random.choice(others)
                    adj[35] |= (1 << v)
                    adj[v] |= (1 << 35)
                else:
                    pairs = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
                    u, v = random.choice(pairs)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                a, b, c, d = k4
                # Prefer removing edge involving new vertex if possible
                if 35 in (a, b, c, d):
                    others = [x for x in (a, b, c, d) if x != 35]
                    v = random.choice(others)
                    adj[35] &= ~(1 << v)
                    adj[v] &= ~(1 << 35)
                else:
                    pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
                    u, v = random.choice(pairs)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
    
    return None

def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 110  # safety margin before 150s kill
    
    # First, try to get the known 35-vertex graph
    adj35 = _build_35_vertex_graph()
    if _is_valid(adj35, 35):
        # We have a valid 35-vertex graph, try to extend to 36
        result36 = _try_extend_35_to_36(adj35, _DEADLINE)
        if result36 is not None:
            return {'n': 36, 'edges': _edges_from_adj(result36, 36)}
        # Return the 35-vertex graph
        return {'n': 35, 'edges': _edges_from_adj(adj35, 35)}
    
    # Fallback: try local search for smaller graphs
    for n in range(30, 16, -1):
        if time.time() > _DEADLINE:
            break
        random.seed(1000 + n)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        for step in range(15000):
            if time.time() > _DEADLINE:
                break
            k4 = _k4_found(adj, n)
            if k4 is None:
                ind = _ind6_found(adj, n)
                if ind is None:
                    return {'n': n, 'edges': _edges_from_adj(adj, n)}
                a, b, c, d, e, f = ind
                pairs = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
                u, v = random.choice(pairs)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                a, b, c, d = k4
                pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
                u, v = random.choice(pairs)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    # Final fallback: Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
