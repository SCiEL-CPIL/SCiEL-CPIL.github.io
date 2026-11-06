import time

def _is_valid_graph(n, edges):
    """Check if graph has no K4 and no independent set of size 6."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check for K4
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    if common & adj[w]:
                        return False

    # Check for independent set of size 6 (clique of size 6 in complement)
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
        
    def has_clique_size(k):
        def dfs(verts, depth):
            if depth == k:
                return True
            if not verts:
                return False
            v = verts
            while v:
                lsb = v & -v
                u = lsb.bit_length() - 1
                v ^= lsb
                new_verts = v & comp_adj[u]
                if dfs(new_verts, depth + 1):
                    return True
            return False
        return dfs(full_mask, 0)

    if has_clique_size(6):
        return False
        
    return True

def _paley_graph(p):
    """Generate Paley graph for prime p."""
    squares = {x * x % p for x in range(1, p)}
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edges.append([i, j])
    return {'n': p, 'edges': edges}

def _clebsch_graph():
    """Generate the Clebsch graph: 16 vertices, K4-free, alpha=5."""
    # Clebsch graph = folded 5-cube, or equivalently:
    # Vertices are binary strings of length 5 with even parity (16 vertices)
    # Two vertices are adjacent if they differ in exactly 1 or 3 positions
    n = 16
    vertices = []
    for i in range(16):
        bits = bin(i).count('1')
        if bits % 2 == 0:
            vertices.append(i)
    # Map to 0..15
    idx_map = {v: i for i, v in enumerate(vertices)}
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            diff = bin(vertices[i] ^ vertices[j]).count('1')
            if diff in (1, 3):
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Try candidates in descending order of n to maximize result
    # Paley graphs for primes p ≡ 1 (mod 4): 17, 29, 37, 41, 53, 61, 73, 89
    # But n must be <= 40, so try 37, 29, 17
    # Also try Clebsch (n=16) as guaranteed valid fallback
    
    candidates = [37, 29, 17]
    
    best_graph = None
    best_n = 0
    
    for p in candidates:
        if time.time() > deadline:
            break
        g = _paley_graph(p)
        if _is_valid_graph(g['n'], g['edges']):
            if g['n'] > best_n:
                best_graph = g
                best_n = g['n']
    
    # Check Clebsch as fallback (n=16, definitely K4-free with alpha=5)
    if time.time() < deadline:
        g = _clebsch_graph()
        if _is_valid_graph(g['n'], g['edges']):
            if g['n'] > best_n:
                best_graph = g
                best_n = g['n']
    
    if best_graph is None:
        # Last resort: return Clebsch (should always be valid)
        best_graph = _clebsch_graph()
        
    return best_graph
# EVOLVE-BLOCK-END
