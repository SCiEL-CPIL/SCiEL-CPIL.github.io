import time
import random

def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_k4(adj, n):
    """Check if graph has a K4. Return True if found."""
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
                        return True
    return False

def _has_independent_6(adj, n):
    """Check if graph has an independent set of size 6."""
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
    
    def dfs(verts, depth):
        if depth == 6:
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
    return dfs(full_mask, 0)

def _is_valid_graph(n, edges):
    adj = _build_adj(n, edges)
    if _has_k4(adj, n):
        return False
    if _has_independent_6(adj, n):
        return False
    return True

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        m = adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if u < v:
                edges.append([u, v])
    return edges

def _local_search(n, time_limit, seed=None):
    """Local search to find a valid graph on n vertices."""
    if seed is not None:
        random.seed(seed)
    
    start = time.time()
    deadline = start + time_limit
    
    # Start with random graph
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    best_valid = None
    best_n = 0
    
    while time.time() < deadline:
        # Check validity
        if not _has_k4(adj, n) and not _has_independent_6(adj, n):
            if n > best_n:
                best_n = n
                best_valid = adj[:]
            # Try to continue searching for improvements
            # But we already have a valid graph, so we can return
            break
        
        # Find a violation and try to fix it
        if _has_k4(adj, n):
            # Find a K4 and try to break it by removing an edge
            found = False
            for u in range(n):
                if found:
                    break
                for v in range(u + 1, n):
                    if found:
                        break
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        m = common
                        while m:
                            lsb = m & -m
                            w = lsb.bit_length() - 1
                            m ^= lsb
                            if common & adj[w]:
                                # Found K4: {u, v, w, x} where x is in common & adj[w]
                                # Remove a random edge from this K4
                                x_candidates = common & adj[w]
                                xc = x_candidates
                                while xc:
                                    lsb2 = xc & -xc
                                    x = lsb2.bit_length() - 1
                                    xc ^= lsb2
                                    # Remove edge between two random vertices of the K4
                                    k4 = [u, v, w, x]
                                    i, j = random.sample(k4, 2)
                                    adj[i] &= ~(1 << j)
                                    adj[j] &= ~(1 << i)
                                    found = True
                                    break
                            if found:
                                break
                    if found:
                        break
        else:
            # Has independent 6-set, try to add an edge to break it
            # Find an independent 6-set
            full_mask = (1 << n) - 1
            comp_adj = [0] * n
            for i in range(n):
                comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
            
            def find_independent_6(verts, path):
                if len(path) == 6:
                    return path[:]
                if not verts:
                    return None
                v = verts
                while v:
                    lsb = v & -v
                    u = lsb.bit_length() - 1
                    v ^= lsb
                    new_verts = v & comp_adj[u]
                    result = find_independent_6(new_verts, path + [u])
                    if result:
                        return result
                return None
            
            ind_set = find_independent_6(full_mask, [])
            if ind_set:
                # Add an edge between two random vertices in the independent set
                i, j = random.sample(ind_set, 2)
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    if best_valid is not None:
        return _edges_from_adj(best_valid, n)
    return None

def _clebsch_graph():
    """Generate the Clebsch graph: 16 vertices, K4-free, alpha=5."""
    n = 16
    vertices = []
    for i in range(16):
        bits = bin(i).count('1')
        if bits % 2 == 0:
            vertices.append(i)
    idx_map = {v: i for i, v in enumerate(vertices)}
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            diff = bin(vertices[i] ^ vertices[j]).count('1')
            if diff in (1, 3):
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def _paley_graph(p):
    """Generate Paley graph for prime p."""
    squares = {x * x % p for x in range(1, p)}
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edges.append([i, j])
    return {'n': p, 'edges': edges}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, try known constructions
    candidates = [
        ('paley37', _paley_graph(37)),
        ('paley29', _paley_graph(29)),
        ('paley17', _paley_graph(17)),
        ('clebsch', _clebsch_graph()),
    ]
    
    best_graph = None
    best_n = 0
    
    for name, g in candidates:
        if time.time() > deadline:
            break
        if _is_valid_graph(g['n'], g['edges']):
            if g['n'] > best_n:
                best_graph = g
                best_n = g['n']
    
    # Now try local search for larger n
    for target_n in [35, 32, 30, 28, 25, 22, 20, 18]:
        if time.time() > deadline:
            break
        if target_n <= best_n:
            continue
        
        # Give each target a time budget
        remaining = deadline - time.time()
        if remaining < 2:
            break
        time_per_target = min(remaining / 2, 15)  # At most 15s per target
        
        # Try multiple random seeds
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            edges = _local_search(target_n, time_per_target / 5, seed=42 + attempt)
            if edges is not None:
                g = {'n': target_n, 'edges': edges}
                if _is_valid_graph(target_n, edges):
                    if target_n > best_n:
                        best_graph = g
                        best_n = target_n
                    found = True
                    break
        
        if found:
            break
    
    if best_graph is None:
        best_graph = _clebsch_graph()
    
    return best_graph
# EVOLVE-BLOCK-END
