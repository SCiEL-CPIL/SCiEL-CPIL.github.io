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

def _find_k4(adj, n):
    """Find and return a K4 as a list of 4 vertices, or None."""
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
                        xc = common & adj[w]
                        lsb2 = xc & -xc
                        x = lsb2.bit_length() - 1
                        return [u, v, w, x]
    return None

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
        # Pruning: if number of bits < remaining depth, return False
        if bin(verts).count('1') < 6 - depth:
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

def _find_independent_6(adj, n):
    """Find and return an independent 6-set as a list, or None."""
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
    
    def dfs(verts, path):
        if len(path) == 6:
            return path[:]
        if not verts:
            return None
        if bin(verts).count('1') < 6 - len(path):
            return None
        v = verts
        while v:
            lsb = v & -v
            u = lsb.bit_length() - 1
            v ^= lsb
            new_verts = v & comp_adj[u]
            result = dfs(new_verts, path + [u])
            if result:
                return result
        return None
    
    return dfs(full_mask, [])

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

def _degree(adj, n):
    return [bin(adj[i]).count('1') for i in range(n)]

def _local_search_improved(n, time_limit, seed=None):
    """Improved local search with conflict-driven moves."""
    if seed is not None:
        random.seed(seed)
    
    start = time.time()
    deadline = start + time_limit
    
    # Initialize with a random graph biased toward moderate density
    # For K4-free graphs, we want relatively sparse but not too sparse
    # Target edge probability ~ 0.3-0.4 for n=30-35
    p = 0.35
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    best_valid = None
    best_n = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Check validity
        k4 = _find_k4(adj, n)
        if k4 is None:
            ind6 = _find_independent_6(adj, n)
            if ind6 is None:
                # Valid graph found!
                if n > best_n:
                    best_n = n
                    best_valid = adj[:]
                return _edges_from_adj(best_valid, n)
            else:
                # Has independent 6-set, need to add edges
                # Add an edge that breaks this independent set
                i, j = random.sample(ind6, 2)
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                no_improve_count = 0
        else:
            # Has K4, need to remove an edge
            # Remove an edge from the K4 that appears in the most other K4s
            # For simplicity, remove a random edge from this K4
            u, v, w, x = k4
            # Try all 6 edges and pick one randomly
            edge_pairs = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
            i, j = random.choice(edge_pairs)
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
            no_improve_count = 0
    
    if best_valid is not None:
        return _edges_from_adj(best_valid, n)
    return None

def _local_search_complement(n, time_limit, seed=None):
    """Search in the complement: maximize edges while keeping alpha <= 5 and no K4."""
    if seed is not None:
        random.seed(seed)
    
    start = time.time()
    deadline = start + time_limit
    
    # Start with a dense graph and remove edges to satisfy constraints
    # Actually, let's start sparse and add edges
    # For no K4: if we add edge (u,v), check that u and v don't share 2+ common neighbors
    # For no independent 6: if we remove edge (u,v), check that u and v aren't in an independent 6-set
    
    # Better approach: start with empty graph, add edges one by one
    adj = [0] * n
    
    # Pre-compute which edges can be safely added
    # An edge (u,v) can be added if:
    # 1. u and v have fewer than 2 common neighbors (to avoid K4)
    # 2. Adding it doesn't create an independent 6-set (but adding edges can't create independent sets)
    # Wait: adding edges BREAKS independent sets, so we should add as many as possible!
    # The constraint is only K4-free.
    # So: greedily add edges while maintaining K4-free property.
    
    # Greedy: for each pair (u,v) in random order, add edge if it doesn't create K4
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(pairs)
    
    for u, v in pairs:
        if time.time() > deadline:
            break
        # Check if adding (u,v) would create a K4
        # A K4 containing (u,v) would need two common neighbors w,x that are connected
        common = adj[u] & adj[v]
        # Check if any two vertices in common are connected
        m = common
        has_conflict = False
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            m ^= lsb
            if common & adj[w]:
                has_conflict = True
                break
        if not has_conflict:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Now check if we have an independent 6-set
    # If yes, we need to add more edges (but we already added all safe ones)
    # So this greedy approach might not work well
    
    # Alternative: start with the greedy K4-free graph, then try to add edges
    # that break independent sets (but all K4-safe edges are already added)
    # So we're stuck. Need a different approach.
    
    # Let's try: start with random graph, then iteratively fix violations
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        k4 = _find_k4(adj, n)
        if k4 is not None:
            u, v, w, x = k4
            edge_pairs = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
            i, j = random.choice(edge_pairs)
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
            continue
        
        ind6 = _find_independent_6(adj, n)
        if ind6 is not None:
            # Try to add an edge within the independent set that is K4-safe
            added = False
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = ind6[i], ind6[j]
                    if not (adj[u] >> v) & 1:
                        # Check if adding (u,v) creates K4
                        common = adj[u] & adj[v]
                        m = common
                        safe = True
                        while m:
                            lsb = m & -m
                            w = lsb.bit_length() - 1
                            m ^= lsb
                            if common & adj[w]:
                                safe = False
                                break
                        if safe:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added = True
                            break
                if added:
                    break
            if not added:
                # Can't add any edge within the independent set without creating K4
                # Remove a random edge elsewhere to try to break the independent set
                # This is tricky - removing edges can create independent sets
                # Instead, let's just give up on this configuration
                break
            continue
        
        # Valid!
        return _edges_from_adj(adj, n)
    
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
    # Try from largest to smallest, with multiple seeds
    for target_n in [35, 34, 33, 32, 31, 30, 28, 26, 24]:
        if time.time() > deadline:
            break
        if target_n <= best_n:
            continue
        
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        # Try multiple random seeds with the improved local search
        found = False
        max_attempts = 8
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            time_per_attempt = min((deadline - time.time()) / (max_attempts - attempt), 10)
            if time_per_attempt < 0.5:
                break
            
            edges = _local_search_improved(target_n, time_per_attempt, seed=1000 + attempt)
            if edges is not None:
                g = {'n': target_n, 'edges': edges}
                if _is_valid_graph(target_n, edges):
                    if target_n > best_n:
                        best_graph = g
                        best_n = target_n
                    found = True
                    break
        
        # Also try complement-based search
        if not found and time.time() < deadline:
            remaining = deadline - time.time()
            if remaining > 2:
                for attempt in range(3):
                    if time.time() > deadline:
                        break
                    time_per_attempt = min((deadline - time.time()) / (3 - attempt), 8)
                    if time_per_attempt < 0.5:
                        break
                    edges = _local_search_complement(target_n, time_per_attempt, seed=2000 + attempt)
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
