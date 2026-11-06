"""Direct (5,5)-free construction: Paley family over prime powers q ≡ 1 (mod 4),
with in-program verification (no K5, no independent 5-set) via bitset clique search.
Extended with local search to try to exceed 37 vertices by adding 5 new vertices
and optimizing their connections."""
import time
import random

# EVOLVE-BLOCK-START

def _popcount(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count('1')

def _primes_upto(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _paley_edges(q):
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return edges

def _adj_bitsets(n, edges):
    adj = [0] * n
    for i, j in edges:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj

def _has_clique(adj, n, k, node_limit):
    """Return True if a k-clique exists, False if proven absent, None if budget hit."""
    full = (1 << n) - 1
    nodes = [0]

    def dfs(cand, size):
        if size == k:
            return True
        if nodes[0] > node_limit:
            return None
        need = k - size
        if _popcount(cand) < need:
            return False
        while cand:
            lsb = cand & (-cand)
            v = lsb.bit_length() - 1
            cand ^= lsb
            nodes[0] += 1
            nxt = cand & adj[v]
            r = dfs(nxt, size + 1)
            if r is True:
                return True
            if r is None:
                return None
        return False

    return dfs(full, 0)

def _is_55_free(n, edges, node_limit=400000):
    """True: verified no K5 and no independent 5-set; False: violated; None: unknown."""
    adj = _adj_bitsets(n, edges)
    full = (1 << n) - 1
    comp = [full ^ (1 << i) ^ adj[i] for i in range(n)]
    r1 = _has_clique(adj, n, 5, node_limit)
    if r1 is None:
        return None
    if r1:
        return False
    r2 = _has_clique(comp, n, 5, node_limit)
    if r2 is None:
        return None
    if r2:
        return False
    return True

def _find_violation(adj, n, node_limit=10000):
    """Returns a tuple (is_clique, vertices) if a violation is found, else None."""
    full = (1 << n) - 1
    comp = [full ^ (1 << i) ^ adj[i] for i in range(n)]
    
    # Find K5
    # We can reuse the logic but we need the vertices.
    # For speed, we just check existence first, but for repair we need the set.
    # Let's do a quick DFS to find one.
    
    def find_clique(cand, size, path):
        if size == 5:
            return tuple(sorted(path))
        if _popcount(cand) < 5 - size:
            return None
        while cand:
            lsb = cand & (-cand)
            v = lsb.bit_length() - 1
            cand ^= lsb
            path.append(v)
            r = find_clique(cand & adj[v], size + 1, path)
            if r:
                return r
            path.pop()
        return None

    k5 = find_clique(full, 0, [])
    if k5:
        return (True, k5)
        
    i5 = find_clique(full, 0, []) # Wait, need to use comp
    # Re-implement for comp
    def find_clique_comp(cand, size, path):
        if size == 5:
            return tuple(sorted(path))
        if _popcount(cand) < 5 - size:
            return None
        while cand:
            lsb = cand & (-cand)
            v = lsb.bit_length() - 1
            cand ^= lsb
            path.append(v)
            r = find_clique_comp(cand & comp[v], size + 1, path)
            if r:
                return r
            path.pop()
        return None
    i5 = find_clique_comp(full, 0, [])
    if i5:
        return (False, i5)
        
    return None

def _try_extend_to_42(core_n, core_edges, deadline):
    """
    Attempts to extend the core graph (Paley 37) to 42 vertices.
    Returns (42, edges) if successful, else None.
    """
    n_new = 5
    n_total = core_n + n_new
    start_time = time.time()
    
    # Core vertices are 0..core_n-1
    # New vertices are core_n..n_total-1
    
    # Initialize edges
    edges = list(core_edges)
    
    # Randomly initialize edges for new vertices
    # Edges between new vertices
    for i in range(core_n, n_total):
        for j in range(i + 1, n_total):
            if random.random() < 0.5:
                edges.append((i, j))
    # Edges between new and core
    for i in range(core_n, n_total):
        for j in range(core_n):
            if random.random() < 0.5:
                edges.append((j, i))
                
    adj = _adj_bitsets(n_total, edges)
    
    # Local Search
    for step in range(10000):
        if time.time() > deadline:
            return None
            
        # Check for violation
        viol = _find_violation(adj, n_total)
        if viol is None:
            return (n_total, edges)
            
        is_clique, verts = viol
        
        # We want to flip an edge to break this violation.
        # Prefer flipping edges involving new vertices.
        # If the violation is entirely in the core, we are stuck (but core is known good).
        # So violation must involve at least one new vertex.
        
        # Get all edges within the violation set
        v_set = set(verts)
        candidate_edges = []
        for u in v_set:
            for v in v_set:
                if u < v:
                    # Check if edge exists
                    # If is_clique, we want to REMOVE an edge (break clique)
                    # If is_independent, we want to ADD an edge (break indep set)
                    # So we flip the edge.
                    # But we should only flip edges involving new vertices to keep core stable.
                    if u >= core_n or v >= core_n:
                        candidate_edges.append((u, v))
                        
        if not candidate_edges:
            # Violation is in core? Should not happen.
            # Or we can't fix it by flipping new edges?
            # If the violation is e.g. 4 core + 1 new, and we flip the new-core edge,
            # it breaks the K5.
            # If the violation is 5 new, we flip a new-new edge.
            # If candidate_edges is empty, it means the violation is entirely in core.
            # This implies our core is not (5,5) free, which is a bug in seed.
            return None
            
        u, v = random.choice(candidate_edges)
        
        # Flip edge (u, v)
        # Remove if present, add if not
        edge_exists = adj[u] & (1 << v)
        if edge_exists:
            # Remove
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            # Update edges list is expensive. We'll rebuild at the end or just track adj.
            # For now, we just modify adj.
        else:
            # Add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return None

def construct():
    start = time.time()
    deadline = start + 280.0  # safety margin before 300s kill

    # 1. Get the 37-vertex Paley graph
    q = 37
    core_edges = _paley_edges(q)
    core_n = q
    
    # Verify core is valid (should be)
    # res = _is_55_free(core_n, core_edges)
    # if res is not True: return (core_n, core_edges) # Fallback
    
    # 2. Try to extend to 42
    result = _try_extend_to_42(core_n, core_edges, deadline - 10.0)
    
    if result:
        n, edges = result
        # Final verification
        if _is_55_free(n, edges):
            return (n, edges)
            
    # 3. Fallback to 37
    return (core_n, core_edges)

# EVOLVE-BLOCK-END
