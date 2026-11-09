import time
import numpy as np
import itertools

def paley_graph(q):
    """Construct the Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges

def is_5_5_free_fast(adj, n):
    """
    Fast check: no K5 and no independent set of size 5.
    Uses bitsets for speed. adj is a numpy bool array of shape (n, n).
    """
    # Precompute neighbor bitsets
    # For each vertex, a bitmask of its neighbors (and non-neighbors)
    # n <= 46, so we can use Python ints as bitsets
    neighbors = [0] * n
    non_neighbors = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                if adj[i, j]:
                    neighbors[i] |= (1 << j)
                else:
                    non_neighbors[i] |= (1 << j)
    
    # Check for K5: for each 4-tuple, check if they form a K4 and if any 5th vertex connects to all 4
    # Better: enumerate all 5-tuples but use bitset intersection
    # For n=38, C(38,5) = 501942. With bitset ops this is fast.
    
    # Alternative: check all 5-sets using bitset intersection
    # A 5-set {v0,v1,v2,v3,v4} is a clique iff for each vi, the other 4 are all in neighbors[vi]
    # i.e., neighbors[vi] & mask_of_others == mask_of_others
    
    # For independent set: non_neighbors[vi] & mask_of_others == mask_of_others
    
    for v0 in range(n):
        for v1 in range(v0+1, n):
            # Quick check: are v0 and v1 adjacent?
            if not adj[v0, v1]:
                continue
            for v2 in range(v1+1, n):
                if not adj[v0, v2] or not adj[v1, v2]:
                    continue
                mask012 = (1 << v0) | (1 << v1) | (1 << v2)
                # Check if v0,v1,v2 form a triangle
                if (neighbors[v0] & ((1<<v1)|(1<<v2))) != ((1<<v1)|(1<<v2)):
                    continue
                if (neighbors[v1] & ((1<<v0)|(1<<v2))) != ((1<<v0)|(1<<v2)):
                    continue
                if (neighbors[v2] & ((1<<v0)|(1<<v1))) != ((1<<v0)|(1<<v1)):
                    continue
                for v3 in range(v2+1, n):
                    if not adj[v0, v3] or not adj[v1, v3] or not adj[v2, v3]:
                        continue
                    # Check K4
                    if (neighbors[v3] & ((1<<v0)|(1<<v1)|(1<<v2))) != ((1<<v0)|(1<<v1)|(1<<v2)):
                        continue
                    mask0123 = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3)
                    for v4 in range(v3+1, n):
                        if not adj[v0, v4] or not adj[v1, v4] or not adj[v2, v4] or not adj[v3, v4]:
                            continue
                        # Check K5
                        if (neighbors[v4] & mask0123) == mask0123:
                            return False
    
    # Check for independent set of size 5
    for v0 in range(n):
        for v1 in range(v0+1, n):
            if adj[v0, v1]:
                continue
            for v2 in range(v1+1, n):
                if adj[v0, v2] or adj[v1, v2]:
                    continue
                for v3 in range(v2+1, n):
                    if adj[v0, v3] or adj[v1, v3] or adj[v2, v3]:
                        continue
                    for v4 in range(v3+1, n):
                        if adj[v0, v4] or adj[v1, v4] or adj[v2, v4] or adj[v3, v4]:
                            continue
                        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Fallback: Paley graph on 37 vertices
    n_seed, edges_seed = paley_graph(37)
    best_n = n_seed
    best_edges = edges_seed
    
    # Helper to build adjacency matrix from edges
    def build_adj(n, edges):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = True
            adj[v, u] = True
        return adj

    # Try deterministic constructions for n=38
    # 2-fold cover of Paley(19)
    
    n_base = 19
    _, base_edges = paley_graph(n_base)
    base_adj = build_adj(n_base, base_edges)
    residues = {(x * x) % n_base for x in range(1, n_base)}
    
    def build_candidate(cross_rule):
        n = 2 * n_base
        adj = np.zeros((n, n), dtype=bool)
        
        # Internal edges
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if base_adj[i, j]:
                    adj[i, j] = True
                    adj[j, i] = True
                    adj[i + n_base, j + n_base] = True
                    adj[j + n_base, i + n_base] = True
        
        # Cross edges
        for i in range(n_base):
            for j in range(n_base):
                if cross_rule(i, j, base_adj[i, j]):
                    adj[i, j + n_base] = True
                    adj[j + n_base, i] = True
                    
        return n, adj

    # Generate many cross-edge rules to try
    candidates = []
    
    # Rule: base edge exists
    candidates.append(("base_edge", lambda i, j, e: e))
    # Rule: base edge does NOT exist
    candidates.append(("base_noedge", lambda i, j, e: not e))
    # Rule: i == j
    candidates.append(("diag", lambda i, j, e: i == j))
    # Rule: i != j
    candidates.append(("offdiag", lambda i, j, e: i != j))
    
    # Rule: (j - i) mod n_base is a quadratic residue
    for shift in range(n_base):
        candidates.append((f"paley_shift_{shift}", 
            (lambda s: lambda i, j, e: ((j - i + s) % n_base) in residues)(shift)))
    
    # Rule: (i - j) mod n_base is a quadratic residue
    for shift in range(n_base):
        candidates.append((f"paley_shift_neg_{shift}", 
            (lambda s: lambda i, j, e: ((i - j + s) % n_base) in residues)(shift)))
    
    # Rule: base_adj[i, j] XOR (i == j)
    candidates.append(("xor_diag", lambda i, j, e: e ^ (i == j)))
    # Rule: base_adj[i, j] XOR (i != j)
    candidates.append(("xor_offdiag", lambda i, j, e: e ^ (i != j)))
    
    # Rule: base_adj[i, j] AND (i == j)
    candidates.append(("and_diag", lambda i, j, e: e and (i == j)))
    # Rule: base_adj[i, j] OR (i == j)
    candidates.append(("or_diag", lambda i, j, e: e or (i == j)))
    
    # Rule: base_adj[i, j] AND (i != j)
    candidates.append(("and_offdiag", lambda i, j, e: e and (i != j)))
    # Rule: base_adj[i, j] OR (i != j)
    candidates.append(("or_offdiag", lambda i, j, e: e or (i != j)))
    
    # Rule: base_adj[i, j] XOR ((j - i) mod n_base in residues)
    for shift in range(n_base):
        candidates.append((f"xor_paley_{shift}", 
            (lambda s: lambda i, j, e: e ^ (((j - i + s) % n_base) in residues))(shift)))
    
    # Rule: base_adj[i, j] AND ((j - i) mod n_base in residues)
    for shift in range(n_base):
        candidates.append((f"and_paley_{shift}", 
            (lambda s: lambda i, j, e: e and (((j - i + s) % n_base) in residues))(shift)))
    
    # Rule: base_adj[i, j] OR ((j - i) mod n_base in residues)
    for shift in range(n_base):
        candidates.append((f"or_paley_{shift}", 
            (lambda s: lambda i, j, e: e or (((j - i + s) % n_base) in residues))(shift)))
    
    # Try all candidates
    for name, rule in candidates:
        if time.time() > deadline - 5:
            break
        n, adj = build_candidate(rule)
        if is_5_5_free_fast(adj, n):
            # Convert to edge list
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        edges.append((i, j))
            return n, edges
    
    # If no deterministic construction works, try a local search for n=38
    if time.time() < deadline - 20:
        n_try = 38
        # Start from a random graph with p=0.5
        np.random.seed(42)
        adj = np.zeros((n_try, n_try), dtype=bool)
        for i in range(n_try):
            for j in range(i+1, n_try):
                val = np.random.rand() < 0.5
                adj[i, j] = val
                adj[j, i] = val
        
        # Local search: flip edges to reduce violations
        # We need a fast way to count violations
        # Use the fast checker but we need incremental updates
        
        # Simpler: just try random restarts with the fast checker
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            adj = np.zeros((n_try, n_try), dtype=bool)
            for i in range(n_try):
                for j in range(i+1, n_try):
                    val = np.random.rand() < 0.5
                    adj[i, j] = val
                    adj[j, i] = val
            
            if is_5_5_free_fast(adj, n_try):
                edges = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if adj[i, j]:
                            edges.append((i, j))
                return n_try, edges
            
            # Hill climbing: try flipping edges
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                u = np.random.randint(0, n_try)
                v = np.random.randint(0, n_try)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                # Flip edge
                adj[u, v] = not adj[u, v]
                adj[v, u] = adj[u, v]
                
                if is_5_5_free_fast(adj, n_try):
                    edges = []
                    for i in range(n_try):
                        for j in range(i + 1, n_try):
                            if adj[i, j]:
                                edges.append((i, j))
                    return n_try, edges
                
                # Keep the flip if it doesn't make things worse
                # For simplicity, just keep all flips (random walk)
    
    return best_n, best_edges
