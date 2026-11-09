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

def build_bitsets(adj, n):
    """Build neighbor and non-neighbor bitsets from adjacency matrix."""
    neighbors = [0] * n
    non_neighbors = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                if adj[i, j]:
                    neighbors[i] |= (1 << j)
                else:
                    non_neighbors[i] |= (1 << j)
    return neighbors, non_neighbors

def has_k5(neighbors, n):
    """Check for K5 using bitsets. Returns True if K5 exists."""
    for v0 in range(n):
        n0 = neighbors[v0]
        for v1 in range(v0 + 1, n):
            if not (n0 >> v1) & 1:
                continue
            n01 = n0 & neighbors[v1]
            for v2 in range(v1 + 1, n):
                if not (n01 >> v2) & 1:
                    continue
                n012 = n01 & neighbors[v2]
                for v3 in range(v2 + 1, n):
                    if not (n012 >> v3) & 1:
                        continue
                    n0123 = n012 & neighbors[v3]
                    # Check if any vertex > v3 is connected to all of v0,v1,v2,v3
                    mask = n0123 & (~((1 << (v3 + 1)) - 1))
                    if mask:
                        return True
    return False

def has_independent5(non_neighbors, n):
    """Check for independent set of size 5 using bitsets. Returns True if exists."""
    for v0 in range(n):
        nn0 = non_neighbors[v0]
        for v1 in range(v0 + 1, n):
            if not (nn0 >> v1) & 1:
                continue
            nn01 = nn0 & non_neighbors[v1]
            for v2 in range(v1 + 1, n):
                if not (nn01 >> v2) & 1:
                    continue
                nn012 = nn01 & non_neighbors[v2]
                for v3 in range(v2 + 1, n):
                    if not (nn012 >> v3) & 1:
                        continue
                    nn0123 = nn012 & non_neighbors[v3]
                    mask = nn0123 & (~((1 << (v3 + 1)) - 1))
                    if mask:
                        return True
    return False

def is_5_5_free(adj, n):
    """Check if graph has no K5 and no independent set of size 5."""
    neighbors, non_neighbors = build_bitsets(adj, n)
    if has_k5(neighbors, n):
        return False
    if has_independent5(non_neighbors, n):
        return False
    return True

def count_violations(neighbors, non_neighbors, n):
    """Count number of K5 cliques and independent 5-sets. Lower is better."""
    k5_count = 0
    for v0 in range(n):
        n0 = neighbors[v0]
        for v1 in range(v0 + 1, n):
            if not (n0 >> v1) & 1:
                continue
            n01 = n0 & neighbors[v1]
            for v2 in range(v1 + 1, n):
                if not (n01 >> v2) & 1:
                    continue
                n012 = n01 & neighbors[v2]
                for v3 in range(v2 + 1, n):
                    if not (n012 >> v3) & 1:
                        continue
                    n0123 = n012 & neighbors[v3]
                    mask = n0123 & (~((1 << (v3 + 1)) - 1))
                    if mask:
                        k5_count += bin(mask).count('1')
    
    ind_count = 0
    for v0 in range(n):
        nn0 = non_neighbors[v0]
        for v1 in range(v0 + 1, n):
            if not (nn0 >> v1) & 1:
                continue
            nn01 = nn0 & non_neighbors[v1]
            for v2 in range(v1 + 1, n):
                if not (nn01 >> v2) & 1:
                    continue
                nn012 = nn01 & non_neighbors[v2]
                for v3 in range(v2 + 1, n):
                    if not (nn012 >> v3) & 1:
                        continue
                    nn0123 = nn012 & non_neighbors[v3]
                    mask = nn0123 & (~((1 << (v3 + 1)) - 1))
                    if mask:
                        ind_count += bin(mask).count('1')
    
    return k5_count + ind_count

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
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
        if time.time() > deadline - 30:
            break
        n, adj = build_candidate(rule)
        if is_5_5_free(adj, n):
            # Convert to edge list
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        edges.append((i, j))
            return n, edges
    
    # Local search for n=38 using violation counting
    if time.time() < deadline - 10:
        n_try = 38
        np.random.seed(42)
        
        best_violations = float('inf')
        best_adj = None
        
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Start from Paley(37) + one extra vertex, or random
            if restart == 0:
                # Start from Paley(37) with an extra vertex connected to ~half
                adj = np.zeros((n_try, n_try), dtype=bool)
                for i in range(37):
                    for j in range(i + 1, 37):
                        if (j - i) % 37 in {(x * x) % 37 for x in range(1, 37)}:
                            adj[i, j] = True
                            adj[j, i] = True
                # Connect vertex 37 to random subset
                for i in range(37):
                    if np.random.rand() < 0.5:
                        adj[i, 37] = True
                        adj[37, i] = True
            else:
                # Random start with p=0.5
                adj = np.zeros((n_try, n_try), dtype=bool)
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        val = np.random.rand() < 0.5
                        adj[i, j] = val
                        adj[j, i] = val
            
            neighbors, non_neighbors = build_bitsets(adj, n_try)
            current_violations = count_violations(neighbors, non_neighbors, n_try)
            
            if current_violations == 0:
                edges = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if adj[i, j]:
                            edges.append((i, j))
                return n_try, edges
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_adj = adj.copy()
            
            # Hill climbing: try flipping edges to reduce violations
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                
                u = np.random.randint(0, n_try)
                v = np.random.randint(0, n_try)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                # Flip edge
                old_val = adj[u, v]
                adj[u, v] = not old_val
                adj[v, u] = adj[u, v]
                
                # Update bitsets incrementally
                if adj[u, v]:
                    neighbors[u] |= (1 << v)
                    neighbors[v] |= (1 << u)
                    non_neighbors[u] &= ~(1 << v)
                    non_neighbors[v] &= ~(1 << u)
                else:
                    neighbors[u] &= ~(1 << v)
                    neighbors[v] &= ~(1 << u)
                    non_neighbors[u] |= (1 << v)
                    non_neighbors[v] |= (1 << u)
                
                new_violations = count_violations(neighbors, non_neighbors, n_try)
                
                if new_violations == 0:
                    edges = []
                    for i in range(n_try):
                        for j in range(i + 1, n_try):
                            if adj[i, j]:
                                edges.append((i, j))
                    return n_try, edges
                
                if new_violations <= current_violations:
                    current_violations = new_violations
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_adj = adj.copy()
                else:
                    # Revert
                    adj[u, v] = old_val
                    adj[v, u] = old_val
                    if old_val:
                        neighbors[u] |= (1 << v)
                        neighbors[v] |= (1 << u)
                        non_neighbors[u] &= ~(1 << v)
                        non_neighbors[v] &= ~(1 << u)
                    else:
                        neighbors[u] &= ~(1 << v)
                        neighbors[v] &= ~(1 << u)
                        non_neighbors[u] |= (1 << v)
                        non_neighbors[v] |= (1 << u)
        
        # If we found a valid graph during search, return it
        if best_adj is not None and is_5_5_free(best_adj, n_try):
            edges = []
            for i in range(n_try):
                for j in range(i + 1, n_try):
                    if best_adj[i, j]:
                        edges.append((i, j))
            return n_try, edges
    
    return best_n, best_edges
