"""Construct a (5,5)-Ramsey-free graph. Seed: Paley(37). Attempt to extend to n=42."""
import time
import random

# EVOLVE-BLOCK-START
def _paley_37():
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[False] * q for _ in range(q)]
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i][j] = adj[j][i] = True
                edges.append((i, j))
    return adj, edges

def _find_conflict(adj, k, size=5):
    """
    Find a K_5 or independent 5-set containing vertex k in graph adj[0..k-1].
    Returns (is_clique, set_of_vertices) or (None, None).
    is_clique=True means K_5 (clique), False means independent set.
    """
    # For each subset of 4 vertices from 0..k-1, check if adding k creates a conflict
    # We need 4 vertices that form a K_4 (for clique) or independent 4-set (for independent)
    # among themselves, and are all connected to k (for clique) or all not connected to k (for independent).
    
    # Pre-compute: for each vertex v < k, whether edge (k, v) exists
    conn = [adj[k][v] for v in range(k)]
    
    # We need to find 4 vertices {a,b,c,d} such that:
    # For clique: all 6 edges among them exist, and conn[a]=conn[b]=conn[c]=conn[d]=True
    # For independent: all 6 edges among them absent, and conn[a]=conn[b]=conn[c]=conn[d]=False
    
    n = k
    # Use itertools combinations for speed
    from itertools import combinations
    
    # Check for K_5: find 4 mutually connected vertices all connected to k
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        # Check all connected to k
        if conn[a] and conn[b] and conn[c] and conn[d]:
            # Check if they form K_4
            if adj[a][b] and adj[a][c] and adj[a][d] and adj[b][c] and adj[b][d] and adj[c][d]:
                return (True, {a, b, c, d})
    
    # Check for independent 5-set: find 4 mutually non-connected vertices all non-connected to k
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        # Check all NOT connected to k
        if not conn[a] and not conn[b] and not conn[c] and not conn[d]:
            # Check if they form independent 4-set
            if not adj[a][b] and not adj[a][c] and not adj[a][d] and not adj[b][c] and not adj[b][d] and not adj[c][d]:
                return (False, {a, b, c, d})
    
    return (None, None)

def _resolve_conflict(adj, k, is_clique, conflict_set, n_total):
    """
    Given a conflict (K_5 or independent 5-set) involving vertex k and a set of 4 vertices,
    modify one edge between k and a vertex in conflict_set to break the conflict.
    Returns the modified adj matrix and the edge that was changed.
    """
    # For a clique conflict: k is connected to all 4, and they form K_4.
    # Remove one edge (k, v) for v in conflict_set.
    # For an independent conflict: k is NOT connected to all 4, and they form independent 4-set.
    # Add one edge (k, v) for v in conflict_set.
    
    # Choose a vertex to modify. Random choice to diversify.
    v = random.choice(list(conflict_set))
    
    if is_clique:
        # Remove edge
        adj[k][v] = adj[v][k] = False
        return adj, (k, v) if k < v else (v, k), "remove"
    else:
        # Add edge
        adj[k][v] = adj[v][k] = True
        return adj, (k, v) if k < v else (v, k), "add"

def _try_extend(adj_base, n_base, target_n, deadline, max_iterations=5000):
    """
    Try to extend the base graph to target_n vertices while maintaining (5,5)-free property.
    Returns (success, adj, edges) or (False, None, None).
    """
    n = n_base
    adj = [row[:] for row in adj_base]
    
    # Extend adj to target_n
    while len(adj) < target_n:
        adj.append([False] * target_n)
    for i in range(len(adj)):
        adj[i] = adj[i] + [False] * (target_n - len(adj[i]))
    
    # For each new vertex, initialize its edges randomly
    for new_v in range(n_base, target_n):
        for old_v in range(new_v):
            if random.random() < 0.5:
                adj[new_v][old_v] = adj[old_v][new_v] = True
    
    # Local search: resolve conflicts
    iterations = 0
    while iterations < max_iterations:
        if time.time() > deadline:
            return (False, None, None)
        
        # Find any conflict in the entire graph
        conflict_found = False
        for k in range(n_base, target_n):
            result = _find_conflict(adj, k)
            if result[0] is not None:
                is_clique, conflict_set = result
                # Resolve by modifying one edge
                adj, _, _ = _resolve_conflict(adj, k, is_clique, conflict_set, target_n)
                conflict_found = True
                break
        
        if not conflict_found:
            # No conflicts found - success!
            # Build edge list
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if adj[i][j]:
                        edges.append((i, j))
            return (True, adj, edges)
        
        iterations += 1
    
    return (False, None, None)

def construct():
    deadline = time.time() + 280  # Safety margin before 300s limit
    
    # Start with Paley graph on 37 vertices
    adj_base, edges_base = _paley_37()
    n_base = 37
    
    # Try to extend to 42 vertices
    target_n = 42
    random.seed(42)  # Deterministic for reproducibility
    
    # Multiple restarts
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        success, adj, edges = _try_extend(adj_base, n_base, target_n, deadline)
        if success:
            return (target_n, edges)
    
    # Fallback to the base graph
    return (n_base, edges_base)
# EVOLVE-BLOCK-END
