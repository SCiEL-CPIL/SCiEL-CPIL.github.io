"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _count_c4(adj):
    """Count 4-cycles in a graph given its adjacency matrix (vectorized)."""
    common = adj @ adj
    c = common - adj
    iu = np.triu_indices(common.shape[0], k=1)
    vals = c[iu]
    return int(np.sum(vals * (vals - 1) // 2))

def _build_polarity_graph():
    """
    Construct the polarity graph of PG(2,5) with 90 edges.
    For q=5 (odd), the orthogonal polarity graph of PG(2,q) has:
    - n = q^2 + q + 1 = 31 vertices
    - 90 edges (C4-free)
    """
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q - 2, q)
                        vec = [(v * inv) % q for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]], dtype=int)
    
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            x = np.array(points[i], dtype=int)
            y = np.array(points[j], dtype=int)
            val = (x @ M @ y) % q
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                edges.append([i, j])
    
    return edges, adj

def _delta_c4_add(adj, i, j):
    """Compute change in C4 count when adding edge (i,j)."""
    ni = adj[i]
    nj = adj[j]
    common_with_j = adj @ nj
    return int(ni @ common_with_j)

def _delta_c4_remove(adj, i, j):
    """Compute change in C4 count when removing edge (i,j). Returns negative of C4s destroyed."""
    # C4s destroyed = number of C4s containing edge (i,j)
    # = number of paths i-a-b-j where a~i, b~j, a~b, a!=b
    ni = adj[i].copy()
    nj = adj[j].copy()
    ni[i] = 0
    nj[j] = 0
    # Temporarily remove edge i-j for counting
    adj[i, j] = 0
    adj[j, i] = 0
    common_with_j = adj @ nj
    count = int(ni @ common_with_j)
    adj[i, j] = 1
    adj[j, i] = 1
    return -count

def _greedy_local_search(adj, n, max_iters=5000, deadline=None):
    """
    Greedy local search: try all single edge swaps (remove one, add one)
    and pick the one that most reduces C4 count. Repeat until no improvement.
    """
    current_c4 = _count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        best_delta = 0
        best_swap = None
        
        # Get current edges and non-edges
        edges_set = set()
        non_edges_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    edges_set.add((i, j))
                else:
                    non_edges_list.append((i, j))
        
        # For each non-edge e2, compute C4s it would create
        # For each edge e1, compute C4s it destroys
        # Net change = C4s_created_by_e2 - C4s_destroyed_by_e1 + interaction
        
        # Simpler: just try each swap and compute delta directly
        # But that's O(|E|*|non-E|) which is ~465*465 = 216k, each needing O(n^2) work
        # Too slow. Instead, precompute.
        
        # Precompute for each non-edge: how many C4s it creates
        c4_create = {}
        for e2 in non_edges_list:
            i, j = e2
            c4_create[e2] = _delta_c4_add(adj, i, j)
        
        # Precompute for each edge: how many C4s it destroys
        c4_destroy = {}
        for e1 in edges_set:
            i, j = e1
            c4_destroy[e1] = -_delta_c4_remove(adj, i, j)  # positive number
        
        # For a swap (remove e1, add e2):
        # new_c4 = current_c4 - c4_destroy[e1] + c4_create[e2] + interaction
        # where interaction accounts for C4s that use both e1 and e2
        # This is complex. Let's just do a limited search.
        
        # Try all swaps but only evaluate promising ones
        # First, find best e2 to add (min c4_create)
        min_create = min(c4_create.values()) if c4_create else 0
        best_e2_candidates = [e2 for e2, c in c4_create.items() if c <= min_create + 2]
        
        # Find best e1 to remove (max c4_destroy)
        max_destroy = max(c4_destroy.values()) if c4_destroy else 0
        best_e1_candidates = [e1 for e1, c in c4_destroy.items() if c >= max_destroy - 2]
        
        # Try all combinations of candidate e1 and e2
        for e1 in best_e1_candidates:
            i1, j1 = e1
            for e2 in best_e2_candidates:
                i2, j2 = e2
                if (i1, j1) == (i2, j2) or (i1, j1) == (j2, i2):
                    continue
                # Compute actual C4 count after swap
                adj[i1, j1] = 0
                adj[j1, i1] = 0
                adj[i2, j2] = 1
                adj[j2, i2] = 1
                new_c4 = _count_c4(adj)
                adj[i1, j1] = 1
                adj[j1, i1] = 1
                adj[i2, j2] = 0
                adj[j2, i2] = 0
                
                delta = new_c4 - current_c4
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (e1, e2)
                    improved = True
        
        if not improved:
            # Try all swaps if limited search didn't find improvement
            for e1 in edges_set:
                i1, j1 = e1
                for e2 in non_edges_list:
                    i2, j2 = e2
                    if (i1, j1) == (i2, j2) or (i1, j1) == (j2, i2):
                        continue
                    adj[i1, j1] = 0
                    adj[j1, i1] = 0
                    adj[i2, j2] = 1
                    adj[j2, i2] = 1
                    new_c4 = _count_c4(adj)
                    adj[i1, j1] = 1
                    adj[j1, i1] = 1
                    adj[i2, j2] = 0
                    adj[j2, i2] = 0
                    
                    delta = new_c4 - current_c4
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (e1, e2)
                        improved = True
                    if deadline is not None and time.time() > deadline:
                        break
                if deadline is not None and time.time() > deadline:
                    break
            
            if not improved:
                break
        
        # Apply best swap
        e1, e2 = best_swap
        i1, j1 = e1
        i2, j2 = e2
        adj[i1, j1] = 0
        adj[j1, i1] = 0
        adj[i2, j2] = 1
        adj[j2, i2] = 1
        current_c4 += best_delta
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_adj = adj.copy()
    
    return best_adj, best_c4

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Build the polarity graph (C4-free, 90 edges)
    edges_90, adj = _build_polarity_graph()
    
    n = 31
    c4_base = _count_c4(adj)
    
    # Find the non-edge with minimum C4 creation
    best_single = None
    best_single_c4 = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                delta = _delta_c4_add(adj, i, j)
                if delta < best_single_c4:
                    best_single_c4 = delta
                    best_single = [i, j]
    
    # Add the best single edge
    if best_single is not None:
        i, j = best_single
        adj[i, j] = 1
        adj[j, i] = 1
    
    # Run greedy local search to minimize C4 count
    adj_opt, c4_opt = _greedy_local_search(adj, n, max_iters=2000, deadline=deadline - 10.0)
    
    # Extract edges from optimized adjacency matrix
    edges_final = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_opt[i, j] == 1:
                edges_final.append([i, j])
    
    # Verify we have exactly 91 edges
    assert len(edges_final) == 91, f"Expected 91 edges, got {len(edges_final)}"
    
    return edges_final

# EVOLVE-BLOCK-END
