"""Improved program for R(K2,2,2) maximization via vectorized local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_k222_vectorized(adj, n):
    """
    Count K2,2,2 subgraphs in a graph using vectorized NumPy operations.
    
    K2,2,2 is a complete tripartite graph with parts of size 2.
    A K2,2,2 exists if there are 3 disjoint non-edges (a,b), (c,d), (e,f)
    such that all cross-pairs among {a,b,c,d,e,f} are edges.
    
    Equivalently, for each pair of disjoint non-edges (a,b) and (c,d),
    we check if there exists a non-edge (e,f) in the common neighborhood
    of {a,b,c,d}.
    
    For efficiency, we use the complement: count K2,2,2 in the complement
    graph where edges are non-edges of the original.
    """
    if n < 6:
        return 0
    
    # Get non-edges (edges in complement)
    # adj is n x n boolean, symmetric, zero diagonal
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                non_edges.append((i, j))
    
    if len(non_edges) < 3:
        return 0
    
    # For each pair of disjoint non-edges, check for third non-edge in common neighborhood
    count = 0
    len_ne = len(non_edges)
    
    # Precompute adjacency for speed
    adj_bool = adj
    
    for i1 in range(len_ne):
        a, b = non_edges[i1]
        
        # Common neighbors of a and b (vertices adjacent to both)
        common_ab = np.where(adj_bool[a] & adj_bool[b])[0]
        common_ab = common_ab[(common_ab != a) & (common_ab != b)]
        
        for i2 in range(i1 + 1, len_ne):
            c, d = non_edges[i2]
            
            # Check disjoint
            if c == a or c == b or d == a or d == b:
                continue
            
            # c and d must be in common_ab
            if c not in common_ab or d not in common_ab:
                continue
            
            # Common neighbors of all four: a, b, c, d
            common_all = common_ab
            common_all = common_all[(common_all != c) & (common_all != d)]
            
            # Check which of these are also neighbors of c and d
            # (they already are neighbors of a and b by construction)
            valid_candidates = []
            for v in common_all:
                if adj_bool[c, v] and adj_bool[d, v]:
                    valid_candidates.append(v)
            
            if len(valid_candidates) < 2:
                continue
            
            # Check if any pair in valid_candidates is a non-edge
            for k1 in range(len(valid_candidates)):
                e = valid_candidates[k1]
                for k2 in range(k1 + 1, len(valid_candidates)):
                    f = valid_candidates[k2]
                    if not adj_bool[e, f]:
                        count += 1
    
    return count

def _is_valid_fast(adj, n):
    """Check that neither color class contains K2,2,2 using vectorized count."""
    if _count_k222_vectorized(adj, n) > 0:
        return False
    # Complement
    comp = ~adj
    np.fill_diagonal(comp, False)
    comp = (comp | comp.T)  # Ensure symmetric
    np.fill_diagonal(comp, False)
    if _count_k222_vectorized(comp, n) > 0:
        return False
    return True

def _local_search_vectorized(adj, n, deadline, max_iters=50000):
    """
    Local search using vectorized K2,2,2 counting.
    Flip edges to minimize total violations in both color classes.
    """
    adj = adj.copy()
    comp = ~adj
    np.fill_diagonal(comp, False)
    comp = (comp | comp.T)
    np.fill_diagonal(comp, False)
    
    def total_violations():
        v0 = _count_k222_vectorized(adj, n)
        v1 = _count_k222_vectorized(comp, n)
        return v0 + v1
    
    current_viol = total_violations()
    if current_viol == 0:
        return adj
    
    no_improve_count = 0
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick random edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Flip edge
        old_val = adj[i, j]
        adj[i, j] = not old_val
        adj[j, i] = not old_val
        
        # Update complement
        comp[i, j] = old_val
        comp[j, i] = old_val
        
        new_viol = total_violations()
        
        if new_viol <= current_viol:
            current_viol = new_viol
            no_improve_count = 0
            if current_viol == 0:
                break
        else:
            # Revert
            adj[i, j] = old_val
            adj[j, i] = old_val
            comp[i, j] = not old_val
            comp[j, i] = not old_val
            no_improve_count += 1
            
            # If stuck for too long, try a random restart of the local search
            if no_improve_count > 1000:
                # Perturb: flip 3-5 random edges
                for _ in range(random.randint(3, 5)):
                    pi = random.randint(0, n - 1)
                    pj = random.randint(0, n - 1)
                    if pi != pj:
                        adj[pi, pj] = not adj[pi, pj]
                        adj[pj, pi] = not adj[pj, pi]
                        comp[pi, pj] = not comp[pi, pj]
                        comp[pj, pi] = not comp[pj, pi]
                current_viol = total_violations()
                no_improve_count = 0
    
    return adj

def _paley_coloring(p):
    """Paley coloring of K_p where p is prime, p ≡ 1 mod 4."""
    if p % 4 != 1:
        return None
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            colors.append(0 if diff in squares else 1)
    return colors

def _colors_to_adj(colors, n):
    """Convert color list to adjacency matrix (color 0 = edge in graph 0)."""
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            adj[i, j] = (colors[idx] == 0)
            adj[j, i] = (colors[idx] == 0)
            idx += 1
    return adj

def _adj_to_colors(adj, n):
    """Convert adjacency matrix to color list."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if adj[i, j] else 1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 1
    best_colors = []
    
    # Try Paley colorings as starting points
    start_adj = None
    for p in [29, 17, 13, 5]:
        if p % 4 != 1:
            continue
        cols = _paley_coloring(p)
        if cols is None:
            continue
        adj_test = _colors_to_adj(cols, p)
        if _is_valid_fast(adj_test, p):
            start_adj = adj_test
            best_n = p
            best_colors = cols
            break
    
    if start_adj is None:
        # Fallback: try random small graphs
        for n_try in [5, 6, 7]:
            for attempt in range(20):
                adj_rand = np.random.rand(n_try, n_try) > 0.5
                np.fill_diagonal(adj_rand, False)
                adj_rand = (adj_rand | adj_rand.T)
                np.fill_diagonal(adj_rand, False)
                if _is_valid_fast(adj_rand, n_try):
                    start_adj = adj_rand
                    best_n = n_try
                    best_colors = _adj_to_colors(adj_rand, n_try)
                    break
            if start_adj is not None:
                break
    
    if start_adj is None:
        # Absolute fallback: n=1
        return {'n': 1, 'colors': []}
    
    # Now try to extend from best_n to 30
    current_adj = start_adj.copy()
    current_n = best_n
    
    # Multi-start local search for each target size
    for target_n in range(current_n + 1, 31):
        if time.time() > deadline:
            break
        
        n = target_n
        found_valid = False
        
        # Try multiple random initializations
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize: keep old edges, randomize new ones
            adj_new = np.zeros((n, n), dtype=bool)
            adj_new[:current_n, :current_n] = current_adj
            
            # New edges: random
            for i in range(current_n, n):
                for j in range(i + 1, n):
                    adj_new[i, j] = random.random() < 0.5
                    adj_new[j, i] = adj_new[i, j]
            for i in range(current_n):
                for j in range(current_n, n):
                    adj_new[i, j] = random.random() < 0.5
                    adj_new[j, i] = adj_new[i, j]
            
            # Local search
            adj_fixed = _local_search_vectorized(adj_new, n, deadline, max_iters=20000)
            
            if _is_valid_fast(adj_fixed, n):
                found_valid = True
                current_adj = adj_fixed
                current_n = n
                best_n = n
                best_colors = _adj_to_colors(adj_fixed, n)
                break
        
        if not found_valid:
            # If we can't extend to this size, try a few more restarts
            for extra_restart in range(3):
                if time.time() > deadline:
                    break
                
                adj_new = np.zeros((n, n), dtype=bool)
                adj_new[:current_n, :current_n] = current_adj
                
                for i in range(current_n, n):
                    for j in range(i + 1, n):
                        adj_new[i, j] = random.random() < 0.5
                        adj_new[j, i] = adj_new[i, j]
                for i in range(current_n):
                    for j in range(current_n, n):
                        adj_new[i, j] = random.random() < 0.5
                        adj_new[j, i] = adj_new[i, j]
                
                adj_fixed = _local_search_vectorized(adj_new, n, deadline, max_iters=15000)
                
                if _is_valid_fast(adj_fixed, n):
                    found_valid = True
                    current_adj = adj_fixed
                    current_n = n
                    best_n = n
                    best_colors = _adj_to_colors(adj_fixed, n)
                    break
        
        if not found_valid:
            # Can't extend further, break
            break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
