"""Improved program for R(K2,2,2) maximization via vectorized K2,2,2 counting + simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_k222_np(adj, n):
    """
    Count K2,2,2 subgraphs in graph represented by n×n adjacency matrix (0/1, symmetric, zero diag).
    K2,2,2 = complete tripartite K_{2,2,2}: 6 vertices partitioned into 3 pairs, all 12 cross-edges present.
    Equivalent: find 3 disjoint pairs (a,b), (c,d), (e,f) such that all 12 edges between them exist.
    
    Approach: enumerate pairs of non-adjacent vertices as the "missing internal edges" of each pair.
    For K2,2,2, the 6 vertices have NO edges within each pair but ALL edges between pairs.
    So we need: non-edge (a,b), non-edge (c,d), non-edge (e,f), all disjoint,
    and all 12 cross-edges present.
    """
    if n < 6:
        return 0
    
    # Get list of non-edges
    non_edges = []
    for i in range(n):
        row = adj[i]
        for j in range(i + 1, n):
            if row[j] == 0:
                non_edges.append((i, j))
    
    if len(non_edges) < 3:
        return 0
    
    count = 0
    len_ne = len(non_edges)
    
    for i1 in range(len_ne):
        a, b = non_edges[i1]
        # Common neighbors of a and b
        common_ab = adj[a] & adj[b]
        common_ab[a] = 0
        common_ab[b] = 0
        
        for i2 in range(i1 + 1, len_ne):
            c, d = non_edges[i2]
            if c == a or c == b or d == a or d == b:
                continue
            if common_ab[c] == 0 or common_ab[d] == 0:
                continue
            
            # Common neighbors of a,b,c,d
            common_all = common_ab & adj[c] & adj[d]
            common_all[c] = 0
            common_all[d] = 0
            
            # Need 2 vertices e,f from common_all that form a non-edge
            cands = np.nonzero(common_all)[0]
            if len(cands) < 2:
                continue
            
            for k1 in range(len(cands)):
                e = cands[k1]
                for k2 in range(k1 + 1, len(cands)):
                    f = cands[k2]
                    if adj[e, f] == 0:
                        count += 1
    
    return count

def _total_violations_np(adj, n):
    """Total K2,2,2 count in both color classes."""
    v0 = _count_k222_np(adj, n)
    v1 = _count_k222_np(1 - adj, n)
    return v0 + v1

def _is_valid_np(adj, n):
    return _total_violations_np(adj, n) == 0

def _paley_mask(p):
    """Build Paley graph adjacency matrix (p prime, p ≡ 1 mod 4). Color 0 = edge."""
    squares = set(x * x % p for x in range(1, p))
    adj = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj

def _adj_to_colors(adj, n):
    """Convert n×n adj matrix to color list (0=adjacent, 1=non-adjacent) in lex order."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if adj[i, j] == 1 else 1)
    return colors

def _colors_to_adj(colors, n):
    """Convert color list to n×n adj matrix."""
    adj = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj[i, j] = 1
                adj[j, i] = 1
            idx += 1
    return adj

def _simulated_annealing(adj, n, deadline, max_iters=500000):
    """
    Simulated annealing to minimize total K2,2,2 violations.
    Returns (adj, final_violations).
    """
    adj = adj.copy()
    current_viol = _total_violations_np(adj, n)
    if current_viol == 0:
        return adj, 0
    
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_edges = len(edge_list)
    
    t0 = 1.0
    t_min = 0.001
    cooling = 0.9995
    
    best_adj = adj.copy()
    best_viol = current_viol
    
    no_improve = 0
    best_no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Dynamic temperature
        t = max(t_min, t0 * (cooling ** it))
        
        # Pick random edge
        ei = random.randint(0, n_edges - 1)
        i, j = edge_list[ei]
        
        # Flip
        adj[i, j] ^= 1
        adj[j, i] ^= 1
        
        new_viol = _total_violations_np(adj, n)
        delta = new_viol - current_viol
        
        if delta <= 0 or random.random() < np.exp(-delta / max(t, 1e-10)):
            current_viol = new_viol
            no_improve = 0
            if current_viol < best_viol:
                best_viol = current_viol
                best_adj = adj.copy()
                best_no_improve = 0
            else:
                best_no_improve += 1
            if current_viol == 0:
                break
        else:
            # Revert
            adj[i, j] ^= 1
            adj[j, i] ^= 1
            no_improve += 1
            best_no_improve += 1
        
        # Restart if stuck too long
        if best_no_improve > 5000:
            # Perturb: flip random edges
            for _ in range(random.randint(3, 8)):
                pi, pj = random.choice(edge_list)
                adj[pi, pj] ^= 1
                adj[pj, pi] ^= 1
            current_viol = _total_violations_np(adj, n)
            if current_viol < best_viol:
                best_viol = current_viol
                best_adj = adj.copy()
            best_no_improve = 0
            t0 = max(t, 0.5)
    
    return best_adj, best_viol

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (evaluator kills at 150s)
    
    best_n = 1
    best_colors = []
    
    # Try Paley colorings as starting points
    start_adj = None
    for p in [29, 17, 13, 5]:
        if p % 4 != 1:
            continue
        adj_test = _paley_mask(p)
        if _is_valid_np(adj_test, p):
            start_adj = adj_test
            best_n = p
            best_colors = _adj_to_colors(adj_test, p)
            break
    
    if start_adj is None:
        # Fallback: try random small graphs
        for n_try in [7, 6, 5]:
            for attempt in range(30):
                adj_rand = np.zeros((n_try, n_try), dtype=np.int8)
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if random.random() < 0.5:
                            adj_rand[i, j] = 1
                            adj_rand[j, i] = 1
                if _is_valid_np(adj_rand, n_try):
                    start_adj = adj_rand
                    best_n = n_try
                    best_colors = _adj_to_colors(adj_rand, n_try)
                    break
            if start_adj is not None:
                break
    
    if start_adj is None:
        return {'n': 1, 'colors': []}
    
    current_adj = start_adj.copy()
    current_n = best_n
    
    # Try to extend from current_n to 30
    for target_n in range(current_n + 1, 31):
        if time.time() > deadline:
            break
        
        n = target_n
        found_valid = False
        
        # Budget per target: use remaining time divided among restarts
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        num_restarts = min(12, max(3, int(remaining / 3)))
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize: keep old edges, randomize new ones
            adj_new = np.zeros((n, n), dtype=np.int8)
            adj_new[:current_n, :current_n] = current_adj
            
            # New edges: random
            for i in range(current_n, n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_new[i, j] = 1
                        adj_new[j, i] = 1
            for i in range(current_n):
                for j in range(current_n, n):
                    if random.random() < 0.5:
                        adj_new[i, j] = 1
                        adj_new[j, i] = 1
            
            # Simulated annealing
            adj_fixed, viol = _simulated_annealing(adj_new, n, deadline, max_iters=200000)
            
            if viol == 0:
                found_valid = True
                current_adj = adj_fixed
                current_n = n
                best_n = n
                best_colors = _adj_to_colors(adj_fixed, n)
                break
        
        if not found_valid:
            # Try a few more with different bias
            for extra in range(5):
                if time.time() > deadline:
                    break
                
                adj_new = np.zeros((n, n), dtype=np.int8)
                adj_new[:current_n, :current_n] = current_adj
                
                # Use biased initialization: try to match Paley-like structure
                for i in range(current_n, n):
                    for j in range(i + 1, n):
                        adj_new[i, j] = 1 if random.random() < 0.45 else 0
                        adj_new[j, i] = adj_new[i, j]
                for i in range(current_n):
                    for j in range(current_n, n):
                        adj_new[i, j] = 1 if random.random() < 0.45 else 0
                        adj_new[j, i] = adj_new[i, j]
                
                adj_fixed, viol = _simulated_annealing(adj_new, n, deadline, max_iters=150000)
                
                if viol == 0:
                    found_valid = True
                    current_adj = adj_fixed
                    current_n = n
                    best_n = n
                    best_colors = _adj_to_colors(adj_fixed, n)
                    break
        
        if not found_valid:
            break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
