# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _has_k222_fast(adj, n):
    """
    Returns True if adj (boolean matrix, 1=edge in G) contains K2,2,2.
    Equivalent to: complement graph comp contains an induced subgraph isomorphic to 3K2.
    """
    if n < 6:
        return False
    
    # Build list of edges in complement (non-edges in G)
    comp_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                comp_edges.append((i, j))
                
    m = len(comp_edges)
    if m < 3:
        return False
    
    # Iterate over pairs of disjoint edges in comp
    for i1 in range(m):
        a, b = comp_edges[i1]
        for i2 in range(i1 + 1, m):
            c, d = comp_edges[i2]
            
            # Check disjoint
            if c == a or c == b or d == a or d == b:
                continue
            
            # Check if {a,b} and {c,d} form an induced matching in comp
            # i.e., no edges in comp between these 4 vertices
            # Edges to check in comp: (a,c), (a,d), (b,c), (b,d)
            # If any of these is an edge in comp, it's not an induced matching of size 2.
            # Edge in comp means NOT adj
            if (not adj[a, c]) or (not adj[a, d]) or (not adj[b, c]) or (not adj[b, d]):
                continue
            
            # We have an induced matching of size 2: (a,b) and (c,d)
            # Now look for a third edge (e,f) in comp
            # such that it is disjoint from {a,b,c,d}
            # and no edges in comp between {e,f} and {a,b,c,d}
            
            for i3 in range(i2 + 1, m):
                e, f = comp_edges[i3]
                
                # Check disjoint
                if e == a or e == b or e == c or e == d:
                    continue
                if f == a or f == b or f == c or f == d:
                    continue
                
                # Check no comp edges between {e,f} and {a,b,c,d}
                # This means all pairs (x,y) with x in {e,f}, y in {a,b,c,d} must be edges in G
                # i.e., adj[x,y] must be True
                
                # Check e against a,b,c,d
                if not (adj[e, a] and adj[e, b] and adj[e, c] and adj[e, d]):
                    continue
                # Check f against a,b,c,d
                if not (adj[f, a] and adj[f, b] and adj[f, c] and adj[f, d]):
                    continue
                
                # Found a valid K2,2,2
                return True
                
    return False

def _is_valid_fast(adj, n):
    """Check that neither color class contains K2,2,2 using fast boolean check."""
    if _has_k222_fast(adj, n):
        return False
    # Complement
    comp = ~adj
    np.fill_diagonal(comp, False)
    comp = (comp | comp.T)  # Ensure symmetric
    np.fill_diagonal(comp, False)
    if _has_k222_fast(comp, n):
        return False
    return True

def _local_search_fast(adj, n, deadline):
    """
    Local search using fast boolean validity check.
    Tries all single flips. If none work, perturbs.
    """
    adj = adj.copy()
    
    if _is_valid_fast(adj, n):
        return adj
        
    for step in range(500): # Max steps
        if time.time() > deadline:
            break
            
        # Try all single flips
        for i in range(n):
            for j in range(i + 1, n):
                # Flip
                old_val = adj[i, j]
                adj[i, j] = not old_val
                adj[j, i] = not old_val
                
                if _is_valid_fast(adj, n):
                    return adj
                    
                # Revert
                adj[i, j] = old_val
                adj[j, i] = old_val
                
        # If no single flip worked, perturb
        # Flip 3-5 random edges
        num_perturb = random.randint(3, 5)
        for _ in range(num_perturb):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                if i > j: i, j = j, i
                adj[i, j] = not adj[i, j]
                adj[j, i] = not adj[j, i]
                
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
            adj_fixed = _local_search_fast(adj_new, n, deadline)
            
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
                
                adj_fixed = _local_search_fast(adj_new, n, deadline)
                
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
