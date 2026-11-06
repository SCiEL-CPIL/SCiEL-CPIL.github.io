"""Improved program for R(K2,2,2) maximization via fast bitmask local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _has_k222_bitmask(adj_masks, n):
    """
    Check if graph with adjacency bitmasks contains K2,2,2.
    K2,2,2: complete tripartite with parts {a,b}, {c,d}, {e,f}.
    All 12 cross-edges must be present.
    """
    if n < 6:
        return False
    
    # Build list of non-edges
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj_masks[i] >> j) & 1:
                non_edges.append((i, j))
    
    if len(non_edges) < 3:
        return False
    
    len_ne = len(non_edges)
    
    for i1 in range(len_ne):
        a, b = non_edges[i1]
        # Common neighbors of a and b (vertices adjacent to both)
        common_ab = adj_masks[a] & adj_masks[b]
        # Remove a and b themselves
        common_ab &= ~(1 << a) & ~(1 << b)
        
        for i2 in range(i1 + 1, len_ne):
            c, d = non_edges[i2]
            # Check disjoint
            if c == a or c == b or d == a or d == b:
                continue
            # c and d must be common neighbors of a and b
            if not ((common_ab >> c) & 1) or not ((common_ab >> d) & 1):
                continue
            
            # Common neighbors of all four: a, b, c, d
            common_all = common_ab & adj_masks[c] & adj_masks[d]
            common_all &= ~(1 << c) & ~(1 << d)
            
            # Need at least 2 vertices in common_all that form a non-edge
            # Iterate over set bits
            candidates = []
            mask = common_all
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                candidates.append(v)
                mask ^= lsb
            
            if len(candidates) < 2:
                continue
            
            for k1 in range(len(candidates)):
                e = candidates[k1]
                for k2 in range(k1 + 1, len(candidates)):
                    f = candidates[k2]
                    if not ((adj_masks[e] >> f) & 1):
                        return True
    
    return False

def _is_valid_bitmask(adj_masks, n):
    """Check that neither color class contains K2,2,2."""
    if _has_k222_bitmask(adj_masks, n):
        return False
    # Complement: flip all edges, keep diagonal zero
    comp_masks = [0] * n
    for i in range(n):
        comp = 0
        for j in range(n):
            if i != j:
                if not ((adj_masks[i] >> j) & 1):
                    comp |= (1 << j)
        comp_masks[i] = comp
    if _has_k222_bitmask(comp_masks, n):
        return False
    return True

def _local_search_bitmask(adj_masks, n, deadline, max_iters=200000):
    """
    Local search using bitmask adjacency. Flip edges to eliminate K2,2,2.
    """
    adj_masks = list(adj_masks)
    
    def get_comp_masks(am):
        comp = [0] * n
        for i in range(n):
            c = 0
            for j in range(n):
                if i != j:
                    if not ((am[i] >> j) & 1):
                        c |= (1 << j)
            comp[i] = c
        return comp
    
    def total_violations():
        v0 = 1 if _has_k222_bitmask(adj_masks, n) else 0
        comp = get_comp_masks(adj_masks)
        v1 = 1 if _has_k222_bitmask(comp, n) else 0
        return v0 + v1
    
    current_viol = total_violations()
    if current_viol == 0:
        return adj_masks
    
    no_improve_count = 0
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick random edge
        i, j = random.choice(edge_list)
        
        # Flip edge
        bit = 1 << j
        adj_masks[i] ^= bit
        adj_masks[j] ^= (1 << i)
        
        new_viol = total_violations()
        
        if new_viol <= current_viol:
            current_viol = new_viol
            no_improve_count = 0
            if current_viol == 0:
                break
        else:
            # Revert
            adj_masks[i] ^= bit
            adj_masks[j] ^= (1 << i)
            no_improve_count += 1
            
            if no_improve_count > 200:
                # Perturb
                for _ in range(random.randint(2, 4)):
                    pi, pj = random.choice(edge_list)
                    adj_masks[pi] ^= (1 << pj)
                    adj_masks[pj] ^= (1 << pi)
                current_viol = total_violations()
                no_improve_count = 0
    
    return adj_masks

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

def _colors_to_masks(colors, n):
    """Convert color list to adjacency bitmasks (color 0 = edge in graph 0)."""
    masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
            idx += 1
    return masks

def _masks_to_colors(masks, n):
    """Convert adjacency bitmasks to color list."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (masks[i] >> j) & 1 else 1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 1
    best_colors = []
    
    # Try Paley colorings as starting points
    start_masks = None
    for p in [29, 17, 13, 5]:
        if p % 4 != 1:
            continue
        cols = _paley_coloring(p)
        if cols is None:
            continue
        masks_test = _colors_to_masks(cols, p)
        if _is_valid_bitmask(masks_test, p):
            start_masks = masks_test
            best_n = p
            best_colors = cols
            break
    
    if start_masks is None:
        # Fallback: try random small graphs
        for n_try in [5, 6, 7]:
            for attempt in range(20):
                masks_rand = [0] * n_try
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if random.random() < 0.5:
                            masks_rand[i] |= (1 << j)
                            masks_rand[j] |= (1 << i)
                if _is_valid_bitmask(masks_rand, n_try):
                    start_masks = masks_rand
                    best_n = n_try
                    best_colors = _masks_to_colors(masks_rand, n_try)
                    break
            if start_masks is not None:
                break
    
    if start_masks is None:
        return {'n': 1, 'colors': []}
    
    # Now try to extend from best_n to 30
    current_masks = list(start_masks)
    current_n = best_n
    
    # Multi-start local search for each target size
    for target_n in range(current_n + 1, 31):
        if time.time() > deadline:
            break
        
        n = target_n
        found_valid = False
        
        # Try multiple random initializations
        num_restarts = 8
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize: keep old edges, randomize new ones
            masks_new = [0] * n
            for i in range(current_n):
                masks_new[i] = current_masks[i]
            
            # New edges: random
            for i in range(current_n, n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        masks_new[i] |= (1 << j)
                        masks_new[j] |= (1 << i)
            for i in range(current_n):
                for j in range(current_n, n):
                    if random.random() < 0.5:
                        masks_new[i] |= (1 << j)
                        masks_new[j] |= (1 << i)
            
            # Local search
            masks_fixed = _local_search_bitmask(masks_new, n, deadline, max_iters=100000)
            
            if _is_valid_bitmask(masks_fixed, n):
                found_valid = True
                current_masks = masks_fixed
                current_n = n
                best_n = n
                best_colors = _masks_to_colors(masks_fixed, n)
                break
        
        if not found_valid:
            # Try a few more restarts with different perturbation
            for extra_restart in range(5):
                if time.time() > deadline:
                    break
                
                masks_new = [0] * n
                for i in range(current_n):
                    masks_new[i] = current_masks[i]
                
                for i in range(current_n, n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            masks_new[i] |= (1 << j)
                            masks_new[j] |= (1 << i)
                for i in range(current_n):
                    for j in range(current_n, n):
                        if random.random() < 0.5:
                            masks_new[i] |= (1 << j)
                            masks_new[j] |= (1 << i)
                
                masks_fixed = _local_search_bitmask(masks_new, n, deadline, max_iters=80000)
                
                if _is_valid_bitmask(masks_fixed, n):
                    found_valid = True
                    current_masks = masks_fixed
                    current_n = n
                    best_n = n
                    best_colors = _masks_to_colors(masks_fixed, n)
                    break
        
        if not found_valid:
            break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
