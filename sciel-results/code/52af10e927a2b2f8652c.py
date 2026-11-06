"""Improved program for R(K2,2,2) maximization via bitmask local search."""
# EVOLVE-BLOCK-START
import time
import random

def _has_k222_bitmask(adj_masks, n):
    """
    Check if graph with adjacency masks contains K2,2,2.
    adj_masks[i] is a bitmask where bit j is set if (i,j) is an edge.
    K2,2,2: three disjoint non-edges (a,b),(c,d),(e,f) with all 12 cross-pairs as edges.
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
        # Common neighbors of a and b
        common_ab = adj_masks[a] & adj_masks[b]
        common_ab &= ~(1 << a)
        common_ab &= ~(1 << b)
        
        for i2 in range(i1 + 1, len_ne):
            c, d = non_edges[i2]
            # Check disjoint
            if (1 << c) & (1 << a) or (1 << c) & (1 << b) or (1 << d) & (1 << a) or (1 << d) & (1 << b):
                continue
            # c and d must be in common_ab
            if not ((common_ab >> c) & 1) or not ((common_ab >> d) & 1):
                continue
            
            # Common neighbors of all four
            common_all = common_ab & adj_masks[c] & adj_masks[d]
            common_all &= ~(1 << c)
            common_all &= ~(1 << d)
            
            # Need at least 2 vertices in common_all that form a non-edge
            # Collect candidates
            candidates = []
            mask = common_all
            while mask:
                low_bit = mask & (-mask)
                v = low_bit.bit_length() - 1
                candidates.append(v)
                mask ^= low_bit
            
            if len(candidates) < 2:
                continue
            
            # Check if any pair is a non-edge
            for k1 in range(len(candidates)):
                e = candidates[k1]
                for k2 in range(k1 + 1, len(candidates)):
                    f = candidates[k2]
                    if not ((adj_masks[e] >> f) & 1):
                        return True
    
    return False

def _count_k222_bitmask(adj_masks, n):
    """Count K2,2,2 subgraphs using bitmask operations."""
    if n < 6:
        return 0
    
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj_masks[i] >> j) & 1:
                non_edges.append((i, j))
    
    if len(non_edges) < 3:
        return 0
    
    count = 0
    len_ne = len(non_edges)
    
    for i1 in range(len_ne):
        a, b = non_edges[i1]
        common_ab = adj_masks[a] & adj_masks[b]
        common_ab &= ~(1 << a)
        common_ab &= ~(1 << b)
        
        for i2 in range(i1 + 1, len_ne):
            c, d = non_edges[i2]
            if (1 << c) & ((1 << a) | (1 << b)) or (1 << d) & ((1 << a) | (1 << b)):
                continue
            if not ((common_ab >> c) & 1) or not ((common_ab >> d) & 1):
                continue
            
            common_all = common_ab & adj_masks[c] & adj_masks[d]
            common_all &= ~(1 << c)
            common_all &= ~(1 << d)
            
            candidates = []
            mask = common_all
            while mask:
                low_bit = mask & (-mask)
                v = low_bit.bit_length() - 1
                candidates.append(v)
                mask ^= low_bit
            
            if len(candidates) < 2:
                continue
            
            for k1 in range(len(candidates)):
                e = candidates[k1]
                for k2 in range(k1 + 1, len(candidates)):
                    f = candidates[k2]
                    if not ((adj_masks[e] >> f) & 1):
                        count += 1
    
    return count

def _is_valid_bitmask(adj_masks, n):
    """Check both color classes are K2,2,2-free."""
    if _has_k222_bitmask(adj_masks, n):
        return False
    # Complement: non-edges become edges
    comp_masks = []
    for i in range(n):
        comp = 0
        for j in range(n):
            if i != j and not ((adj_masks[i] >> j) & 1):
                comp |= (1 << j)
        comp_masks.append(comp)
    if _has_k222_bitmask(comp_masks, n):
        return False
    return True

def _local_search_bitmask(adj_masks, n, deadline, max_iters=50000):
    """
    Local search: flip edges to minimize total K2,2,2 violations in both colors.
    adj_masks: list of n integers, bitmask for each vertex.
    """
    adj_masks = adj_masks[:]  # copy
    full_mask = (1 << n) - 1
    
    def get_comp_masks(masks):
        comp = []
        for i in range(n):
            c = 0
            for j in range(n):
                if i != j and not ((masks[i] >> j) & 1):
                    c |= (1 << j)
            comp.append(c)
        return comp
    
    def total_violations(masks):
        v0 = _count_k222_bitmask(masks, n)
        comp = get_comp_masks(masks)
        v1 = _count_k222_bitmask(comp, n)
        return v0 + v1
    
    current_viol = total_violations(adj_masks)
    if current_viol == 0:
        return adj_masks
    
    no_improve = 0
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Flip edge (i,j)
        adj_masks[i] ^= (1 << j)
        adj_masks[j] ^= (1 << i)
        
        new_viol = total_violations(adj_masks)
        
        if new_viol <= current_viol:
            current_viol = new_viol
            no_improve = 0
            if current_viol == 0:
                break
        else:
            # Revert
            adj_masks[i] ^= (1 << j)
            adj_masks[j] ^= (1 << i)
            no_improve += 1
            
            if no_improve > 800:
                # Perturb
                for _ in range(random.randint(3, 6)):
                    pi = random.randint(0, n - 1)
                    pj = random.randint(0, n - 1)
                    if pi != pj:
                        adj_masks[pi] ^= (1 << pj)
                        adj_masks[pj] ^= (1 << pi)
                current_viol = total_violations(adj_masks)
                no_improve = 0
    
    return adj_masks

def _paley_mask(p):
    """Paley coloring as bitmask list. Returns None if p not valid."""
    if p % 4 != 1:
        return None
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    masks = []
    for i in range(p):
        m = 0
        for j in range(p):
            if i == j:
                continue
            diff = (j - i) % p
            if diff in squares:
                m |= (1 << j)
        masks.append(m)
    return masks

def _masks_to_colors(masks, n):
    """Convert bitmask list to color list (color 0 = edge in graph 0)."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (masks[i] >> j) & 1 else 1)
    return colors

def _colors_to_masks(colors, n):
    """Convert color list to bitmask list."""
    masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
            idx += 1
    return masks

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    best_n = 1
    best_colors = []
    
    # Try Paley colorings
    start_masks = None
    for p in [29, 17, 13, 5]:
        if p % 4 != 1:
            continue
        masks = _paley_mask(p)
        if masks is None:
            continue
        if _is_valid_bitmask(masks, p):
            start_masks = masks
            best_n = p
            best_colors = _masks_to_colors(masks, p)
            break
    
    if start_masks is None:
        # Fallback: random small
        for n_try in [5, 6, 7]:
            for attempt in range(30):
                masks = [0] * n_try
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if random.random() < 0.5:
                            masks[i] |= (1 << j)
                            masks[j] |= (1 << i)
                if _is_valid_bitmask(masks, n_try):
                    start_masks = masks
                    best_n = n_try
                    best_colors = _masks_to_colors(masks, n_try)
                    break
            if start_masks is not None:
                break
    
    if start_masks is None:
        return {'n': 1, 'colors': []}
    
    # Extend from best_n to 30
    current_masks = start_masks[:]
    current_n = best_n
    
    for target_n in range(current_n + 1, 31):
        if time.time() > deadline:
            break
        
        n = target_n
        found = False
        
        # Multiple restarts
        num_restarts = 8
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize: keep old, randomize new
            new_masks = [0] * n
            for i in range(current_n):
                new_masks[i] = current_masks[i]
            # Add new vertices
            for i in range(current_n, n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        new_masks[i] |= (1 << j)
                        new_masks[j] |= (1 << i)
            for i in range(current_n):
                for j in range(current_n, n):
                    if random.random() < 0.5:
                        new_masks[i] |= (1 << j)
                        new_masks[j] |= (1 << i)
            
            # Local search
            fixed = _local_search_bitmask(new_masks, n, deadline, max_iters=30000)
            
            if _is_valid_bitmask(fixed, n):
                found = True
                current_masks = fixed
                current_n = n
                best_n = n
                best_colors = _masks_to_colors(fixed, n)
                break
        
        if not found:
            # Extra attempts
            for extra in range(4):
                if time.time() > deadline:
                    break
                
                new_masks = [0] * n
                for i in range(current_n):
                    new_masks[i] = current_masks[i]
                for i in range(current_n, n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            new_masks[i] |= (1 << j)
                            new_masks[j] |= (1 << i)
                for i in range(current_n):
                    for j in range(current_n, n):
                        if random.random() < 0.5:
                            new_masks[i] |= (1 << j)
                            new_masks[j] |= (1 << i)
                
                fixed = _local_search_bitmask(new_masks, n, deadline, max_iters=20000)
                
                if _is_valid_bitmask(fixed, n):
                    found = True
                    current_masks = fixed
                    current_n = n
                    best_n = n
                    best_colors = _masks_to_colors(fixed, n)
                    break
        
        if not found:
            break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
