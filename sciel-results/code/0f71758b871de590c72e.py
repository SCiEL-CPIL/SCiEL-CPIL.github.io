# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _find_k222(masks, n):
    """
    Find a K2,2,2 in the graph defined by masks.
    Returns list of 6 vertices or None.
    """
    if n < 6:
        return None
        
    all_ones = (1 << n) - 1
    non_edges = [0] * n
    for i in range(n):
        non_edges[i] = (all_ones ^ masks[i]) & ~(1 << i)
        
    ne_list = []
    for i in range(n):
        m = non_edges[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                ne_list.append((i, j))
            m ^= lsb
            
    if len(ne_list) < 3:
        return None

    len_ne = len(ne_list)
    
    # Iterate over pairs of non-edges
    for i1 in range(len_ne):
        a, b = ne_list[i1]
        # Common neighbors of a and b in G (edges)
        common_ab = masks[a] & masks[b]
        common_ab &= ~(1 << a) & ~(1 << b)
        
        for i2 in range(i1 + 1, len_ne):
            c, d = ne_list[i2]
            # Disjoint check
            if c == a or c == b or d == a or d == b:
                continue
            
            # c and d must be in common_ab
            if not ((common_ab >> c) & 1) or not ((common_ab >> d) & 1):
                continue
            
            # Candidates for e, f must be in common_ab AND common_cd
            common_cd = masks[c] & masks[d]
            candidates = common_ab & common_cd
            candidates &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
            
            if candidates == 0:
                continue
            
            # Check if there is a non-edge (e, f) within candidates
            m = candidates
            while m:
                lsb = m & -m
                e = lsb.bit_length() - 1
                # Check if any other vertex f in candidates is a non-neighbor of e
                # non_edges[e] & (candidates ^ (1<<e))
                rest = candidates ^ lsb
                if non_edges[e] & rest:
                    lsb2 = (non_edges[e] & rest) & -(non_edges[e] & rest)
                    f = lsb2.bit_length() - 1
                    return [a, b, c, d, e, f]
                m ^= lsb
                
    return None

def _is_valid(masks, n):
    """Check if graph is K2,2,2 free."""
    return _find_k222(masks, n) is None

def _paley_mask(n):
    """
    Generate the Paley graph mask for n = q (prime power).
    """
    if n < 2:
        return [0] * n
    
    residues = set()
    for i in range(1, n):
        residues.add((i * i) % n)
        
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i == j:
                continue
            diff = (i - j) % n
            if diff in residues:
                mask |= (1 << j)
        masks[i] = mask
    return masks

def _local_search_repair(masks, n, time_limit):
    """
    Attempt to remove K2,2,2s by flipping edges involved in them.
    """
    start = time.time()
    current_masks = masks[:]
    
    # We need to check both colors.
    # Color 0 is the graph defined by masks.
    # Color 1 is the complement.
    
    def complement_masks(masks):
        all_ones = (1 << n) - 1
        comp = [0] * n
        for i in range(n):
            comp[i] = (all_ones ^ masks[i]) & ~(1 << i)
        return comp

    max_steps = 10000
    for step in range(max_steps):
        if time.time() > time_limit:
            break
            
        # Check color 0
        witness = _find_k222(current_masks, n)
        if witness is None:
            # Check color 1
            comp_masks = complement_masks(current_masks)
            witness = _find_k222(comp_masks, n)
            if witness is None:
                return True, current_masks
            else:
                # Witness is in color 1, which means it's a non-edge structure in color 0?
                # No, witness is a K2,2,2 in the complement.
                # This means in current_masks, these 6 vertices form the complement of K2,2,2, i.e., 3 disjoint edges.
                # To break the K2,2,2 in color 1, we need to add an edge to color 1 (remove from color 0) 
                # or remove an edge from color 1 (add to color 0)?
                # If we have K2,2,2 in G1, we want to destroy it.
                # G1 edges are non-edges in G0.
                # So we need to flip an edge that is currently 1 in G1 (0 in G0) to 0 in G1 (1 in G0)?
                # Or flip an edge that is 0 in G1 (1 in G0) to 1 in G1 (0 in G0)?
                # Flipping any edge in the induced subgraph of the witness will break the K2,2,2.
                # Let's pick a random pair from the witness and flip it.
                u, v = random.sample(witness, 2)
                bit = 1 << v
                if current_masks[u] & bit:
                    current_masks[u] &= ~bit
                    current_masks[v] &= ~(1 << u)
                else:
                    current_masks[u] |= bit
                    current_masks[v] |= (1 << u)
                continue
        
        # Witness found in color 0
        u, v = random.sample(witness, 2)
        bit = 1 << v
        if current_masks[u] & bit:
            current_masks[u] &= ~bit
            current_masks[v] &= ~(1 << u)
        else:
            current_masks[u] |= bit
            current_masks[v] |= (1 << u)
            
    return False, current_masks

def _masks_to_colors(masks, n):
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(1 if (masks[i] >> j) & 1 else 0)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    best_n = 1
    best_colors = [0]
    
    # Try n=29, 28, ... down to 6
    for target_n in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() > deadline:
            break
            
        n = target_n
        found = False
        
        # 1. Try Paley construction if n is prime
        def is_prime(num):
            if num < 2: return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True
            
        if is_prime(n):
            masks = _paley_mask(n)
            # Run local search repair
            success, masks = _local_search_repair(masks, n, deadline)
            if success:
                found = True
                best_n = n
                best_colors = _masks_to_colors(masks, n)
                break
            
        # 2. Try random restarts
        if not found:
             for restart in range(50):
                if time.time() > deadline: break
                masks = [0] * n
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < 0.5:
                            masks[i] |= (1 << j)
                            masks[j] |= (1 << i)
                            
                success, masks = _local_search_repair(masks, n, deadline)
                if success:
                    found = True
                    best_n = n
                    best_colors = _masks_to_colors(masks, n)
                    break

        if found:
            break

    # If we didn't find anything (unlikely), return a small valid graph
    if best_n == 1:
        best_n = 5
        best_colors = [0] * (5*4//2)
        
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
