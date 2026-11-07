"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        u, v = e
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u in range(n):
        for v in range(n):
            if (u, v) in edge_set:
                adj[u] |= (1 << v)
    
    # Complement adjacency for independent sets
    comp_adj = [0] * n
    for u in range(n):
        for v in range(n):
            if u != v and (u, v) not in edge_set:
                comp_adj[u] |= (1 << v)

    maximal_sets = []

    def bron_kerbosch(R, P_mask, X_mask, adj_mask):
        """
        Bron-Kerbosch with bitmasks.
        R: bitmask of current clique
        P_mask: bitmask of candidates
        X_mask: bitmask of excluded candidates
        adj_mask: list of adjacency bitmasks
        """
        if P_mask == 0 and X_mask == 0:
            if R != 0:
                maximal_sets.append(R)
            return
        
        # Pivot selection: pick a vertex in P_mask
        # We choose the vertex in P_mask with the highest degree within P_mask to minimize candidates
        if P_mask:
            # Simple pivot: first vertex in P_mask
            # Better: vertex with max degree in P_mask
            # For n=14, even a simple pivot is fast, but let's try to be smart
            # We'll just pick the first one for simplicity and speed of pivot selection
            pivot_bit = P_mask & -P_mask
            pivot = pivot_bit.bit_length() - 1
            
            # Candidates are vertices in P_mask that are NOT adjacent to pivot
            neighbors_of_pivot_in_P = P_mask & adj_mask[pivot]
            candidates = P_mask & ~neighbors_of_pivot_in_P
        else:
            candidates = 0
            
        while candidates:
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            
            # Recurse
            # New R: R | v
            # New P: P & N(v)
            # New X: X & N(v)
            bron_kerbosch(R | v_bit, P_mask & adj_mask[v], X_mask & adj_mask[v], adj_mask)
            
            # Update P and X
            P_mask &= ~v_bit
            X_mask |= v_bit
            candidates &= ~v_bit

    # Find maximal cliques
    bron_kerbosch(0, (1 << n) - 1, 0, adj)
    
    # Find maximal independent sets (maximal cliques in complement)
    bron_kerbosch(0, (1 << n) - 1, 0, comp_adj)

    # Remove duplicates
    unique_masks = list(set(maximal_sets))
    unique_masks = [m for m in unique_masks if m != 0]
    
    # Sort by size descending
    unique_masks.sort(key=lambda x: x.bit_count(), reverse=True)
    
    # Precompute vertices for each mask
    mask_to_vertices = {}
    for m in unique_masks:
        verts = []
        mm = m
        while mm:
            bit = mm & -mm
            v = bit.bit_length() - 1
            verts.append(v)
            mm &= ~bit
        mask_to_vertices[m] = verts

    all_vertices_mask = (1 << n) - 1
    best_partition = []
    best_count = n + 1
    
    # Precompute which masks contain each vertex
    vertex_to_masks = [[] for _ in range(n)]
    for m in unique_masks:
        mm = m
        while mm:
            bit = mm & -mm
            v = bit.bit_length() - 1
            vertex_to_masks[v].append(m)
            mm &= ~bit

    def solve(remaining_mask, current_count, current_partition):
        nonlocal best_partition, best_count
        
        if current_count >= best_count:
            return
        
        if remaining_mask == 0:
            if current_count < best_count:
                best_count = current_count
                best_partition = current_partition[:]
            return
        
        popcount = remaining_mask.bit_count()
        
        # Lower bound calculation
        max_fit_size = 0
        for m in unique_masks:
            if (m & remaining_mask) == m:
                sz = m.bit_count()
                if sz > max_fit_size:
                    max_fit_size = sz
                if m == remaining_mask:
                    if current_count + 1 < best_count:
                        best_count = current_count + 1
                        best_partition = current_partition + [m]
                    return
        
        if max_fit_size == 0:
            return
            
        lb = (popcount + max_fit_size - 1) // max_fit_size
        if current_count + lb >= best_count:
            return
            
        # Heuristic: Pick a vertex from remaining_mask with fewest valid covering masks
        min_options_count = float('inf')
        target_vertex = -1
        mm = remaining_mask
        while mm:
            bit = mm & -mm
            v = bit.bit_length() - 1
            count = 0
            for m in vertex_to_masks[v]:
                if (m & remaining_mask) == m:
                    count += 1
                    if count >= min_options_count:
                        break
            if count < min_options_count:
                min_options_count = count
                target_vertex = v
                if min_options_count == 1:
                    break
            mm &= ~bit
            
        if target_vertex == -1:
            return
            
        candidate_masks = []
        for m in vertex_to_masks[target_vertex]:
            if (m & remaining_mask) == m:
                candidate_masks.append(m)
                
        candidate_masks.sort(key=lambda x: x.bit_count(), reverse=True)
        
        for m in candidate_masks:
            new_remaining = remaining_mask & ~m
            current_partition.append(m)
            solve(new_remaining, current_count + 1, current_partition)
            current_partition.pop()
            
            if best_count == 1:
                return

    solve(all_vertices_mask, 0, [])
    
    if not best_partition:
        return [[v] for v in range(n)]
        
    result = []
    for m in best_partition:
        result.append(sorted(mask_to_vertices[m]))
        
    return result

# EVOLVE-BLOCK-END
