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
    
    # Complement adjacency (for independent sets = cliques in complement)
    comp_adj = [0] * n
    for u in range(n):
        for v in range(n):
            if u != v and (u, v) not in edge_set:
                comp_adj[u] |= (1 << v)
    
    # Generate all maximal cliques in G (for clique parts)
    # and all maximal cliques in complement G (for independent set parts)
    all_cliques = []
    
    def bron_kerbosch(R, P_mask, X_mask, adj_list):
        """Bron-Kerbosch with pivot to find all maximal cliques."""
        if P_mask == 0 and X_mask == 0:
            if R:
                all_cliques.append(frozenset(R))
            return
        
        # Choose pivot: pick vertex from P_mask with max degree in P_mask
        if P_mask:
            pivot = -1
            max_deg = -1
            pm = P_mask
            while pm:
                lsb = pm & (-pm)
                v = lsb.bit_length() - 1
                deg = bin(adj_list[v] & P_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
                pm &= pm - 1
            # Candidates are vertices in P_mask not adjacent to pivot
            candidates_mask = P_mask & ~adj_list[pivot]
        else:
            candidates_mask = 0
        
        cm = candidates_mask
        while cm:
            lsb = cm & (-cm)
            v = lsb.bit_length() - 1
            bron_kerbosch(R | {v}, P_mask & adj_list[v], X_mask & adj_list[v], adj_list)
            P_mask &= ~lsb
            X_mask |= lsb
            cm &= cm - 1
    
    # Find maximal cliques in G
    bron_kerbosch(set(), (1 << n) - 1, 0, adj)
    cliques = list(all_cliques)
    all_cliques.clear()
    
    # Find maximal cliques in complement (= maximal independent sets in G)
    bron_kerbosch(set(), (1 << n) - 1, 0, comp_adj)
    independent_sets = list(all_cliques)
    
    # Combine: all homogeneous sets are maximal cliques + maximal independent sets
    # Remove duplicates (a set can be both a clique and independent set, e.g., single vertices)
    homogeneous = []
    seen = set()
    for s in cliques:
        if s not in seen:
            homogeneous.append(s)
            seen.add(s)
    for s in independent_sets:
        if s not in seen:
            homogeneous.append(s)
            seen.add(s)
    
    # Convert to bitmasks and sort by size descending
    set_info = []  # (mask, vertices_list)
    for s in homogeneous:
        mask = 0
        for v in s:
            mask |= (1 << v)
        set_info.append((mask, sorted(s)))
    
    # Sort by size descending, then by mask for determinism
    set_info.sort(key=lambda x: (-bin(x[0]).count('1'), x[0]))
    
    set_masks = [info[0] for info in set_info]
    set_verts = [info[1] for info in set_info]
    
    # For each set, precompute which other sets it's disjoint from
    # Actually, we'll check on the fly for efficiency
    
    all_vertices_mask = (1 << n) - 1
    best_count = n + 1
    best_partition = None
    
    # Precompute popcount for all masks up to 2^n
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # For lower bound: for each remaining mask, what's the max size of a homogeneous set that fits?
    # Precompute: for each mask, the max size of any set_info[i] that is subset of mask
    max_set_size_in = [0] * (1 << n)
    for mask in range(1 << n):
        max_sz = 0
        for sm in set_masks:
            if sm & mask == sm:
                sz = popcount[sm]
                if sz > max_sz:
                    max_sz = sz
        max_set_size_in[mask] = max_sz
    
    def solve(remaining_mask, current_parts, current_count):
        nonlocal best_count, best_partition
        
        if current_count >= best_count:
            return
        
        if remaining_mask == 0:
            if current_count < best_count:
                best_count = current_count
                best_partition = [list(p) for p in current_parts]
            return
        
        # Lower bound: ceil(popcount(remaining) / max_set_size_in[remaining])
        pc = popcount[remaining_mask]
        max_sz = max_set_size_in[remaining_mask]
        if max_sz == 0:
            return
        lb = (pc + max_sz - 1) // max_sz
        if current_count + lb >= best_count:
            return
        
        # Additional lower bound: chromatic number of the remaining graph
        # (each part is either a clique or independent set, so we need at least chi(G_remaining) parts
        #  if all parts were independent sets, but since some can be cliques, this is a weaker bound)
        # Actually, the cochromatic number is at least the chromatic number of the complement... 
        # Let's use a simpler bound: the clique number of the remaining graph gives a lower bound
        # on the number of independent sets needed, but we can also use clique parts.
        # For now, stick with the size-based bound.
        
        # Try each set that is a subset of remaining_mask
        # Order: try larger sets first (already sorted)
        for i, sm in enumerate(set_masks):
            if sm & remaining_mask == sm:
                new_remaining = remaining_mask & ~sm
                current_parts.append(set_verts[i])
                solve(new_remaining, current_parts, current_count + 1)
                current_parts.pop()
                
                if best_count == 1:
                    return
    
    solve(all_vertices_mask, [], 0)
    
    if best_partition is None:
        return [[v] for v in range(n)]
    
    return best_partition

# EVOLVE-BLOCK-END
