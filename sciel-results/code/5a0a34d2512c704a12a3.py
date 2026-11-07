"""Rewritten program with branch-and-bound for minimum cochromatic partition."""
import time
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    deadline = time.time() + 55.0  # 55s safety margin
    
    # Precompute: for each subset (as bitmask), is it a clique? is it an independent set?
    # We'll compute these on-the-fly with memoization
    
    # Helper: check if a bitmask represents a clique
    def is_clique(mask):
        if mask == 0:
            return True
        # Get vertices
        v = (mask & -mask).bit_length() - 1
        rest = mask & (mask - 1)
        if rest == 0:
            return True
        # v must be adjacent to all in rest
        return (adj[v] & rest) == rest
    
    # Helper: check if a bitmask represents an independent set
    def is_independent(mask):
        if mask == 0:
            return True
        v = (mask & -mask).bit_length() - 1
        rest = mask & (mask - 1)
        if rest == 0:
            return True
        # v must NOT be adjacent to any in rest
        return (adj[v] & rest) == 0
    
    # Helper: check if a bitmask is a valid part (clique or independent set)
    def is_valid_part(mask):
        return is_clique(mask) or is_independent(mask)
    
    # Precompute validity for all subsets (2^14 = 16384 subsets)
    valid_part = [False] * (1 << n)
    valid_part[0] = True
    for mask in range(1, 1 << n):
        # Check if valid by checking if it's a clique or independent set
        # For efficiency, check incrementally
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        if rest == 0:
            valid_part[mask] = True
        else:
            # Check clique: v adjacent to all in rest
            clique_ok = (adj[v] & rest) == rest
            # Check independent: v not adjacent to any in rest
            indep_ok = (adj[v] & rest) == 0
            # But rest itself must also be valid... actually we need the whole set to be clique or independent
            # Let's just compute directly
            # A set is a clique iff every pair is adjacent
            # A set is independent iff no pair is adjacent
            # Check if mask is a clique
            is_c = True
            m = mask
            while m:
                lsb2 = m & -m
                v2 = lsb2.bit_length() - 1
                m ^= lsb2
                if (adj[v2] & m) != m:
                    is_c = False
                    break
            if is_c:
                valid_part[mask] = True
            else:
                # Check if mask is independent
                is_i = True
                m = mask
                while m:
                    lsb2 = m & -m
                    v2 = lsb2.bit_length() - 1
                    m ^= lsb2
                    if (adj[v2] & m) != 0:
                        is_i = False
                        break
                if is_i:
                    valid_part[mask] = True
    
    # Now use branch and bound to find minimum number of parts
    # State: remaining vertices mask, current number of parts
    # We want to minimize parts
    
    best = [n]  # worst case: each vertex its own part
    
    def solve(remaining, parts_used):
        if time.time() > deadline:
            return
        if remaining == 0:
            if parts_used < best[0]:
                best[0] = parts_used
            return
        if parts_used >= best[0]:
            return
        
        # Lower bound: at least 1 more part needed
        if parts_used + 1 >= best[0]:
            return
        
        # Try to find a valid part from remaining vertices
        # Use heuristic: try larger valid parts first to reduce remaining faster
        # Get list of vertices in remaining
        rem_list = []
        m = remaining
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            rem_list.append(v)
            m ^= lsb
        
        # Try all subsets of remaining that are valid parts
        # To avoid duplicates, use canonical ordering: the smallest vertex in the part must be included
        # and we only consider subsets where the min vertex is the first element
        min_v = rem_list[0]
        
        # Generate subsets containing min_v from remaining
        # For each such subset, check if valid, then recurse
        # To keep it manageable, limit subset size
        # Actually with 14 vertices, let's be smart
        
        # Try all valid subsets of remaining that contain min_v
        # We'll enumerate subsets by building them
        remaining_mask = remaining
        # Get bits of remaining
        bits = []
        m = remaining_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            bits.append(v)
            m ^= lsb
        
        # Try subsets containing bits[0] (the smallest)
        # Enumerate all subsets of bits[1:] and add bits[0]
        rest_bits = bits[1:]
        num_rest = len(rest_bits)
        
        # Try different sizes of parts, from larger to smaller (to prune better)
        # Actually, let's try all valid subsets
        for size in range(num_rest, -1, -1):
            if time.time() > deadline:
                return
            # Generate all combinations of size from rest_bits
            def gen_combos(idx, chosen):
                if time.time() > deadline:
                    return
                if len(chosen) == size:
                    part_mask = (1 << bits[0])
                    for v in chosen:
                        part_mask |= (1 << v)
                    if valid_part[part_mask]:
                        new_remaining = remaining_mask & ~part_mask
                        solve(new_remaining, parts_used + 1)
                    return
                if idx >= num_rest:
                    return
                # Include rest_bits[idx]
                chosen.append(rest_bits[idx])
                gen_combos(idx + 1, chosen)
                chosen.pop()
                # Exclude rest_bits[idx]
                gen_combos(idx + 1, chosen)
            
            gen_combos(0, [])
    
    full_mask = (1 << n) - 1
    solve(full_mask, 0)
    
    # Now reconstruct the actual partition with best[0] parts
    # We need to find a valid partition with exactly best[0] parts
    result_parts = []
    
    def reconstruct(remaining, parts_left):
        if time.time() > deadline:
            return False
        if remaining == 0:
            return parts_left == 0
        if parts_left == 0:
            return remaining == 0
        
        # Find a valid part from remaining
        bits = []
        m = remaining
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            bits.append(v)
            m ^= lsb
        
        min_v = bits[0]
        rest_bits = bits[1:]
        num_rest = len(rest_bits)
        
        # Try all valid subsets containing min_v
        for size in range(num_rest, -1, -1):
            if time.time() > deadline:
                return False
            
            def gen_and_try(idx, chosen):
                if time.time() > deadline:
                    return False
                if len(chosen) == size:
                    part_mask = (1 << min_v)
                    for v in chosen:
                        part_mask |= (1 << v)
                    if valid_part[part_mask]:
                        new_remaining = remaining & ~part_mask
                        # Record this part
                        part_vertices = [v for v in range(n) if (part_mask >> v) & 1]
                        result_parts.append(part_vertices)
                        if reconstruct(new_remaining, parts_left - 1):
                            return True
                        result_parts.pop()
                    return False
                if idx >= num_rest:
                    return False
                chosen.append(rest_bits[idx])
                if gen_and_try(idx + 1, chosen):
                    return True
                chosen.pop()
                return gen_and_try(idx + 1, chosen)
            
            if gen_and_try(0, []):
                return True
        
        return False
    
    reconstruct(full_mask, best[0])
    
    # Convert to proper types
    final_parts = []
    for part in result_parts:
        final_parts.append([int(v) for v in part])
    
    return final_parts

# EVOLVE-BLOCK-END
