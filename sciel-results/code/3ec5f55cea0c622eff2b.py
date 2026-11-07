"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time
import sys

sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Build adjacency bitmasks for O(1) checks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Full mask of all vertices
    full_mask = (1 << n) - 1
    
    # Order vertices by degree (descending) for better pruning
    vertices = list(range(n))
    vertices.sort(key=lambda v: bin(adj_mask[v]).count('1'), reverse=True)
    
    def can_join(v, pm):
        """Check if vertex v can be added to part with mask pm (clique or independent)."""
        if pm == 0:
            return True
        # v can join as clique member iff v is adjacent to ALL vertices in pm
        # v can join as independent member iff v is NOT adjacent to ANY vertex in pm
        is_clique_join = (pm & ~adj_mask[v] & full_mask) == 0
        is_indep_join = (pm & adj_mask[v]) == 0
        return is_clique_join or is_indep_join
    
    def can_partition_with_k_parts(k):
        """Check if we can partition all vertices into k valid parts."""
        
        def backtrack(idx, part_masks):
            if time.time() > deadline:
                return None
            
            if idx == len(vertices):
                # All vertices assigned; check all parts are non-empty
                for pm in part_masks:
                    if pm == 0:
                        return None
                return list(part_masks)
            
            v = vertices[idx]
            v_bit = 1 << v
            
            # Try adding v to each existing non-empty part
            for i in range(k):
                if part_masks[i] != 0:
                    if can_join(v, part_masks[i]):
                        part_masks[i] |= v_bit
                        result = backtrack(idx + 1, part_masks)
                        if result is not None:
                            return result
                        part_masks[i] ^= v_bit
            
            # Try putting v in a new empty part (symmetry breaking: first empty only)
            for i in range(k):
                if part_masks[i] == 0:
                    part_masks[i] = v_bit
                    result = backtrack(idx + 1, part_masks)
                    if result is not None:
                        return result
                    part_masks[i] = 0
                    break  # Only try first empty part
            
            return None
        
        return backtrack(0, [0] * k)
    
    # Get a greedy upper bound first
    def greedy_partition():
        remaining = full_mask
        parts = []
        while remaining:
            best_sub = 0
            best_size = 0
            sub = remaining
            while sub:
                # Check if sub is a valid part (clique or independent)
                verts = [i for i in range(n) if sub & (1 << i)]
                is_c = True
                is_ind = True
                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        if (1 << verts[j]) & adj_mask[verts[i]]:
                            is_ind = False
                        else:
                            is_c = False
                        if not is_c and not is_ind:
                            break
                    if not is_c and not is_ind:
                        break
                if is_c or is_ind:
                    size = bin(sub).count('1')
                    if size > best_size:
                        best_size = size
                        best_sub = sub
                sub = (sub - 1) & remaining
            if best_sub == 0:
                lsb = remaining & (-remaining)
                best_sub = lsb
            parts.append(best_sub)
            remaining ^= best_sub
        return parts
    
    greedy_result = greedy_partition()
    best_count = len(greedy_result)
    best_partition = greedy_result
    
    # Iterative deepening: try smaller k values
    for k in range(1, best_count):
        if time.time() > deadline:
            break
        result = can_partition_with_k_parts(k)
        if result is not None:
            best_count = k
            best_partition = result
            break  # Found minimum
    
    # Convert bitmask partition to list of vertex lists
    result_parts = []
    for mask in best_partition:
        if mask:
            verts = [i for i in range(n) if mask & (1 << i)]
            if verts:
                result_parts.append([int(v) for v in verts])
    
    # Ensure we have a valid result
    if not result_parts:
        result_parts = [[i] for i in range(n)]
    
    return result_parts
# EVOLVE-BLOCK-END
