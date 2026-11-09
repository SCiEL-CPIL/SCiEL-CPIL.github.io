"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 55.0  # safety margin before 60s limit
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute which subsets are cliques or independent sets
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_valid = [False] * (1 << n)
    
    is_clique[0] = True
    is_independent[0] = True
    is_valid[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        
        if is_independent[rest] and (rest & adj[v]) == 0:
            is_independent[mask] = True
        
        is_valid[mask] = is_clique[mask] or is_independent[mask]
    
    # Backtracking search with branch-and-bound
    # We try to assign vertices to existing parts or create new parts
    # Each part is either a clique or an independent set
    
    # State: list of parts, where each part is (mask, is_clique_flag)
    # is_clique_flag: None = undetermined, True = must be clique, False = must be independent
    
    best_partition = None
    best_count = n + 1  # worst case is n parts (each vertex alone)
    
    # Order vertices by degree (descending) for better pruning
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    vertex_order = sorted(range(n), key=lambda x: -degrees[x])
    
    # For each vertex, precompute which existing part types it can join
    # A vertex v can join a clique part S if v is adjacent to all vertices in S
    # A vertex v can join an independent set part S if v is NOT adjacent to any vertex in S
    
    def can_join_clique(v, part_mask):
        # v must be adjacent to all vertices in part_mask
        return (part_mask & ~adj[v]) == 0
    
    def can_join_independent(v, part_mask):
        # v must NOT be adjacent to any vertex in part_mask
        return (part_mask & adj[v]) == 0
    
    # Backtracking: assign vertices in vertex_order
    # parts: list of (mask, type) where type is None, True, or False
    
    def backtrack(idx, parts, count):
        nonlocal best_partition, best_count
        
        if time.time() > deadline:
            return False
        
        if count >= best_count:
            return False
        
        if idx == n:
            # Found a valid partition
            if count < best_count:
                best_count = count
                # Reconstruct partition from parts
                partition = []
                for mask, ptype in parts:
                    part = [i for i in range(n) if (mask >> i) & 1]
                    partition.append(part)
                best_partition = partition
            return True
        
        v = vertex_order[idx]
        
        # Try adding v to each existing part
        for i, (pmask, ptype) in enumerate(parts):
            if ptype is None:
                # Part is undetermined - check both clique and independent
                # Check if adding v keeps it as clique
                if can_join_clique(v, pmask):
                    # Try making it a clique
                    new_parts = parts.copy()
                    new_parts[i] = (pmask | (1 << v), True)
                    if backtrack(idx + 1, new_parts, count):
                        return True
                # Check if adding v keeps it as independent set
                if can_join_independent(v, pmask):
                    # Try making it an independent set
                    new_parts = parts.copy()
                    new_parts[i] = (pmask | (1 << v), False)
                    if backtrack(idx + 1, new_parts, count):
                        return True
            elif ptype is True:
                # Part is a clique
                if can_join_clique(v, pmask):
                    new_parts = parts.copy()
                    new_parts[i] = (pmask | (1 << v), True)
                    if backtrack(idx + 1, new_parts, count):
                        return True
            else:
                # Part is an independent set
                if can_join_independent(v, pmask):
                    new_parts = parts.copy()
                    new_parts[i] = (pmask | (1 << v), False)
                    if backtrack(idx + 1, new_parts, count):
                        return True
        
        # Try creating a new part with just v
        if count + 1 < best_count:
            new_parts = parts + [(1 << v, None)]
            if backtrack(idx + 1, new_parts, count + 1):
                return True
        
        return False
    
    # Start backtracking
    backtrack(0, [], 0)
    
    if best_partition is not None:
        # Verify the partition
        seen = [False] * n
        for part in best_partition:
            if not part:
                return [[i] for i in range(n)]
            for v in part:
                if seen[v]:
                    return [[i] for i in range(n)]
                seen[v] = True
            mask_part = 0
            for v in part:
                mask_part |= (1 << v)
            if not is_valid[mask_part]:
                return [[i] for i in range(n)]
        
        for i in range(n):
            if not seen[i]:
                return [[i] for i in range(n)]
        
        result = [list(part) for part in best_partition]
        return result
    
    # Fallback: each vertex in its own part
    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
