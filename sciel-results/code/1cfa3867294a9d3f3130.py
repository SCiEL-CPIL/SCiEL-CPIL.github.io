"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_mask = [0] * n  # bitmask of neighbors
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return n, adj, adj_mask

def _is_clique_mask(mask, adj_mask):
    """Check if vertices in mask form a clique using bitmasks."""
    # For each vertex in mask, all other vertices in mask must be neighbors
    remaining = mask
    while remaining:
        lsb = remaining & (-remaining)
        v = lsb.bit_length() - 1
        remaining ^= lsb
        if remaining & ~adj_mask[v]:
            return False
    return True

def _is_independent_mask(mask, adj_mask):
    """Check if vertices in mask form an independent set using bitmasks."""
    remaining = mask
    while remaining:
        lsb = remaining & (-remaining)
        v = lsb.bit_length() - 1
        remaining ^= lsb
        if remaining & adj_mask[v]:
            return False
    return True

def _popcount(x):
    return bin(x).count('1')

def _greedy_partition(n, adj, adj_mask):
    """Greedy: assign each vertex to first compatible part, else new part."""
    parts = []
    part_masks = []  # bitmask of vertices in each part
    part_types = []  # 0 = clique, 1 = indep
    
    for v in range(n):
        placed = False
        vbit = 1 << v
        for idx in range(len(parts)):
            ptype = part_types[idx]
            if ptype == 0:  # clique
                # v must be adjacent to all in part
                if (part_masks[idx] & ~adj_mask[v]) == 0:
                    parts[idx].append(v)
                    part_masks[idx] |= vbit
                    placed = True
                    break
            else:  # indep
                # v must not be adjacent to any in part
                if (part_masks[idx] & adj_mask[v]) == 0:
                    parts[idx].append(v)
                    part_masks[idx] |= vbit
                    placed = True
                    break
        if not placed:
            parts.append([v])
            part_masks.append(vbit)
            part_types.append(0)  # start as clique
    return parts

def _cochromatic_search(n, adj_mask, deadline):
    """
    Search for minimum cochromatic number.
    Uses recursive backtracking with bitmask state.
    State: (remaining_mask, num_parts_used, part_masks_tuple, part_types_tuple)
    We try to assign vertices one by one to existing parts or create new parts.
    """
    best = [n + 1]
    best_partition = None
    start = time.time()
    
    # Vertex ordering: try vertices with high degree first for better pruning
    degrees = [_popcount(adj_mask[v]) for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -degrees[v])
    
    # Precompute: for each vertex, which subsets of remaining can join it as clique/indep
    # This is expensive, so we'll compute on the fly
    
    def search(remaining_mask, parts_masks, parts_types):
        nonlocal best, best_partition
        if time.time() - start > deadline:
            return False
        
        num_parts = len(parts_masks)
        
        if remaining_mask == 0:
            if num_parts < best[0]:
                best[0] = num_parts
                # Build partition
                partition = []
                for i in range(num_parts):
                    verts = []
                    m = parts_masks[i]
                    while m:
                        lsb = m & (-m)
                        verts.append(lsb.bit_length() - 1)
                        m ^= lsb
                    partition.append(verts)
                best_partition = partition
            return True
        
        if num_parts >= best[0]:
            return False
        
        # Pick next vertex: the one with highest degree in remaining
        # For better pruning, pick the vertex that has the most constraints
        # Try the first unassigned vertex in our ordering
        v = None
        for candidate in vertex_order:
            if remaining_mask & (1 << candidate):
                v = candidate
                break
        
        if v is None:
            return False
        
        vbit = 1 << v
        new_remaining = remaining_mask & ~vbit
        
        # Option 1: Place v in an existing part
        for i in range(len(parts_masks)):
            ptype = parts_types[i]
            if ptype == 0:  # clique
                # v must be adjacent to all in part i
                if (parts_masks[i] & ~adj_mask[v]) == 0:
                    parts_masks[i] |= vbit
                    if search(new_remaining, parts_masks, parts_types):
                        return True
                    parts_masks[i] ^= vbit
            else:  # indep
                # v must not be adjacent to any in part i
                if (parts_masks[i] & adj_mask[v]) == 0:
                    parts_masks[i] |= vbit
                    if search(new_remaining, parts_masks, parts_types):
                        return True
                    parts_masks[i] ^= vbit
        
        # Option 2: Create a new part with v
        if len(parts_masks) + 1 < best[0]:
            # Try as clique
            new_parts_masks = list(parts_masks) + [vbit]
            new_parts_types = list(parts_types) + [0]
            if search(new_remaining, new_parts_masks, new_parts_types):
                return True
            
            # Try as independent set
            new_parts_masks = list(parts_masks) + [vbit]
            new_parts_types = list(parts_types) + [1]
            if search(new_remaining, new_parts_masks, new_parts_types):
                return True
        
        return False
    
    full_mask = (1 << n) - 1
    search(full_mask, [], [])
    
    return best_partition if best_partition else None

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s limit
    
    n, adj, adj_mask = _build_graph()
    
    # Get greedy solution first
    greedy_parts = _greedy_partition(n, adj, adj_mask)
    greedy_count = len(greedy_parts)
    
    # Try exact search
    result = _cochromatic_search(n, adj_mask, deadline)
    
    if result is not None and len(result) < greedy_count:
        best_parts = result
    else:
        best_parts = greedy_parts
    
    # Ensure proper types: list of lists of Python ints
    final_parts = []
    for part in best_parts:
        final_parts.append([int(v) for v in part])
    
    return final_parts

# EVOLVE-BLOCK-END
