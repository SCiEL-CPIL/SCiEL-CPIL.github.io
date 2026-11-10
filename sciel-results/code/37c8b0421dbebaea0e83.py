"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return n, adj

def _is_clique_mask(mask, adj):
    """Check if the set of vertices in mask forms a clique using bitmasks."""
    while mask:
        v = (mask & -mask).bit_length() - 1
        mask ^= (1 << v)
        if mask & ~adj[v]:
            return False
    return True

def _is_indep_mask(mask, adj):
    """Check if the set of vertices in mask forms an independent set using bitmasks."""
    while mask:
        v = (mask & -mask).bit_length() - 1
        mask ^= (1 << v)
        if mask & adj[v]:
            return False
    return True

def _mask_to_list(mask):
    """Convert a bitmask to a list of vertex indices."""
    result = []
    while mask:
        v = (mask & -mask).bit_length() - 1
        result.append(v)
        mask ^= (1 << v)
    return result

def _greedy_partition(n, adj):
    """Greedy: assign each vertex to first compatible part, else new part."""
    parts = []
    part_types = []
    for v in range(n):
        placed = False
        for idx, part in enumerate(parts):
            ptype = part_types[idx]
            if ptype == 0:  # clique
                if all((1 << u) & adj[v] for u in part):
                    part.append(v)
                    placed = True
                    break
            else:  # indep
                if all(not ((1 << u) & adj[v]) for u in part):
                    part.append(v)
                    placed = True
                    break
        if not placed:
            parts.append([v])
            part_types.append(0)
    return parts, part_types

def _can_add_clique(part_mask, v, adj):
    """Check if v can be added to clique part_mask."""
    return (part_mask & ~adj[v]) == 0

def _can_add_indep(part_mask, v, adj):
    """Check if v can be added to independent set part_mask."""
    return (part_mask & adj[v]) == 0

def _try_k_parts(n, adj, k, deadline, vertex_order):
    """Try to partition into exactly k parts using bitmask representation."""
    # Enumerate type assignments: 2^k possibilities
    # For each, try to assign vertices in given order
    
    for type_mask in range(1 << k):
        if time.time() > deadline:
            return None
        
        # Determine part types from type_mask
        part_types = [(type_mask >> i) & 1 for i in range(k)]
        
        # Backtracking assignment
        part_masks = [0] * k
        assignment = [-1] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(vertex_order):
                return True
            
            v = vertex_order[idx]
            
            # Try placing v in each existing part
            for p in range(k):
                if part_types[p] == 0:  # clique
                    if (part_masks[p] & ~adj[v]) == 0:
                        part_masks[p] |= (1 << v)
                        assignment[v] = p
                        if backtrack(idx + 1):
                            return True
                        part_masks[p] ^= (1 << v)
                        assignment[v] = -1
                else:  # indep
                    if (part_masks[p] & adj[v]) == 0:
                        part_masks[p] |= (1 << v)
                        assignment[v] = p
                        if backtrack(idx + 1):
                            return True
                        part_masks[p] ^= (1 << v)
                        assignment[v] = -1
            
            return False
        
        if backtrack(0):
            # Build partition from assignment
            parts = [[] for _ in range(k)]
            for v in range(n):
                parts[assignment[v]].append(v)
            parts = [p for p in parts if p]
            return parts
    
    return None

def _compute_vertex_order(n, adj):
    """Compute vertex ordering by degree (descending) for better pruning."""
    degrees = [bin(adj[v]).count('1') for v in range(n)]
    order = list(range(n))
    order.sort(key=lambda v: -degrees[v])
    return order

def _lower_bound_clique(n, adj):
    """Compute clique number as lower bound."""
    best = 1
    for size in range(2, n + 1):
        found = False
        for combo in itertools.combinations(range(n), size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if _is_clique_mask(mask, adj):
                best = size
                found = True
        if not found:
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n, adj = _build_graph()
    
    # Get greedy solution as initial upper bound
    greedy_parts, _ = _greedy_partition(n, adj)
    best_parts = greedy_parts
    best_count = len(greedy_parts)
    
    # Compute vertex ordering for better search
    vertex_order = _compute_vertex_order(n, adj)
    
    # Try to find better solutions with fewer parts
    lb = _lower_bound_clique(n, adj)
    
    for k in range(max(1, lb), best_count):
        if time.time() > deadline - 2.0:
            break
        result = _try_k_parts(n, adj, k, deadline - 2.0, vertex_order)
        if result is not None:
            best_parts = result
            best_count = k
            break
    
    # Ensure proper types: convert to list of lists of Python ints
    final_parts = []
    for part in best_parts:
        final_parts.append([int(v) for v in part])
    
    return final_parts

# EVOLVE-BLOCK-END
