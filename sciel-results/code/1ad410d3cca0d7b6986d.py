"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [0] * n  # bitmask of neighbors
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return n, adj

def _is_clique_mask(mask, adj):
    """Check if vertices in mask form a clique."""
    m = mask
    while m:
        v_bit = m & (-m)
        v = v_bit.bit_length() - 1
        m -= v_bit
        if (adj[v] & mask) != (mask ^ v_bit):
            return False
    return True

def _is_indep_mask(mask, adj):
    """Check if vertices in mask form an independent set."""
    m = mask
    while m:
        v_bit = m & (-m)
        v = v_bit.bit_length() - 1
        m -= v_bit
        if adj[v] & mask:
            return False
    return True

def _greedy_partition(n, adj):
    """Greedy: assign each vertex to first compatible part, else new part.
    A part is valid if it's a clique OR independent set."""
    parts = []  # list of vertex bitmasks
    for v in range(n):
        v_bit = 1 << v
        placed = False
        for i in range(len(parts)):
            new_mask = parts[i] | v_bit
            if _is_clique_mask(new_mask, adj) or _is_indep_mask(new_mask, adj):
                parts[i] = new_mask
                placed = True
                break
        if not placed:
            parts.append(v_bit)
    return parts

def _mask_to_list(mask):
    """Convert bitmask to list of vertices."""
    result = []
    while mask:
        v_bit = mask & (-mask)
        result.append(v_bit.bit_length() - 1)
        mask -= v_bit
    return result

def _exact_search(n, adj, deadline, initial_upper_bound):
    """Branch and bound to find minimum cochromatic number."""
    best_k = [initial_upper_bound]
    best_parts = [None]
    start_time = time.time()
    
    # Precompute: for each pair of vertices, whether they can be in same clique/indep
    # can_clique[i][j] = True if i and j are adjacent
    # can_indep[i][j] = True if i and j are NOT adjacent
    
    # State: parts is a list of (mask, type) where type is 0=clique, 1=indep, -1=undecided (single vertex)
    # Actually, let's use: parts is list of masks, and we track type separately
    # For efficiency, let's use a different approach:
    # Assign vertices one by one. For each vertex, try adding to existing parts or creating new.
    
    # Use a smarter approach: 
    # parts[i] = mask of vertices in part i
    # part_type[i] = 0 (clique), 1 (indep), or -1 (undecided, only 1 vertex)
    
    parts = []
    part_types = []
    assigned_mask = [0]
    
    def can_add_to_part(part_mask, v_bit, ptype):
        """Check if vertex v can be added to part with given type."""
        if ptype == 0:  # clique
            v = v_bit.bit_length() - 1
            # v must be adjacent to all vertices in part
            return (adj[v] & part_mask) == part_mask
        elif ptype == 1:  # independent
            v = v_bit.bit_length() - 1
            # v must NOT be adjacent to any vertex in part
            return (adj[v] & part_mask) == 0
        else:  # undecided (single vertex)
            # Adding v makes it a clique if adjacent, indep if not
            # We'll decide the type after adding
            return True  # always can add, type will be determined
    
    def decide_type(part_mask, ptype):
        """If part has more than 1 vertex and type is undecided, decide it."""
        if ptype == -1:
            if part_mask & (part_mask - 1):  # more than 1 bit
                # Decide based on whether it's clique or indep
                if _is_clique_mask(part_mask, adj):
                    return 0
                elif _is_indep_mask(part_mask, adj):
                    return 1
                else:
                    return -2  # invalid
            else:
                return -1  # still single vertex
        return ptype
    
    def backtrack(assigned_count):
        nonlocal best_k, best_parts
        if time.time() > deadline:
            return False
        
        if len(parts) >= best_k[0]:
            return False
        
        if assigned_count == n:
            if len(parts) < best_k[0]:
                best_k[0] = len(parts)
                best_parts[0] = [list(_mask_to_list(p)) for p in parts]
            return True
        
        # Find next unassigned vertex
        v = -1
        for i in range(n):
            if not (assigned_mask[0] & (1 << i)):
                v = i
                break
        if v == -1:
            return False
        
        v_bit = 1 << v
        
        # Try adding to existing parts
        for i in range(len(parts)):
            ptype = part_types[i]
            if can_add_to_part(parts[i], v_bit, ptype):
                new_mask = parts[i] | v_bit
                new_type = decide_type(new_mask, ptype)
                if new_type != -2:
                    parts[i] = new_mask
                    part_types[i] = new_type
                    assigned_mask[0] |= v_bit
                    if backtrack(assigned_count + 1):
                        return True
                    parts[i] = parts[i] ^ v_bit
                    # Restore type
                    if new_type != ptype:
                        # Re-decide original type
                        if part_types[i] == -1 and parts[i] & (parts[i] - 1):
                            if _is_clique_mask(parts[i], adj):
                                part_types[i] = 0
                            else:
                                part_types[i] = 1
                        else:
                            part_types[i] = ptype
                    assigned_mask[0] ^= v_bit
        
        # Try creating new part
        if len(parts) + 1 < best_k[0]:
            parts.append(v_bit)
            part_types.append(-1)  # undecided
            assigned_mask[0] |= v_bit
            if backtrack(assigned_count + 1):
                return True
            parts.pop()
            part_types.pop()
            assigned_mask[0] ^= v_bit
        
        return False
    
    backtrack(0)
    return best_parts[0]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n, adj = _build_graph()
    
    # Get greedy solution as upper bound
    greedy_parts = _greedy_partition(n, adj)
    greedy_count = len(greedy_parts)
    
    # Try exact search for better solution
    best_parts = _exact_search(n, adj, deadline, greedy_count)
    
    if best_parts is None:
        best_parts = [list(_mask_to_list(p)) for p in greedy_parts]
    
    # Ensure proper types
    final_parts = []
    for part in best_parts:
        final_parts.append([int(v) for v in part])
    
    return final_parts

# EVOLVE-BLOCK-END
