import time
import random
from itertools import combinations

# Caches for masks to avoid recomputation
_MASKS_6_CACHE = {}
_ALL_MASKS_CACHE = {}

def _get_masks_6(n):
    """Precompute masks for subsets of size 6."""
    if n not in _MASKS_6_CACHE:
        masks = []
        for combo in combinations(range(n), 6):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            masks.append(mask)
        _MASKS_6_CACHE[n] = masks
    return _MASKS_6_CACHE[n]

def _get_all_masks(n):
    """Precompute masks for all subsets of size >= 6."""
    if n not in _ALL_MASKS_CACHE:
        _ALL_MASKS_CACHE[n] = {}
        for size in range(6, n + 1):
            masks = []
            for combo in combinations(range(n), size):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                masks.append(mask)
            _ALL_MASKS_CACHE[n][size] = masks
    return _ALL_MASKS_CACHE[n]

def _is_regular_induced_fast(adj, mask):
    """
    Check if the induced subgraph on 'mask' is regular.
    adj is a list of integers where adj[i] is the bitmask of neighbors of i.
    """
    count = mask.bit_count()
    if count < 6:
        return False
    
    m = mask
    first_deg = -1
    while m:
        lb = m & (-m)
        idx = lb.bit_length() - 1
        # Degree of idx within mask is popcount(adj[idx] & mask)
        deg = (adj[idx] & mask).bit_count()
        if first_deg == -1:
            first_deg = deg
        elif deg != first_deg:
            return False
        m ^= lb
    return True

def _find_violation(adj, n, masks_6, all_masks):
    """
    Find any regular induced subgraph of size >= 6.
    Returns the mask of the violating subgraph, or None if valid.
    Prioritizes size 6 as they are most common and cheapest to check in bulk.
    """
    # 1. Check size 6
    # Randomize the order of checking to avoid bias in local search?
    # Or just iterate. For speed, iterating is fine.
    # To speed up further, we could shuffle masks_6 each call, but that's expensive.
    # Instead, we can just pick a random subset of masks to check first?
    # Let's just iterate. 50k checks is fast in Python if the check is simple.
    
    # Optimization: If we check all size 6 masks, it's deterministic.
    # Let's iterate through them.
    for mask in masks_6:
        if _is_regular_induced_fast(adj, mask):
            return mask
            
    # 2. If no size 6 violation, check larger sizes
    # This is rare if size 6 is clean, so we can be less aggressive here.
    for size in range(7, n + 1):
        if size not in all_masks:
            continue
        masks = all_masks[size]
        for mask in masks:
            if _is_regular_induced_fast(adj, mask):
                return mask
                
    return None

def _extract_vertices(mask):
    """Extract list of vertex indices from mask."""
    verts = []
    m = mask
    while m:
        lb = m & (-m)
        idx = lb.bit_length() - 1
        verts.append(idx)
        m ^= lb
    return verts

def _edges_from_adj(adj, n):
    """Convert bitmask adjacency to edge list."""
    edges = []
    for i in range(n):
        # Only consider j > i to avoid duplicates
        mask_i = adj[i] & ~((1 << (i + 1)) - 1) # Mask out self and lower indices
        m = mask_i
        while m:
            lb = m & (-m)
            j = lb.bit_length() - 1
            edges.append([i, j])
            m ^= lb
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to find valid graphs for n = 22, 21, 20, 19
    # We prioritize larger n as the score is n.
    for target_n in [22, 21, 20, 19]:
        if time.time() > deadline - 5:
            break
            
        n = target_n
        
        # Precompute masks
        masks_6 = _get_masks_6(n)
        all_masks = _get_all_masks(n)
        
        found_valid = False
        restarts_attempted = 0
        max_restarts = 20 # Cap restarts per n
        
        while restarts_attempted < max_restarts and not found_valid:
            if time.time() > deadline - 5:
                break
            restarts_attempted += 1
            
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Local Search Loop
            # We run this until we find a valid graph or run out of time
            # To avoid getting stuck in cycles, we limit steps per restart or use random restarts
            steps = 0
            max_steps = 2000 # Cap steps per restart to allow multiple restarts
            
            while steps < max_steps and time.time() < deadline - 5:
                steps += 1
                
                # Find a violation
                mask = _find_violation(adj, n, masks_6, all_masks)
                
                if mask is None:
                    # Valid graph found!
                    edges = _edges_from_adj(adj, n)
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    found_valid = True
                    break
                else:
                    # Found a violation. Flip an edge within this subgraph to break it.
                    verts = _extract_vertices(mask)
                    if len(verts) < 2:
                        continue
                    
                    # Pick two random vertices from the violating subgraph
                    u_idx, v_idx = random.sample(range(len(verts)), 2)
                    u = verts[u_idx]
                    v = verts[v_idx]
                    
                    # Flip the edge (u, v)
                    if (adj[u] >> v) & 1:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # If we exhausted steps for this restart, we try a new random restart
            # The outer while loop handles this.
            
        if found_valid:
            # Found a valid graph for this n. 
            # Since we iterate n in descending order, this is the best possible for this run.
            # We can break and return, or try to see if we can find a better one?
            # No, n is fixed. We found a valid graph for the max n we tried.
            # Actually, we should break the n-loop because we found a valid graph for the current target_n.
            # And since we go 22->21->20, if we find one for 22, we are done.
            break
            
    # Fallback
    if best_n <= 5:
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
