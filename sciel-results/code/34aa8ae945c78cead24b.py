"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_MASKS_CACHE = {}

def _get_masks(n):
    """Precompute masks for all subsets of size >= 6."""
    if n not in _MASKS_CACHE:
        _MASKS_CACHE[n] = {}
        for size in range(6, n + 1):
            masks = []
            for combo in combinations(range(n), size):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                masks.append(mask)
            _MASKS_CACHE[n][size] = masks
    return _MASKS_CACHE[n]

def _is_regular_induced_fast(adj, mask):
    """Check if induced subgraph on mask is regular."""
    count = mask.bit_count()
    if count < 6:
        return False
    m = mask
    first_deg = -1
    while m:
        lb = m & (-m)
        idx = lb.bit_length() - 1
        deg = (adj[idx] & mask).bit_count()
        if first_deg == -1:
            first_deg = deg
        elif deg != first_deg:
            return False
        m ^= lb
    return True

def _find_one_regular_subgraph(adj, n):
    """Find any regular induced subgraph of size >= 6."""
    masks_by_size = _get_masks(n)
    # Check sizes in increasing order
    for size in range(6, n + 1):
        masks = masks_by_size[size]
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

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to find a valid graph for n=20 down to 15
    # We prioritize larger n
    for target_n in range(20, 14, -1):
        if time.time() > deadline - 10:
            break
            
        n = target_n
        
        # Number of restarts depends on n
        # Larger n is harder, give it more restarts? Or less time per restart?
        # Let's give 3 restarts for n=20, 5 for n=19, etc.
        max_restarts = 3 if n == 20 else 5
        
        for r in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Local Search
            max_steps = 300
            found_valid = False
            for step in range(max_steps):
                if time.time() > deadline - 10:
                    break
                    
                # Find a regular subgraph
                mask = _find_one_regular_subgraph(adj, n)
                
                if mask is None:
                    # No regular subgraph found. Graph is valid.
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    found_valid = True
                    break
                    
                # Found a regular subgraph. Flip an edge inside it to break regularity.
                verts = _extract_vertices(mask)
                if len(verts) < 2:
                    continue
                    
                u_idx, v_idx = random.sample(range(len(verts)), 2)
                u = verts[u_idx]
                v = verts[v_idx]
                
                # Flip edge (u, v)
                if (adj[u] >> v) & 1:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
            if found_valid:
                # We found a valid graph for this n. 
                # Since we are iterating n downwards, this is the best n we can guarantee 
                # without trying larger n again (which we already did or are doing).
                # Actually, if we found n=20, we are done. 
                # If we found n=19, we stop the n=19 loop and return 19 (since 20 failed).
                break
                
    # Fallback if no graph > 5 found
    if best_n <= 5:
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
