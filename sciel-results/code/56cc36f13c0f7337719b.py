"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_MASKS_CACHE = {}

def _get_masks(n, size):
    """Precompute masks for subsets of a specific size."""
    key = (n, size)
    if key not in _MASKS_CACHE:
        masks = []
        for combo in combinations(range(n), size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            masks.append(mask)
        _MASKS_CACHE[key] = masks
    return _MASKS_CACHE[key]

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

def _find_one_regular_subgraph_full(adj, n):
    """Find any regular induced subgraph of size >= 6 (exhaustive)."""
    # Check size 6 first (most common violation), then larger
    for size in range(6, min(n + 1, 12)):  # Cap at 11 for speed; larger regular subgraphs are rarer
        masks = _get_masks(n, size)
        for mask in masks:
            if _is_regular_induced_fast(adj, mask):
                return mask
    # If no violation up to size 11, check remaining sizes
    for size in range(12, n + 1):
        masks = _get_masks(n, size)
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

def _count_violations_size6(adj, n):
    """Count how many 6-subsets are regular."""
    masks6 = _get_masks(n, 6)
    count = 0
    for mask in masks6:
        if _is_regular_induced_fast(adj, mask):
            count += 1
    return count

def _flip_edge(adj, u, v):
    """Flip edge between u and v."""
    if (adj[u] >> v) & 1:
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
    else:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

def _best_flip_in_subgraph(adj, verts):
    """Try all pairs in verts, return the flip that minimizes size-6 violations in the subgraph."""
    n = len(verts)
    best_u, best_v = -1, -1
    best_score = float('inf')
    # Precompute current degrees within subgraph
    deg = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            u = verts[i]
            v = verts[j]
            if (adj[u] >> v) & 1:
                deg[i] += 1
                deg[j] += 1
    # Try each pair
    for i in range(n):
        for j in range(i + 1, n):
            u = verts[i]
            v = verts[j]
            has_edge = (adj[u] >> v) & 1
            # Compute change in regularity: if we flip, degrees change
            # New deg[i] = deg[i] + (1 if not has_edge else -1)
            # New deg[j] = deg[j] + (1 if not has_edge else -1)
            # All other degrees unchanged
            delta = 1 if not has_edge else -1
            new_degs = deg[:]
            new_degs[i] += delta
            new_degs[j] += delta
            # Check if this makes the subgraph more non-regular
            # Score = variance of degrees (lower is more regular = worse)
            # We want to maximize non-regularity, so minimize variance
            mean_deg = sum(new_degs) / n
            variance = sum((d - mean_deg) ** 2 for d in new_degs)
            if variance > best_score:
                best_score = variance
                best_u, best_v = u, v
    return best_u, best_v

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s safety margin
    
    best_n = 5
    best_edges = []
    
    # Precompute masks for all n we might try
    for target_n in range(15, 23):
        _get_masks(target_n, 6)
        _get_masks(target_n, 7)
        _get_masks(target_n, 8)
    
    # Try to find a valid graph for n=22 down to 15
    # We prioritize larger n
    for target_n in range(22, 14, -1):
        if time.time() > deadline - 15:
            break
            
        n = target_n
        
        # Number of restarts depends on n
        max_restarts = 20 if n >= 21 else 30
        
        for r in range(max_restarts):
            if time.time() > deadline - 15:
                break
                
            # Random initialization with slight bias toward sparser graphs
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.42:  # Slightly sparser
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Local Search with adaptive strategy
            found_valid = False
            steps_since_full_check = 0
            max_steps_between_full_checks = 25
            consecutive_no_violation = 0
            min_violations_seen = float('inf')
            
            max_steps = 3000
            
            for step in range(max_steps):
                if time.time() > deadline - 15:
                    break
                    
                # Periodically do a full check
                if steps_since_full_check >= max_steps_between_full_checks:
                    mask = _find_one_regular_subgraph_full(adj, n)
                    steps_since_full_check = 0
                    if mask is None:
                        # Valid!
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
                    else:
                        # Found a violation, use best flip in subgraph
                        verts = _extract_vertices(mask)
                        if len(verts) >= 2:
                            u, v = _best_flip_in_subgraph(adj, verts)
                            if u != -1:
                                _flip_edge(adj, u, v)
                        consecutive_no_violation = 0
                else:
                    # Sampled search for violations
                    masks6 = _get_masks(n, 6)
                    mask = None
                    for _ in range(40):
                        mask = random.choice(masks6)
                        if _is_regular_induced_fast(adj, mask):
                            break
                        mask = None
                    
                    steps_since_full_check += 1
                    
                    if mask is not None:
                        verts = _extract_vertices(mask)
                        if len(verts) >= 2:
                            u, v = _best_flip_in_subgraph(adj, verts)
                            if u != -1:
                                _flip_edge(adj, u, v)
                        consecutive_no_violation = 0
                    else:
                        consecutive_no_violation += 1
                        # If we haven't found violations in a while, do full check sooner
                        if consecutive_no_violation > 8:
                            steps_since_full_check = max_steps_between_full_checks

            if found_valid:
                break
                
    # Fallback if no graph > 5 found
    if best_n <= 5:
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
