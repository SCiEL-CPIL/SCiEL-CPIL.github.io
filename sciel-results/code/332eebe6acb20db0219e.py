"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

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
    # Check sizes in increasing order
    for size in range(6, n + 1):
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

def _init_distance_threshold(n, threshold):
    """Initialize graph using distance threshold on random permutation.
    Edge exists if distance in permutation > threshold."""
    perm = list(range(n))
    random.shuffle(perm)
    pos = {v: i for i, v in enumerate(perm)}
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(pos[i] - pos[j])
            if dist > threshold:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _flip_edge_smart(adj, verts, n):
    """Flip an edge that maximizes degree heterogeneity in the violating set."""
    if len(verts) < 2:
        return
    # Compute degrees within the violating set
    degs = {}
    for v in verts:
        degs[v] = 0
    for i, u in enumerate(verts):
        for v in verts[i+1:]:
            if (adj[u] >> v) & 1:
                degs[u] += 1
                degs[v] += 1
    
    # Find the pair whose flip would most change degree variance
    best_pair = None
    best_score = -1
    # Sample some pairs to save time
    pairs_to_check = []
    for i in range(len(verts)):
        for j in range(i+1, len(verts)):
            pairs_to_check.append((verts[i], verts[j]))
    random.shuffle(pairs_to_check)
    pairs_to_check = pairs_to_check[:20]  # Check up to 20 pairs
    
    for u, v in pairs_to_check:
        has_edge = (adj[u] >> v) & 1
        # Simulate flip
        new_degs = degs.copy()
        if has_edge:
            new_degs[u] -= 1
            new_degs[v] -= 1
        else:
            new_degs[u] += 1
            new_degs[v] += 1
        # Score: variance of degrees (higher is better for breaking regularity)
        vals = list(new_degs.values())
        mean = sum(vals) / len(vals)
        var = sum((x - mean)**2 for x in vals)
        if var > best_score:
            best_score = var
            best_pair = (u, v, has_edge)
    
    if best_pair:
        u, v, has_edge = best_pair
        if has_edge:
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    else:
        # Fallback: random flip
        u_idx, v_idx = random.sample(range(len(verts)), 2)
        u = verts[u_idx]
        v = verts[v_idx]
        if (adj[u] >> v) & 1:
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

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
        if time.time() > deadline - 10:
            break
            
        n = target_n
        
        # Number of restarts depends on n
        max_restarts = 12 if n >= 21 else 18
        
        for r in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Choose initialization strategy
            if r % 3 == 0:
                # Distance threshold init
                # Threshold around n/2 to get moderate edge density
                threshold = random.randint(max(1, n // 3), n - 2)
                adj = _init_distance_threshold(n, threshold)
            elif r % 3 == 1:
                # Random with varied p
                p = random.uniform(0.3, 0.6)
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            else:
                # Complementary of distance threshold
                threshold = random.randint(max(1, n // 3), n - 2)
                adj = _init_distance_threshold(n, n - threshold - 1)
                
            # Local Search with adaptive strategy
            found_valid = False
            steps_since_full_check = 0
            max_steps_between_full_checks = 25
            consecutive_no_violation = 0
            
            max_steps = 1500
            
            for step in range(max_steps):
                if time.time() > deadline - 10:
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
                        # Found a violation, flip an edge smartly
                        verts = _extract_vertices(mask)
                        _flip_edge_smart(adj, verts, n)
                        consecutive_no_violation = 0
                else:
                    # Sampled search for violations
                    masks6 = _get_masks(n, 6)
                    mask = None
                    for _ in range(25):
                        mask = random.choice(masks6)
                        if _is_regular_induced_fast(adj, mask):
                            break
                        mask = None
                    
                    steps_since_full_check += 1
                    
                    if mask is not None:
                        verts = _extract_vertices(mask)
                        _flip_edge_smart(adj, verts, n)
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
