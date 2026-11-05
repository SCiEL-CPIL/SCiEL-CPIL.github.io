import numpy as np
import time
import random

# EVOLVE-BLOCK-START
_N_CACHE = {}

def _fast_c4_check(adj, u, v):
    """
    Fast C4 check using bitsets (int masks).
    Returns True if adding (u, v) creates a C4.
    """
    mask_u = adj[u] & ~(1 << v)
    mask_v = adj[v] & ~(1 << u)
    
    # Iterate over the smaller mask for speed
    if mask_u.bit_count() > mask_v.bit_count():
        # Iterate mask_v instead: exists y in mask_v with neighbor in mask_u
        m = mask_v
        while m:
            lsb = m & (-m)
            y = lsb.bit_length() - 1
            if adj[y] & mask_u:
                return True
            m ^= lsb
    else:
        m = mask_u
        while m:
            lsb = m & (-m)
            x = lsb.bit_length() - 1
            if adj[x] & mask_v:
                return True
            m ^= lsb
    
    return False

def _try_construct_fast(n, required_min_deg, deadline, seed):
    """
    Greedy construction with local search swaps to escape local minima.
    """
    rng = np.random.RandomState(seed)
    
    adj = [0] * n
    degrees = [0] * n
    all_mask = (1 << n) - 1
    
    # Phase 1: Greedy edge addition with varied heuristics
    stuck_count = 0
    while time.time() < deadline:
        if min(degrees) >= required_min_deg:
            break
        
        needy = [i for i in range(n) if degrees[i] < required_min_deg]
        if not needy:
            break
        
        # Pick needy vertex with bias toward most deficient
        u = needy[rng.randint(0, len(needy))]
        non_neighbors_mask = all_mask & ~adj[u] & ~(1 << u)
        
        if non_neighbors_mask == 0:
            stuck_count += 1
            if stuck_count > 30:
                break
            continue
        
        candidates = []
        m = non_neighbors_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            candidates.append(v)
            m ^= lsb
        
        if not candidates:
            stuck_count += 1
            if stuck_count > 30:
                break
            continue
        
        # Varied ordering strategies
        r = rng.random()
        if r < 0.3:
            candidates.sort(key=lambda x: degrees[x])  # ascending degree
        elif r < 0.5:
            candidates.sort(key=lambda x: -degrees[x])  # descending degree  
        else:
            rng.shuffle(candidates)  # random
        
        added = False
        for v in candidates:
            if _fast_c4_check(adj, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
            added = True
            stuck_count = 0
            break
        
        if not added:
            stuck_count += 1
            if stuck_count > 50:
                break
    
    # Phase 2: Local search - edge swaps to escape local minima
    if min(degrees) < required_min_deg:
        for swap_iter in range(300):
            if time.time() > deadline:
                break
            if min(degrees) >= required_min_deg:
                break
            
            needy = [i for i in range(n) if degrees[i] < required_min_deg]
            if not needy:
                break
            
            u = needy[rng.randint(0, len(needy))]
            
            # Get neighbors of u
            neighbors = []
            m = adj[u]
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                neighbors.append(v)
                m ^= lsb
            
            if not neighbors:
                break
            
            # Pick a neighbor to potentially remove
            v = neighbors[rng.randint(0, len(neighbors))]
            
            # Get non-neighbors of u
            non_neighbors_mask = all_mask & ~adj[u] & ~(1 << u)
            if non_neighbors_mask == 0:
                break
            
            candidates = []
            m = non_neighbors_mask
            while m:
                lsb = m & (-m)
                w = lsb.bit_length() - 1
                candidates.append(w)
                m ^= lsb
            
            if not candidates:
                break
            
            rng.shuffle(candidates)
            
            # Try swap: remove (u,v), add (u,w)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            degrees[u] -= 1
            degrees[v] -= 1
            
            success = False
            for w in candidates:
                if _fast_c4_check(adj, u, w):
                    continue
                # Accept if it doesn't make min degree worse
                new_min = min(degrees[u]+1, degrees[v], degrees[w]+1, 
                             min(degrees[i] for i in range(n) if i not in (u,v,w)))
                if new_min >= min(degrees):
                    adj[u] |= (1 << w)
                    adj[w] |= (1 << u)
                    degrees[u] += 1
                    degrees[w] += 1
                    success = True
                    break
            
            if not success:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
    
    if min(degrees) < required_min_deg:
        return None
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
    
    return (n, edges)

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145
    
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        max_attempts = 3000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            result = _try_construct_fast(n, required_min_deg, deadline, attempt * 7919 + 13)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}
    
    return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
