import time
import random
import itertools
import numpy as np

DEADLINE = 150
SAFETY_MARGIN = 15
START_TIME = None

def _elapsed():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    return time.time() - START_TIME

def _deadline():
    return DEADLINE - SAFETY_MARGIN

def _is_regular_mask(mask, adj_masks):
    """Check if the induced subgraph on vertices in mask is regular."""
    degs = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        degs.append((adj_masks[v] & mask).bit_count())
        m ^= lsb
    if not degs:
        return True
    d0 = degs[0]
    for d in degs[1:]:
        if d != d0:
            return False
    return True

def _find_violation(adj_masks, n):
    """Find any regular induced subgraph of size >= 6. Returns (True, combo) or (False, None)."""
    # First check 6-sets (most common violation)
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _is_regular_mask(mask, adj_masks):
            return True, combo
    # Check larger sets
    for k in range(7, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if _is_regular_mask(mask, adj_masks):
                return True, combo
    return False, None

def _build_edge_to_subsets(n):
    """Precompute map from edge (u,v) to list of 6-subset indices containing both u and v."""
    subsets_6 = list(itertools.combinations(range(n), 6))
    subset_masks = [0] * len(subsets_6)
    for i, s in enumerate(subsets_6):
        mask = 0
        for v in s:
            mask |= (1 << v)
        subset_masks[i] = mask
    
    edge_to_subsets = {}
    for i, s in enumerate(subsets_6):
        for u in s:
            for v in s:
                if u < v:
                    key = (u, v)
                    if key not in edge_to_subsets:
                        edge_to_subsets[key] = []
                    edge_to_subsets[key].append(i)
    
    return subsets_6, subset_masks, edge_to_subsets

def _local_search_6(adj_masks, n, subsets_6, subset_masks, edge_to_subsets, rng, max_iters=5000):
    """
    Local search to eliminate all regular 6-sets.
    Returns (success, adj_masks, num_violations).
    """
    # Initialize reg_status
    reg_status = [False] * len(subsets_6)
    violating_indices = []
    for i in range(len(subsets_6)):
        if _is_regular_mask(subset_masks[i], adj_masks):
            reg_status[i] = True
            violating_indices.append(i)
    
    num_violations = len(violating_indices)
    
    for it in range(max_iters):
        if _elapsed() > _deadline():
            break
        
        if num_violations == 0:
            break
        
        if not violating_indices:
            break
        
        # Pick a random violating subset
        idx = rng.choice(violating_indices)
        s_mask = subset_masks[idx]
        
        # Verify it's still violating
        if not _is_regular_mask(s_mask, adj_masks):
            reg_status[idx] = False
            violating_indices.remove(idx)
            num_violations -= 1
            continue
        
        # Get vertices in this subset
        verts = []
        m = s_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        # Try all pairs in this subset, pick the one that breaks the most violations
        best_pair = None
        best_reduction = 0
        best_new_violations = 0
        
        for u_idx in range(len(verts)):
            for v_idx in range(u_idx + 1, len(verts)):
                u, v = verts[u_idx], verts[v_idx]
                edge_exists = (adj_masks[u] >> v) & 1
                
                # Get affected subsets
                key = (min(u, v), max(u, v))
                affected = edge_to_subsets.get(key, [])
                
                # Simulate the flip
                # Count how many affected subsets are currently regular
                current_reg_count = sum(1 for a in affected if reg_status[a])
                
                # After flip, check which affected subsets would be regular
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                
                new_reg_count = 0
                for a in affected:
                    if _is_regular_mask(subset_masks[a], adj_masks):
                        new_reg_count += 1
                
                # Revert
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                
                reduction = current_reg_count - new_reg_count
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_pair = (u, v)
                    best_new_violations = new_reg_count
        
        if best_pair is None or best_reduction <= 0:
            # No flip helps, try a random flip anyway
            u, v = rng.sample(verts, 2)
        else:
            u, v = best_pair
        
        edge_exists = (adj_masks[u] >> v) & 1
        key = (min(u, v), max(u, v))
        affected = edge_to_subsets.get(key, [])
        
        # Perform the flip
        adj_masks[u] ^= (1 << v)
        adj_masks[v] ^= (1 << u)
        
        # Update reg_status for affected subsets
        for a in affected:
            was_reg = reg_status[a]
            is_reg = _is_regular_mask(subset_masks[a], adj_masks)
            reg_status[a] = is_reg
            
            if was_reg and not is_reg:
                num_violations -= 1
                if a in violating_indices:
                    violating_indices.remove(a)
            elif not was_reg and is_reg:
                num_violations += 1
                violating_indices.append(a)
    
    return num_violations == 0, adj_masks, num_violations

def _find_any_regular_6plus(adj_masks, n):
    """Check for any regular induced subgraph of size >= 6."""
    return _find_violation(adj_masks, n)

def _random_graph(n, p, rng):
    """Generate a random graph with edge probability p."""
    adj_masks = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
    return adj_masks

def _get_edges_from_masks(adj_masks, n):
    """Extract edge list from adjacency masks."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_masks[u] >> v) & 1:
                edges.append([u, v])
    return edges

def _try_find_graph(n, rng, max_restarts=10):
    """Try to find a valid graph on n vertices."""
    subsets_6, subset_masks, edge_to_subsets = _build_edge_to_subsets(n)
    
    for restart in range(max_restarts):
        if _elapsed() > _deadline():
            break
        
        # Try different edge probabilities
        p = 0.3 + 0.4 * (restart / max(max_restarts - 1, 1))
        adj_masks = _random_graph(n, p, rng)
        
        # Local search to eliminate 6-set violations
        success, adj_masks, num_viol = _local_search_6(
            adj_masks, n, subsets_6, subset_masks, edge_to_subsets, rng, max_iters=3000
        )
        
        if num_viol > 0:
            continue
        
        # Check for larger violations
        has_viol, viol_set = _find_any_regular_6plus(adj_masks, n)
        if not has_viol:
            edges = _get_edges_from_masks(adj_masks, n)
            return {'n': n, 'edges': edges}
    
    return None

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Try n = 21, 20, 19, 18 in order
    for n in [21, 20, 19, 18]:
        if _elapsed() > _deadline():
            break
        
        result = _try_find_graph(n, rng, max_restarts=8 if n <= 20 else 4)
        if result is not None:
            return result
    
    # Fallback: return a small valid graph
    # A graph with no edges on 5 vertices has no regular induced subgraph on 6+ vertices
    return {'n': 5, 'edges': []}
