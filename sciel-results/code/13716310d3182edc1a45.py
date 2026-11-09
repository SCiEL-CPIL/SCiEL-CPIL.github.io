import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Target n=21. If fails, try 20.
    for n in [21, 20]:
        if time.time() > deadline - 10:
            break
        result = _search_graph(n, deadline - 5)
        if result:
            return result

    # Final fallback
    return {'n': 5, 'edges': []}

def _search_graph(n, deadline):
    rng = random.Random(42)
    
    # Initialize with a Paley-like graph or random
    # Paley graph requires n to be p^k where p is prime congruent to 1 mod 4.
    # For n=21, we can just use random with density 0.5, or a specific construction.
    # Let's try a few initializations.
    
    best_result = None
    
    # Precompute subset masks for sizes 6, 7, 8
    # These are the most likely to be violated and most numerous
    track_sizes = [6, 7, 8]
    subsets_by_size = {s: [] for s in track_sizes}
    masks_by_size = {s: [] for s in track_sizes}
    
    for size in track_sizes:
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= 1 << v
            subsets_by_size[size].append(subset)
            masks_by_size[size].append(mask)
            
    # Flatten for easier access
    all_subsets = []
    all_masks = []
    all_sizes = []
    for s in track_sizes:
        all_subsets.extend(subsets_by_size[s])
        all_masks.extend(masks_by_size[s])
        all_sizes.extend([s] * len(subsets_by_size[s]))
        
    num_subsets = len(all_subsets)
    
    # Map edge (u,v) to indices of subsets containing it
    # This is expensive to build fully, so we might just re-scan if n is small
    # For n=21, num_subsets ~ 170k. 
    # Building edge_to_indices:
    # Each subset of size k has k(k-1)/2 edges.
    # Total edge-subset pairs is manageable.
    
    edge_to_indices = {}
    for i, subset in enumerate(all_subsets):
        s_set = set(subset)
        for u in subset:
            for v in subset:
                if u < v:
                    key = (u, v)
                    if key not in edge_to_indices:
                        edge_to_indices[key] = []
                    edge_to_indices[key].append(i)

    # Try multiple random initializations
    for init_iter in range(50):
        if time.time() > deadline:
            break
            
        adj = [0] * n
        # Random initialization
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i

        is_reg = [False] * num_subsets
        viol_indices = []
        
        # Initial check
        for i in range(num_subsets):
            if _is_regular_fast(adj, all_masks[i]):
                is_reg[i] = True
                viol_indices.append(i)
                
        # If no violations in tracked sizes, check full validity
        if not viol_indices:
            if _is_valid_full(adj, n):
                return _to_result(adj, n)
            else:
                # Find a violation in untracked sizes (9+)
                viol_subset = _find_violation_full(adj, n)
                if viol_subset:
                    # Flip an edge in it
                    u, v = rng.sample(viol_subset, 2)
                    _flip_edge(adj, u, v)
                    # Update tracked violations
                    _update_violations(adj, all_subsets, all_masks, is_reg, viol_indices, edge_to_indices, (u,v))
                else:
                    return _to_result(adj, n)
        
        # Local Search
        stagnation_counter = 0
        while time.time() < deadline:
            if not viol_indices:
                if _is_valid_full(adj, n):
                    return _to_result(adj, n)
                else:
                    viol_subset = _find_violation_full(adj, n)
                    if viol_subset:
                        u, v = rng.sample(viol_subset, 2)
                        _flip_edge(adj, u, v)
                        _update_violations(adj, all_subsets, all_masks, is_reg, viol_indices, edge_to_indices, (u,v))
                    else:
                        return _to_result(adj, n)
                continue

            # Pick a random violating subset
            idx = rng.choice(viol_indices)
            subset = all_subsets[idx]
            u, v = rng.sample(subset, 2)
            
            _flip_edge(adj, u, v)
            _update_violations(adj, all_subsets, all_masks, is_reg, viol_indices, edge_to_indices, (u,v))
            
            # Optional: Random restart if stuck? 
            # The current logic is a simple hill climber.
            # To escape local minima, we can occasionally flip a random edge.
            if stagnation_counter > 100:
                # Flip a random edge anywhere
                u, v = rng.sample(range(n), 2)
                _flip_edge(adj, u, v)
                _update_violations(adj, all_subsets, all_masks, is_reg, viol_indices, edge_to_indices, (u,v))
                stagnation_counter = 0
            else:
                stagnation_counter += 1

    return None

def _is_regular_fast(adj, mask):
    # adj is list of int masks
    # mask is int mask of the subset
    # We need to check if all vertices in mask have the same degree within mask
    # Since n <= 22, we can iterate
    # But we can do it faster if we precompute? No, adj changes.
    # Iterating over bits in mask:
    deg = -1
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        v = lsb.bit_length() - 1
        d = (adj[v] & mask).bit_count()
        if deg == -1:
            deg = d
        elif d != deg:
            return False
        m ^= lsb
    return True

def _is_valid_full(adj, n):
    # Check all subsets of size >= 6
    # Optimization: If no regular subset of size 6, 7, 8, it's very likely valid.
    # But we must be sure.
    # We can iterate sizes from 6 to n
    for size in range(6, n+1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= 1 << v
            if _is_regular_fast(adj, mask):
                return False
    return True

def _find_violation_full(adj, n):
    for size in range(6, n+1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= 1 << v
            if _is_regular_fast(adj, mask):
                return subset
    return None

def _flip_edge(adj, u, v):
    adj[u] ^= 1 << v
    adj[v] ^= 1 << u

def _update_violations(adj, subsets, masks, is_reg, viol_indices, edge_to_indices, edge):
    affected = edge_to_indices.get(edge, [])
    for idx in affected:
        old_reg = is_reg[idx]
        new_reg = _is_regular_fast(adj, masks[idx])
        if old_reg != new_reg:
            is_reg[idx] = new_reg
            if new_reg:
                viol_indices.append(idx)
            else:
                try:
                    viol_indices.remove(idx)
                except ValueError:
                    pass

def _to_result(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
