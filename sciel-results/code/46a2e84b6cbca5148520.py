"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

# Precompute adjacency as bitmasks
N = P['n']
adj = [0] * N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

# Precompute all subsets that are cliques or independent sets
# For a subset S (bitmask), check if it's a clique or independent set
def is_clique_mask(mask):
    """Check if the set of vertices in mask forms a clique."""
    if mask == 0:
        return True
    # For every pair in the set, they must be adjacent
    # Equivalent: for every vertex v in mask, (mask without v) must be subset of adj[v]
    # More efficiently: for every v in mask, mask & ~adj[v] & ~(1<<v) must be 0
    # Actually: for every v in mask, (mask & ~adj[v]) should only possibly contain v itself
    # So (mask & ~adj[v] & ~(1<<v)) == 0 for all v in mask
    m = mask
    while m:
        v = (m & -m).bit_length() - 1
        # Check that all other vertices in mask are adjacent to v
        if (mask & ~adj[v] & ~(1 << v)) != 0:
            return False
        m &= m - 1
    return True

def is_independent_mask(mask):
    """Check if the set of vertices in mask forms an independent set."""
    if mask == 0:
        return True
    # For every v in mask, no other vertex in mask should be adjacent to v
    # (mask & adj[v] & ~(1<<v)) == 0 for all v in mask
    m = mask
    while m:
        v = (m & -m).bit_length() - 1
        if (mask & adj[v] & ~(1 << v)) != 0:
            return False
        m &= m - 1
    return True

# Precompute valid subsets (cliques or independent sets)
# There are 2^14 = 16384 subsets, so we can precompute all
valid_subsets = {}  # mask -> True if valid
for mask in range(1, 1 << N):
    if is_clique_mask(mask) or is_independent_mask(mask):
        valid_subsets[mask] = True

# Also store the list of vertices for each valid subset
subset_vertices = {}
for mask in valid_subsets:
    verts = [i for i in range(N) if (mask >> i) & 1]
    subset_vertices[mask] = verts

def construct():
    """Find minimum cochromatic partition using branch and bound."""
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    all_vertices_mask = (1 << N) - 1
    
    best_parts = None
    best_count = N  # worst case: each vertex in its own part
    
    # Precompute valid subsets sorted by size descending (prefer larger subsets)
    # This helps find good solutions quickly
    valid_masks_sorted = sorted(valid_subsets.keys(), key=lambda m: -bin(m).count('1'))
    
    def backtrack(remaining_mask, current_parts):
        nonlocal best_parts, best_count, deadline
        
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if len(current_parts) < best_count:
                best_count = len(current_parts)
                best_parts = [list(p) for p in current_parts]
            return
        
        # Pruning: if we can't beat best_count, stop
        # Minimum parts needed: at least 1, and at most popcount(remaining_mask)
        # If len(current_parts) + 1 >= best_count, we still need at least 1 more part
        # So if len(current_parts) >= best_count, prune
        if len(current_parts) >= best_count:
            return
        
        # Try to find a valid subset of remaining_mask
        # We'll iterate through valid subsets that are subsets of remaining_mask
        # To make this efficient, we can use the sorted list and check subset relation
        
        # Better approach: iterate through all valid subsets that are subsets of remaining_mask
        # For each, check if it's a valid subset of remaining_mask
        # We can do this by checking (mask & remaining_mask) == mask
        
        # Optimization: only consider maximal valid subsets of remaining_mask
        # But for simplicity, let's just iterate and add pruning
        
        # Get vertices in remaining_mask
        rem_verts = [i for i in range(N) if (remaining_mask >> i) & 1]
        
        # Try each valid subset that is a subset of remaining_mask
        # To reduce search, try larger subsets first
        for vmask in valid_masks_sorted:
            if time.time() > deadline:
                return
            
            # Check if vmask is a subset of remaining_mask
            if (vmask & remaining_mask) != vmask:
                continue
            
            # Also, we should ensure we're making progress
            # The subset must be non-empty
            if vmask == 0:
                continue
            
            # Pruning: if we take this subset, the remaining vertices need
            # at least some number of parts. A simple lower bound is 1 if remaining is non-empty.
            # A better bound: the chromatic number of the remaining graph or
            # the independence number based lower bound.
            
            new_remaining = remaining_mask & ~vmask
            if new_remaining == 0:
                # This completes the partition
                if len(current_parts) + 1 < best_count:
                    best_count = len(current_parts) + 1
                    best_parts = [list(p) for p in current_parts] + [subset_vertices[vmask]]
                return
            else:
                # Continue with this subset
                current_parts.append(subset_vertices[vmask])
                backtrack(new_remaining, current_parts)
                current_parts.pop()
    
    backtrack(all_vertices_mask, [])
    
    if best_parts is None:
        # Fallback: use greedy
        parts = []
        for v in range(N):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in set(map(tuple, P['edges'])) for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts
    
    # Convert to lists of ints
    return [[int(v) for v in part] for part in best_parts]

# EVOLVE-BLOCK-END
