import time
import random
from itertools import combinations

# Precomputation for Q3 detection
# Q3 is K4,4 minus a perfect matching.
# G contains Q3 in bipartition (A,B) iff there exists a perfect matching M
# such that M is disjoint from the missing edges in G[A,B].
# i.e., all edges in M are present in G.

# Generate the 24 perfect matching masks for a 4x4 grid (indices 0..15)
def generate_matching_masks():
    masks = []
    for perm in __import__('itertools').permutations(range(4)):
        mask = 0
        for i in range(4):
            j = perm[i]
            # Bit position i*4 + j
            mask |= (1 << (i * 4 + j))
        masks.append(mask)
    return masks

MATCHING_MASKS = generate_matching_masks()

# Precompute all 8-subsets and their bipartitions
# For each bipartition, we need to quickly extract the 16-bit mask of edges/missing edges.
# We will store:
# 1. The 8 vertices of the subset.
# 2. The 4 vertices of A.
# 3. The 4 vertices of B.
# 4. A list of 16 pairs (u, v) corresponding to the bits in the 16-bit mask.
#    Bit k corresponds to edge (A[k//4], B[k%4]).

BIPARTITIONS = []

def _precompute_bipartitions():
    n = 16
    for s in combinations(range(n), 8):
        s_set = set(s)
        for a in combinations(s, 4):
            a_set = set(a)
            b = tuple(v for v in s if v not in a_set)
            # Store the bipartition
            # We need a fast way to get the mask.
            # We will store the list of (u, v) for the 16 positions.
            edge_pairs = []
            for i in range(4):
                for j in range(4):
                    edge_pairs.append((a[i], b[j]))
            BIPARTITIONS.append((a, b, edge_pairs))

_precompute_bipartitions()

def _get_missing_mask(adj_masks, edge_pairs):
    """
    Returns a 16-bit mask where bit k is 1 if edge edge_pairs[k] is MISSING.
    adj_masks: list of 16 ints.
    """
    mask = 0
    for k, (u, v) in enumerate(edge_pairs):
        if not (adj_masks[u] >> v) & 1:
            mask |= (1 << k)
    return mask

def _contains_q3(adj_masks):
    """
    Returns True if Q3 is present.
    """
    for a, b, edge_pairs in BIPARTITIONS:
        miss_mask = _get_missing_mask(adj_masks, edge_pairs)
        # Check if miss_mask is disjoint from any perfect matching
        for m in MATCHING_MASKS:
            if (miss_mask & m) == 0:
                return True
    return False

def _find_q3_edge(adj_masks):
    """
    Returns an edge (u, v) that is part of a Q3, or None if Q3-free.
    """
    for a, b, edge_pairs in BIPARTITIONS:
        miss_mask = _get_missing_mask(adj_masks, edge_pairs)
        for m in MATCHING_MASKS:
            if (miss_mask & m) == 0:
                # Q3 found. The edges in the Q3 are those in the matching m.
                # We can return any of them.
                # Let's return the first one.
                for k in range(16):
                    if (m >> k) & 1:
                        u, v = edge_pairs[k]
                        return (u, v)
    return None

def _try_add_edge(adj_masks, u, v):
    """
    Checks if adding edge (u, v) keeps the graph Q3-free.
    Returns True if valid, False otherwise.
    """
    # Temporarily add edge
    adj_masks[u] |= (1 << v)
    adj_masks[v] |= (1 << u)
    
    # Check Q3
    is_q3_free = not _contains_q3(adj_masks)
    
    # Revert
    adj_masks[u] &= ~(1 << v)
    adj_masks[v] &= ~(1 << u)
    
    return is_q3_free

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    best_adj = None
    best_count = 0
    
    # We will run several local search iterations
    # Each iteration starts from a random dense graph or K16
    # and performs hill climbing.
    
    num_restarts = 20 # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize adjacency masks
        # Start with K16 for the first few, then random dense
        adj_masks = [0] * n
        if restart < 5:
            # K16
            for i in range(n):
                for j in range(i+1, n):
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        else:
            # Random graph with p=0.8
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.8:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
        
        # Phase 1: Remove edges to destroy all Q3s
        # We do this greedily.
        while time.time() < deadline - 5:
            edge = _find_q3_edge(adj_masks)
            if edge is None:
                break
            u, v = edge
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
        # Check if we have a valid graph
        if _contains_q3(adj_masks):
            # If we ran out of time in removal, this graph is invalid.
            # Skip this restart.
            continue
            
        # Count edges
        count = 0
        for i in range(n):
            count += bin(adj_masks[i]).count('1')
        count //= 2
        
        if count > best_count:
            best_count = count
            best_adj = adj_masks[:]
            
        # Phase 2: Hill Climbing - Add edges back
        # Try adding all missing edges
        missing_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj_masks[i] >> j) & 1:
                    missing_edges.append((i, j))
        
        random.shuffle(missing_edges)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u, v in missing_edges:
                if time.time() > deadline - 5:
                    break
                # Check if edge is already present
                if (adj_masks[u] >> v) & 1:
                    continue
                if _try_add_edge(adj_masks, u, v):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    improved = True
                    missing_edges.remove((u, v)) # Optional, list is small
            
            # After a full pass, shuffle again to try different orders?
            # Or just break if no improvement.
            if not improved:
                break
                
        # Update best
        count = 0
        for i in range(n):
            count += bin(adj_masks[i]).count('1')
        count //= 2
        
        if count > best_count:
            best_count = count
            best_adj = adj_masks[:]

    # If best_adj is None, return empty or a known valid graph
    if best_adj is None:
        # Fallback to empty graph
        best_adj = [0] * n
        best_count = 0
    else:
        # Ensure best_adj is a copy
        best_adj = best_adj[:]

    # Convert to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
