import time
import random
import numpy as np
from itertools import combinations

# Precompute all K4s and their edge indices
def _precompute_k4s():
    n = 16
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            idx += 1
    
    k4_list = []
    # Also map edge -> list of k4 indices containing it
    edge_to_k4s = [[] for _ in range(120)]
    
    k4_idx = 0
    for quad in combinations(range(n), 4):
        edges = []
        for a, b in combinations(quad, 2):
            e_idx = edge_index[(a, b)]
            edges.append(e_idx)
            edge_to_k4s[e_idx].append(k4_idx)
        k4_list.append(tuple(edges))
        k4_idx += 1
        
    return k4_list, edge_to_k4s

_K4S = None
_EDGE_TO_K4S = None

def _ensure_precomputed():
    global _K4S, _EDGE_TO_K4S
    if _K4S is None:
        _K4S, _EDGE_TO_K4S = _precompute_k4s()

def _popcount(x):
    return bin(x).count('1')

def _init_k4_masks(colors):
    """
    Initialize masks and counts for all K4s.
    """
    n_k4 = len(_K4S)
    max_colors = 16 # Assume max 15 colors used, so 0..15 fits in 16 bits
    masks = [0] * n_k4
    counts = [[0] * max_colors for _ in range(n_k4)]
    distinct = [0] * n_k4
    
    for k in range(n_k4):
        edges = _K4S[k]
        for e in edges:
            c = colors[e]
            if counts[k][c] == 0:
                masks[k] |= (1 << c)
                distinct[k] += 1
            counts[k][c] += 1
            
    return masks, counts, distinct

def _is_valid_fast(masks, distinct):
    """Check if all K4s have at least 5 distinct colors."""
    for d in distinct:
        if d < 5:
            return False
    return True

def _try_merge(colors, c1, c2, masks, counts, distinct):
    """
    Try to merge color c2 into c1.
    Returns True if valid, and updates structures if successful.
    If unsuccessful, restores original state.
    """
    # Find all edges with color c2
    edges_c2 = [i for i, c in enumerate(colors) if c == c2]
    
    if not edges_c2:
        return True # Nothing to merge
    
    # Save original state for rollback
    orig_colors = colors.copy()
    orig_masks = masks.copy()
    orig_counts = [row[:] for row in counts]
    orig_distinct = distinct[:]
    
    # Apply merge: change all c2 to c1
    for e in edges_c2:
        colors[e] = c1
        
    # Update K4 structures
    valid = True
    for e in edges_c2:
        for k in _EDGE_TO_K4S[e]:
            # Update count for c2 (decrease)
            counts[k][c2] -= 1
            if counts[k][c2] == 0:
                # Color c2 is gone from this K4
                masks[k] &= ~(1 << c2)
                distinct[k] -= 1
            
            # Update count for c1 (increase)
            if counts[k][c1] == 0:
                masks[k] |= (1 << c1)
                distinct[k] += 1
            counts[k][c1] += 1
            
            # Check validity
            if distinct[k] < 5:
                valid = False
    
    if valid:
        return True
    else:
        # Rollback
        colors[:] = orig_colors
        masks[:] = orig_masks
        counts[:] = orig_counts
        distinct[:] = orig_distinct
        return False

def _local_search_fix(colors, max_colors, max_iters=1000):
    """
    Local search to fix violations in a coloring.
    Returns (colors, valid)
    """
    masks, counts, distinct = _init_k4_masks(colors)
    if _is_valid_fast(masks, distinct):
        return colors, True
            
    for _ in range(max_iters):
        # Find a violating K4
        viol_k = -1
        for k in range(len(_K4S)):
            if distinct[k] < 5:
                viol_k = k
                break
            
        if viol_k == -1:
            return colors, True
                
        # Pick a random edge in this K4
        k4_edges = _K4S[viol_k]
        e = random.choice(k4_edges)
        old_c = colors[e]
        
        # Try changing to a random color
        new_c = random.randint(0, max_colors - 1)
        if new_c == old_c:
            continue
                
        colors[e] = new_c
            
        # Update structures
        for k in _EDGE_TO_K4S[e]:
            counts[k][old_c] -= 1
            if counts[k][old_c] == 0:
                masks[k] &= ~(1 << old_c)
                distinct[k] -= 1
                    
            if counts[k][new_c] == 0:
                masks[k] |= (1 << new_c)
                distinct[k] += 1
            counts[k][new_c] += 1
            
        # Check if valid
        if _is_valid_fast(masks, distinct):
            return colors, True
        else:
            # Revert
            colors[e] = old_c
            for k in _EDGE_TO_K4S[e]:
                counts[k][new_c] -= 1
                if counts[k][new_c] == 0:
                    masks[k] &= ~(1 << new_c)
                    distinct[k] -= 1
                        
                if counts[k][old_c] == 0:
                    masks[k] |= (1 << old_c)
                    distinct[k] += 1
                counts[k][old_c] += 1
                
    return colors, False

def construct():
    _ensure_precomputed()
    n_edges = 120
    n = 16
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Find an initial valid coloring with 15 colors
    best_colors = None
    best_num_colors = 16
    
    # Try to find a valid 15-coloring
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
            
        # Initialize with (i+j) % 15 plus random noise
        colors = np.array([(i + j) % 15 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        if attempt > 0:
            for _ in range(attempt * 5):
                e = random.randint(0, 119)
                colors[e] = random.randint(0, 14)
                
        colors, valid = _local_search_fix(colors, 15, max_iters=500)
        
        if valid:
            best_colors = colors.copy()
            best_num_colors = 15
            break
    else:
        # Fallback to 17 colors if 15 failed
        colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        best_colors = colors.copy()
        best_num_colors = 17
        
    if best_colors is None:
        return [0] * 120

    # 2. Try to reduce colors from current down to 13
    current_colors = best_colors
    current_num_colors = best_num_colors
    
    # Prepare structures for current_colors
    masks, counts, distinct = _init_k4_masks(current_colors)
    
    while current_num_colors > 13:
        if time.time() > deadline - 10:
            break
            
        found_merge = False
        # Try all pairs of colors to merge
        for c1 in range(current_num_colors):
            for c2 in range(c1 + 1, current_num_colors):
                if time.time() > deadline - 5:
                    break
                
                if _try_merge(current_colors, c1, c2, masks, counts, distinct):
                    # Merge successful
                    # Remap colors: remove c2, shift c > c2 down by 1
                    new_colors = []
                    for c in current_colors:
                        if c == c2:
                            new_colors.append(c1)
                        elif c > c2:
                            new_colors.append(c - 1)
                        else:
                            new_colors.append(c)
                    current_colors = np.array(new_colors, dtype=np.int32)
                    current_num_colors -= 1
                    found_merge = True
                    
                    # Re-initialize structures for the new palette
                    masks, counts, distinct = _init_k4_masks(current_colors)
                    break
            
            if found_merge:
                break
                
        if not found_merge:
            # No direct merge worked. Try to find a different valid coloring with current_num_colors
            # where a merge might work.
            improved = False
            for _ in range(10): # 10 restarts
                if time.time() > deadline - 5:
                    break
                    
                # Perturb
                perturbed = current_colors.copy()
                for _ in range(5):
                    e = random.randint(0, 119)
                    perturbed[e] = random.randint(0, current_num_colors - 1)
                    
                # Fix
                perturbed, valid = _local_search_fix(perturbed, current_num_colors, max_iters=500)
                
                if valid:
                    # Check if merge works on this new valid coloring
                    masks_m, counts_m, distinct_m = _init_k4_masks(perturbed)
                    
                    merge_found = False
                    for c1 in range(current_num_colors):
                        for c2 in range(c1 + 1, current_num_colors):
                            if _try_merge(perturbed, c1, c2, masks_m, counts_m, distinct_m):
                                # Success
                                new_colors = []
                                for c in perturbed:
                                    if c == c2:
                                        new_colors.append(c1)
                                    elif c > c2:
                                        new_colors.append(c - 1)
                                    else:
                                        new_colors.append(c)
                                current_colors = np.array(new_colors, dtype=np.int32)
                                current_num_colors -= 1
                                improved = True
                                merge_found = True
                                break
                        if merge_found:
                            break
                            
                    if improved:
                        # Re-init main structures
                        masks, counts, distinct = _init_k4_masks(current_colors)
                        break
                
            if not improved:
                break # Stuck, break out of while loop
                
    # Final remap to 0..k-1
    unique_colors = sorted(set(current_colors.tolist()))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    final_colors = [color_map[c] for c in current_colors.tolist()]
    
    return final_colors
