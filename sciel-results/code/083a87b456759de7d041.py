import time
import random
import math
import numpy as np

N_VERT = 12
M = 40
TIME_LIMIT = 75  # seconds, leave 15s margin for 90s wall-clock limit

def _count_pairs(masks):
    """Count comparable pairs in a list of integer masks."""
    count = 0
    m = len(masks)
    for i in range(m):
        a = masks[i]
        for j in range(i + 1, m):
            b = masks[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _count_pairs_np(masks_np):
    """Count comparable pairs using NumPy vectorization."""
    n = len(masks_np)
    a = masks_np[:, np.newaxis]
    b = masks_np[np.newaxis, :]
    ab = a & b
    comparable = (ab == a) | (ab == b)
    # Exclude diagonal
    np.fill_diagonal(comparable, False)
    return int(np.sum(comparable) // 2)

def _delta_for_swap_np(masks_np, i, new_mask):
    """Compute change in comparable-pair count when masks[i] is replaced by new_mask, using NumPy."""
    old = masks_np[i]
    if new_mask == old:
        return 0
    others = np.delete(masks_np, i)
    # Old contributions
    ab_old = others & old
    comp_old = (ab_old == others) | (ab_old == old)
    # New contributions
    ab_new = others & new_mask
    comp_new = (ab_new == others) | (ab_new == new_mask)
    return int(np.sum(comp_new) - np.sum(comp_old))

def _best_replacement_np(masks_np, i, all_masks_np, current_set):
    """Find the best replacement for position i using vectorized evaluation."""
    old = masks_np[i]
    others = np.delete(masks_np, i)
    others_set = set(others.tolist())
    
    # Precompute old contributions
    ab_old = others & old
    comp_old = (ab_old == others) | (ab_old == old)
    old_contrib = int(np.sum(comp_old))
    
    best_delta = 0
    best_new = None
    
    # Filter out masks already in current set
    available = []
    for mask in all_masks_np:
        if mask not in current_set:
            available.append(mask)
    
    if not available:
        return 0, None
    
    available_np = np.array(available, dtype=np.int64)
    
    # Vectorize: for each candidate, compute comparable count with others
    # Shape: (len(available), len(others))
    ab = available_np[:, np.newaxis] & others[np.newaxis, :]
    comp = (ab == available_np[:, np.newaxis]) | (ab == others[np.newaxis, :])
    new_contribs = comp.sum(axis=1)
    
    deltas = new_contribs - old_contrib
    max_idx = np.argmax(deltas)
    max_delta = int(deltas[max_idx])
    
    if max_delta > 0:
        return max_delta, int(available_np[max_idx])
    return 0, None

def _greedy_init():
    """Greedy construction: build a dense comparable structure by adding masks that maximize marginal gain."""
    all_masks = list(range(1 << N_VERT))
    random.seed(42)
    
    # Start with a few well-chosen seeds to create a base structure
    # Use masks of varying sizes to create comparability
    selected = []
    
    # Add empty set and full set as anchors
    selected.append(0)
    selected.append((1 << N_VERT) - 1)
    
    # Greedily add masks that maximize comparable pairs with existing set
    for _ in range(M - 2):
        best_mask = None
        best_gain = -1
        
        # Sample a subset of candidates for speed (not all 4096)
        candidates = random.sample(all_masks, min(200, len(all_masks)))
        
        for mask in candidates:
            if mask in selected:
                continue
            # Count how many existing sets are comparable with this mask
            gain = 0
            for existing in selected:
                if (mask & existing) == mask or (mask & existing) == existing:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        
        if best_mask is not None:
            selected.append(best_mask)
    
    return selected

def _local_search(masks, deadline):
    """Vectorized local search: try all possible replacements for each position using NumPy."""
    current = list(masks)
    current_np = np.array(current, dtype=np.int64)
    current_score = _count_pairs_np(current_np)
    best = list(current)
    best_score = current_score
    
    all_masks_np = np.arange(1 << N_VERT, dtype=np.int64)
    
    while time.time() < deadline:
        improved = False
        current_set = set(current)
        
        for i in range(M):
            old = current[i]
            
            # Find best replacement using vectorized computation
            best_delta, best_new = _best_replacement_np(current_np, i, all_masks_np, current_set)
            
            if best_new is not None and best_delta > 0:
                current[i] = best_new
                current_np[i] = best_new
                current_set.remove(old)
                current_set.add(best_new)
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
        
        if not improved:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initialize with structured greedy
    masks = _greedy_init()
    
    # Local search with vectorized delta computation
    best_masks, best_score = _local_search(masks, deadline)
    
    # Convert to list of lists of ints
    result = []
    for mask in best_masks:
        result.append([b for b in range(N_VERT) if (mask >> b) & 1])
    
    return result
