import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Try greedy construction for various target sizes
    for target_n in [21, 20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 5:
            break
        
        # Try multiple random seeds for the greedy process
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(seed)
            result = _greedy_construct(target_n, rng, deadline)
            if result is not None:
                return result
    
    # Fallback: trivially valid small graph
    return {'n': 5, 'edges': []}

def _greedy_construct(target_n, rng, deadline):
    """
    Greedily build a graph on target_n vertices, one vertex at a time.
    For each new vertex, try all 2^k possible edge connections to existing
    vertices and pick the one that minimizes new regular induced subgraphs.
    """
    n = target_n
    adj_masks = [0] * n
    edges = []
    
    for new_v in range(n):
        if time.time() > deadline - 3:
            return None
        
        existing = new_v  # number of existing vertices
        if existing == 0:
            continue
        
        # Try all possible edge sets from new_v to existing vertices
        # For efficiency, we'll sample if 2^existing is too large
        if existing <= 12:
            candidates = list(range(1 << existing))
            rng.shuffle(candidates)
        else:
            # Sample a subset of candidates
            num_samples = min(1000, 1 << existing)
            candidates = [rng.randint(0, (1 << existing) - 1) for _ in range(num_samples)]
        
        best_mask = None
        best_score = float('inf')
        
        for mask in candidates:
            if time.time() > deadline - 3:
                return None
            
            # Set up the edges for this candidate
            adj_masks[new_v] = mask
            for i in range(existing):
                if mask & (1 << i):
                    adj_masks[i] |= (1 << new_v)
            
            # Count new violations: subsets of size >= 6 that include new_v
            # and are regular
            score = 0
            # We only need to check subsets that include new_v
            # and have size >= 6
            min_subset_size = 6
            max_subset_size = new_v + 1
            
            # Check subsets of size exactly 6 first (most common)
            # For larger sizes, check fewer
            
            # To be efficient, check sizes from 6 to min(new_v+1, 8)
            # and assume larger sizes are less likely
            check_max_size = min(new_v + 1, 10)
            
            for size in range(min_subset_size, check_max_size + 1):
                if new_v + 1 < size:
                    break
                # Choose size-1 vertices from existing
                existing_indices = list(range(existing))
                for subset_existing in combinations(existing_indices, size - 1):
                    # Build the full subset
                    full_subset = list(subset_existing) + [new_v]
                    mask_full = 0
                    for v in full_subset:
                        mask_full |= (1 << v)
                    
                    # Check if regular
                    degs = []
                    for v in full_subset:
                        degs.append((adj_masks[v] & mask_full).bit_count())
                    
                    if len(set(degs)) == 1:
                        score += 1
                        if score >= best_score:
                            break
                if score >= best_score:
                    break
            
            # Undo the edges
            adj_masks[new_v] = 0
            for i in range(existing):
                if mask & (1 << i):
                    adj_masks[i] &= ~(1 << new_v)
            
            if score < best_score:
                best_score = score
                best_mask = mask
                
                if score == 0:
                    break
        
        if best_mask is None:
            best_mask = 0
        
        # Apply the best mask
        adj_masks[new_v] = best_mask
        for i in range(existing):
            if best_mask & (1 << i):
                adj_masks[i] |= (1 << new_v)
                edges.append([i, new_v])
    
    # Final verification
    if _is_valid_fast(adj_masks, n, deadline):
        return {'n': n, 'edges': edges}
    
    return None

def _is_valid_fast(adj_masks, n, deadline):
    """
    Check if no subset of size >= 6 induces a regular graph.
    Returns True if valid, False if violation found or timeout.
    """
    if n < 6:
        return True

    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if time.time() > deadline:
                return False
            
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            degs = []
            for v in subset:
                degs.append((adj_masks[v] & mask).bit_count())
            
            if len(set(degs)) == 1:
                return False

    return True
