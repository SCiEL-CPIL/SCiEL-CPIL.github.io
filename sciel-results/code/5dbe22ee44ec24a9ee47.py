import time
import random

P = {'n': 12, 'm': 40}

def _count_pairs(masks):
    """Count comparable pairs in a list of masks."""
    count = 0
    m = len(masks)
    for i in range(m):
        a = masks[i]
        for j in range(i + 1, m):
            b = masks[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _greedy_construct():
    """Greedy construction of 40 masks."""
    n = P['n']
    m = P['m']
    chosen = []
    all_masks = list(range(1 << n))
    
    for _ in range(m):
        best_mask = -1
        best_score = -1
        for mask in all_masks:
            if mask in chosen:
                continue
            # Count comparables with existing chosen
            score = 0
            for c in chosen:
                if (mask & c) == mask or (mask & c) == c:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        chosen.append(best_mask)
        
    return chosen

def _local_search(masks, max_iter=5000, deadline=None):
    """Local search to improve the set of masks."""
    n = P['n']
    m = P['m']
    current_score = _count_pairs(masks)
    all_masks = list(range(1 << n))
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random index to swap
        idx = random.randint(0, m - 1)
        old_mask = masks[idx]
        
        # Pick a random new mask not in the set
        new_mask = random.choice(all_masks)
        if new_mask in masks:
            continue
            
        # Try the swap
        masks[idx] = new_mask
        new_score = _count_pairs(masks)
        
        if new_score > current_score:
            current_score = new_score
        else:
            # Revert
            masks[idx] = old_mask
            
    return masks

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Set random seed for reproducibility if needed, but task doesn't specify
    random.seed(42)
    
    # 1. Greedy initialization
    masks = _greedy_construct()
    
    # 2. Local search
    masks = _local_search(masks, max_iter=20000, deadline=deadline)
    
    # Convert masks to lists of vertices
    result = []
    for mask in masks:
        result.append([b for b in range(P['n']) if (mask >> b) & 1])
        
    return result
