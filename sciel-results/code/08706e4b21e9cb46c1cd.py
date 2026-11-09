import time
import random
import math

N = 12
M = 40
FULL_MASK = (1 << N) - 1

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

def _is_comparable(a, b):
    """Check if two masks are comparable (one is subset of other)."""
    return (a & b) == a or (a & b) == b

def _incremental_score(masks, idx, new_mask):
    """Compute score change when replacing masks[idx] with new_mask.
    Returns the delta in comparable pair count."""
    old_mask = masks[idx]
    delta = 0
    for j in range(M):
        if j == idx:
            continue
        c = masks[j]
        old_comp = 1 if _is_comparable(old_mask, c) else 0
        new_comp = 1 if _is_comparable(new_mask, c) else 0
        delta += new_comp - old_comp
    return delta

def _greedy_construct():
    """Greedy construction of 40 masks."""
    chosen = []
    all_masks = list(range(1 << N))
    
    for _ in range(M):
        best_mask = -1
        best_score = -1
        for mask in all_masks:
            if mask in chosen:
                continue
            score = 0
            for c in chosen:
                if (mask & c) == mask or (mask & c) == c:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        chosen.append(best_mask)
        
    return chosen

def _full_scan_local_search(masks, deadline=None):
    """Full-scan local search: for each position, try all 4096 masks to find best replacement.
    Optimized with precomputed comparability counts."""
    current_score = _count_pairs(masks)
    all_masks = list(range(1 << N))
    mask_set = set(masks)
    improved = True
    
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for idx in range(M):
            if deadline and time.time() > deadline:
                break
            old_mask = masks[idx]
            best_delta = 0
            best_new_mask = -1
            
            # Precompute which masks are comparable with each other
            # For efficiency, we can cache the comparability with new_mask
            for new_mask in all_masks:
                if new_mask == old_mask:
                    continue
                if new_mask in mask_set:
                    continue
                
                # Quick check: if new_mask has same popcount as old_mask, 
                # it's less likely to be comparable, but still check
                delta = 0
                for j in range(M):
                    if j == idx:
                        continue
                    c = masks[j]
                    old_comp = 1 if _is_comparable(old_mask, c) else 0
                    new_comp = 1 if _is_comparable(new_mask, c) else 0
                    delta += new_comp - old_comp
                    # Early termination: if delta already exceeds best_delta + max possible remaining
                    # Max remaining is (M - idx - 1) if we're going forward, but this is approximate
                    if delta > best_delta + (M - 1):
                        break
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            if best_new_mask >= 0:
                mask_set.remove(old_mask)
                mask_set.add(best_new_mask)
                masks[idx] = best_new_mask
                current_score += best_delta
                improved = True
    return masks

def _simulated_annealing(masks, deadline=None):
    """Simulated annealing with structured neighborhood moves."""
    current_score = _count_pairs(masks)
    best_score = current_score
    best_masks = list(masks)
    all_masks = list(range(1 << N))
    mask_set = set(masks)
    
    T = 3.0
    T_min = 0.005
    cooling = 0.9992
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if T < T_min:
            T = 1.5
            cooling = 0.9995
            no_improve_count = 0
        
        idx = random.randint(0, M - 1)
        old_mask = masks[idx]
        
        # Generate candidate: 60% random, 40% neighbor (flip 1-3 bits)
        if random.random() < 0.4:
            # Neighbor: flip 1 to 3 random bits
            flips = random.randint(1, 3)
            new_mask = old_mask
            for _ in range(flips):
                bit = random.randint(0, N - 1)
                new_mask ^= (1 << bit)
        else:
            new_mask = random.choice(all_masks)
        
        if new_mask == old_mask:
            continue
        if new_mask in mask_set:
            continue
        
        delta = _incremental_score(masks, idx, new_mask)
        
        if delta >= 0 or random.random() < math.exp(delta / T):
            mask_set.remove(old_mask)
            mask_set.add(new_mask)
            masks[idx] = new_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = list(masks)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        T *= cooling
        
        # If stuck for too long, restart with higher temperature
        if no_improve_count > max_no_improve:
            T = 2.0
            no_improve_count = 0
    
    return best_masks

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    random.seed(42)
    
    # 1. Greedy initialization
    masks = _greedy_construct()
    
    # 2. Full-scan local search (exhaustive per position)
    masks = _full_scan_local_search(masks, deadline=deadline - 15)
    
    # 3. Simulated annealing to escape local optima
    sa_deadline = deadline - 10
    if time.time() < sa_deadline:
        masks = _simulated_annealing(masks, deadline=sa_deadline)
    
    # 4. Final full-scan polish
    if time.time() < deadline - 3:
        masks = _full_scan_local_search(masks, deadline=deadline - 2)
    
    # 5. Second simulated annealing pass with different seed
    if time.time() < deadline - 2:
        random.seed(123)
        masks = _simulated_annealing(masks, deadline=deadline - 1)
    
    # 6. Final local search polish
    if time.time() < deadline - 0.5:
        masks = _full_scan_local_search(masks, deadline=deadline - 0.1)
    
    # Convert masks to lists of vertices
    result = []
    for mask in masks:
        result.append([b for b in range(N) if (mask >> b) & 1])
        
    return result
