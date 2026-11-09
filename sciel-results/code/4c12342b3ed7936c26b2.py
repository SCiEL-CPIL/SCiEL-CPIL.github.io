import time
import random
import math

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

def _delta_for_swap(masks, i, new_mask):
    """Compute change in comparable-pair count when masks[i] is replaced by new_mask."""
    old = masks[i]
    if new_mask == old:
        return 0
    delta = 0
    for j in range(M):
        if j == i:
            continue
        m = masks[j]
        # Remove contribution of (old, m)
        if (old & m) == old or (old & m) == m:
            delta -= 1
        # Add contribution of (new_mask, m)
        if (new_mask & m) == new_mask or (new_mask & m) == m:
            delta += 1
    return delta

def _greedy_chain_init():
    """
    Initialize with a structure containing multiple chains.
    We create several chains of varying lengths to maximize initial comparability.
    """
    masks = set()
    # We want 40 distinct masks.
    # Let's create 10 chains of length 4? 10*4=40.
    # Or mix sizes.
    # A better heuristic: start with empty and full, and fill in between with random extensions?
    # Let's try to construct disjoint chains in the lattice.
    # Chain 1: 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095 (length 13)
    # We can't have disjoint chains easily covering 40 sets without overlap or careful selection.
    
    # Simpler: Randomly generate 40 masks, but bias towards chains.
    # Actually, just random is fine if SA is good, but a better start helps.
    # Let's use a "greedy add" approach for initialization.
    # Start with empty set.
    current_masks = [0]
    random.seed(42)
    
    # Precompute all masks
    all_masks = list(range(1 << N_VERT))
    random.shuffle(all_masks)
    
    # Greedily add masks that maximize marginal gain
    # This is O(M * 4096 * M) which is 40 * 4096 * 40 ~ 6.5M ops. Fast enough.
    while len(current_masks) < M:
        best_gain = -1
        best_mask = None
        current_set = set(current_masks)
        
        # Sample a subset of candidates to speed up init? Or just iterate all.
        # Iterating all 4096 for 40 additions is 163,840 checks. Each check is O(40). Total ~6.5M.
        for m in all_masks:
            if m in current_set:
                continue
            # Calculate gain of adding m
            gain = 0
            for existing in current_masks:
                if (m & existing) == m or (m & existing) == existing:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = m
                
        if best_mask is not None:
            current_masks.append(best_mask)
        else:
            # Should not happen
            break
            
    return current_masks

def _simulated_annealing(masks, deadline):
    """
    Simulated Annealing to maximize comparable pairs.
    """
    current = list(masks)
    current_set = set(current)
    current_score = _count_pairs(current)
    
    best = list(current)
    best_score = current_score
    
    # SA Parameters
    T = 10.0  # Initial temperature
    alpha = 0.995  # Cooling rate
    min_T = 0.1
    
    # Precompute all masks for random selection
    all_masks = list(range(1 << N_VERT))
    
    start_time = time.time()
    last_improve_time = start_time
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a random index to change
        i = random.randint(0, M - 1)
        old_mask = current[i]
        
        # Pick a random new mask
        # To avoid bias, pick from all masks.
        new_mask = random.choice(all_masks)
        
        # Check distinctness
        if new_mask in current_set and new_mask != old_mask:
            continue
            
        # Calculate delta
        delta = _delta_for_swap(current, i, new_mask)
        
        # Acceptance criterion
        accept = False
        if delta > 0:
            accept = True
        else:
            # Metropolis criterion
            if T > 0:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    accept = True
            
        if accept:
            # Update current state
            current_set.remove(old_mask)
            current_set.add(new_mask)
            current[i] = new_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best = list(current)
                last_improve_time = time.time()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # If stagnating for a long time, restart from best
            if no_improve_count > 5000 and time.time() - last_improve_time > 5.0:
                # Restart
                current = list(best)
                current_set = set(current)
                current_score = best_score
                no_improve_count = 0
                # Slight reheat
                T = max(T, 5.0)
                
        # Cool down
        T *= alpha
        if T < min_T:
            T = min_T

    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initialize
    masks = _greedy_chain_init()
    
    # Simulated Annealing
    best_masks, best_score = _simulated_annealing(masks, deadline)
    
    # Convert to list of lists of ints
    result = []
    for mask in best_masks:
        result.append([b for b in range(N_VERT) if (mask >> b) & 1])
    
    return result
