# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

M = 205

def _compute_forbidden():
    forbidden = set()
    for x in range(1, M):
        forbidden.add(pow(x, 2, M))
    forbidden.discard(0)
    return forbidden

FORBIDDEN = _compute_forbidden()

# Precompute conflict bitmasks
# conflict_mask[i] is a bitmask where bit j is 1 if i and j conflict (i.e., (i-j) % M is in FORBIDDEN)
# Note: We only need to check if candidate is in conflict with ANY selected element.
# So, for a selected set S (bitmask), we can compute the union of conflicts:
# conflict_union = OR(conflict_mask[s] for s in S)
# Then a candidate c is valid if bit c in conflict_union is 0.

CONFLICT_MASKS = [0] * M
for a in range(M):
    mask = 0
    for b in range(M):
        if a != b and (a - b) % M in FORBIDDEN:
            mask |= (1 << b)
    CONFLICT_MASKS[a] = mask

def _count_bits(mask):
    return bin(mask).count('1')

def _random_bit(mask):
    """Pick a random index where bit is 1 in mask. Returns -1 if mask is 0."""
    if mask == 0:
        return -1
    # Get list of set bits
    bits = []
    m = mask
    i = 0
    while m:
        if m & 1:
            bits.append(i)
        m >>= 1
        i += 1
    return random.choice(bits)

def _simulate_annealing(start_mask, deadline, initial_temp=10.0, cooling_rate=0.9999, min_temp=0.01):
    """
    Simulated annealing to maximize the number of bits in the mask.
    """
    current_mask = start_mask
    current_size = _count_bits(current_mask)
    best_mask = current_mask
    best_size = current_size
    
    temp = initial_temp
    
    # Precompute list of all elements for random selection
    all_elements = list(range(M))
    
    while temp > min_temp:
        if time.time() > deadline:
            break
        
        # Generate a move:
        # Strategy: 
        # 1. Try to add a random element that is compatible.
        # 2. If can't add, try to remove a random element and see if we can add a different one (swap).
        # 3. Occasionally just remove an element (to allow escape, though usually bad for size).
        
        # Calculate current conflict union
        conflict_union = 0
        m = current_mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            conflict_union |= CONFLICT_MASKS[idx]
            m ^= lsb
        
        # Identify available elements (not in current_mask and not in conflict_union)
        # available_mask = ~ (current_mask | conflict_union) & ((1 << M) - 1)
        all_bits = (1 << M) - 1
        available_mask = all_bits & ~(current_mask | conflict_union)
        
        move_type = random.random()
        
        if move_type < 0.7 and available_mask != 0:
            # Try to add a random available element
            cand = _random_bit(available_mask)
            if cand != -1:
                new_mask = current_mask | (1 << cand)
                new_size = current_size + 1
                # Always accept improvement
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
        elif move_type < 0.9:
            # Try a swap: remove one random element from current, then try to add a random element from new available
            if current_mask != 0:
                rem = _random_bit(current_mask)
                if rem != -1:
                    temp_mask = current_mask & ~(1 << rem)
                    # Recalculate conflict union for temp_mask
                    # Optimization: We can just recompute or update. Recomputing is O(k) which is small.
                    conflict_union_tmp = 0
                    m = temp_mask
                    while m:
                        lsb = m & -m
                        idx = (lsb.bit_length() - 1)
                        conflict_union_tmp |= CONFLICT_MASKS[idx]
                        m ^= lsb
                    
                    available_tmp = all_bits & ~(temp_mask | conflict_union_tmp)
                    # We must NOT add the removed element back immediately (that's just a no-op swap)
                    # But adding it back is a valid move (size unchanged), which helps mixing.
                    # However, to find improvements, we want to add something else if possible.
                    
                    if available_tmp != 0:
                        # Pick a random candidate from available_tmp
                        cand = _random_bit(available_tmp)
                        if cand != -1:
                            new_mask = temp_mask | (1 << cand)
                            new_size = _count_bits(new_mask)
                            delta = new_size - current_size
                            
                            # Metropolis criterion
                            if delta > 0 or random.random() < math.exp(delta / temp):
                                current_mask = new_mask
                                current_size = new_size
                                if current_size > best_size:
                                    best_mask = current_mask
                                    best_size = current_size
        else:
            # Random removal (to escape deep local minima, though rare)
            if current_mask != 0:
                rem = _random_bit(current_mask)
                if rem != -1:
                    new_mask = current_mask & ~(1 << rem)
                    new_size = current_size - 1
                    delta = new_size - current_size
                    if random.random() < math.exp(delta / temp):
                        current_mask = new_mask
                        current_size = new_size
        
        temp *= cooling_rate
        
    return best_mask

def _greedy_init():
    """
    Fast greedy initialization using random order.
    """
    elements = list(range(M))
    random.shuffle(elements)
    mask = 0
    conflict_union = 0
    
    for e in elements:
        # Check if e is compatible with current mask
        # e is compatible if bit e in conflict_union is 0
        if not (conflict_union & (1 << e)):
            mask |= (1 << e)
            conflict_union |= CONFLICT_MASKS[e]
    return mask

def furstenberg_sarkozy_205() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin for 630s limit
    
    random.seed(42)
    
    best_mask = 0
    best_size = 0
    
    # Phase 1: Greedy Initialization + Polish
    # Run a few greedy starts to find a good initial candidate
    for i in range(10):
        if time.time() > deadline - 10:
            break
        mask = _greedy_init()
        size = _count_bits(mask)
        if size > best_size:
            best_mask = mask
            best_size = size
            
    # Phase 2: Simulated Annealing from best and random restarts
    # We allocate time for multiple SA runs
    # Each SA run has a budget. If time is running out, we stop.
    
    # Determine how many full SA runs we can do based on a rough estimate
    # Or just loop until deadline, with each run having a soft limit or just checking deadline inside.
    # The SA function checks deadline internally.
    
    # Strategy: 
    # 1. Run SA on the best greedy result.
    # 2. Run SA on other good greedy results.
    # 3. Run SA from random masks if time permits? No, greedy is better start.
    
    # Let's keep a list of top greedy starts
    greedy_starts = []
    for i in range(50):
        if time.time() > deadline - 20:
            break
        mask = _greedy_init()
        size = _count_bits(mask)
        greedy_starts.append((size, mask))
        
    # Sort by size descending
    greedy_starts.sort(key=lambda x: x[0], reverse=True)
    
    # Run SA on the top few starts
    for i, (size, mask) in enumerate(greedy_starts[:10]):
        if time.time() > deadline - 5:
            break
        
        # If we already have a very large set, maybe skip smaller starts?
        # But SA might improve them more.
        
        # Adjust parameters based on remaining time?
        # For now, use standard params.
        
        res_mask = _simulate_annealing(mask, deadline, initial_temp=5.0, cooling_rate=0.99995)
        res_size = _count_bits(res_mask)
        
        if res_size > best_size:
            best_mask = res_mask
            best_size = res_size
            
    # Phase 3: Final polish on the absolute best found
    if time.time() < deadline - 1:
        # Run one more SA on the best found, with higher temperature to explore more?
        # Or lower temperature to refine?
        # Let's try a "restart" on the best mask with a slightly higher temp to see if we can jump.
        res_mask = _simulate_annealing(best_mask, deadline, initial_temp=10.0, cooling_rate=0.9999)
        res_size = _count_bits(res_mask)
        if res_size > best_size:
            best_mask = res_mask
            best_size = res_size

    # Convert mask to array
    result = []
    m = best_mask
    i = 0
    while m:
        if m & 1:
            result.append(i)
        m >>= 1
        i += 1
        
    if not result:
        # Fallback: return a single element (0 is always valid? 0-0=0 not in F. 0-x = -x. Is -x in F? 
        # F contains squares. -x is square iff x is square. 
        # If we return [0], no pairs. Valid.
        return np.array([0], dtype=np.int64)
        
    return np.array(sorted(result), dtype=np.int64)

# EVOLVE-BLOCK-END
