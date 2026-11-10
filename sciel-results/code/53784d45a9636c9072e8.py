import time
import random

N = 256
DEADLINE = 85.0  # Safety margin before 90s limit

def _precompute():
    """Precompute 3-AP and 4-AP information using bitmasks."""
    # 3-APs: (a, a+d, a+2d)
    three_aps = []
    three_aps_by_v = [[] for _ in range(N + 1)]
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            idx = len(three_aps)
            three_aps.append(mask)
            
            # Store the "other" elements for each vertex in the 3-AP
            # For vertex a, others are b, c
            three_aps_by_v[a].append((1 << (b-1)) | (1 << (c-1)))
            # For vertex b, others are a, c
            three_aps_by_v[b].append((1 << (a-1)) | (1 << (c-1)))
            # For vertex c, others are a, b
            three_aps_by_v[c].append((1 << (a-1)) | (1 << (b-1)))
    
    # 4-APs: (a, a+d, a+2d, a+3d)
    four_aps = []
    four_aps_by_v = [[] for _ in range(N + 1)]
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            idx = len(four_aps)
            four_aps.append(mask)
            
            # Store the "other" elements for each vertex in the 4-AP
            four_aps_by_v[a].append((1 << (b-1)) | (1 << (c-1)) | (1 << (e-1)))
            four_aps_by_v[b].append((1 << (a-1)) | (1 << (c-1)) | (1 << (e-1)))
            four_aps_by_v[c].append((1 << (a-1)) | (1 << (b-1)) | (1 << (e-1)))
            four_aps_by_v[e].append((1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)))
            
    return three_aps, four_aps, four_aps_by_v, three_aps_by_v

def _can_add(mask, v, four_aps_by_v):
    """Check if adding v to the set represented by mask creates a 4-AP."""
    for other_mask in four_aps_by_v[v]:
        if (mask & other_mask) == other_mask:
            return False
    return True

def _gain_add(mask, v, three_aps_by_v):
    """Calculate the number of new 3-APs created by adding v to the set."""
    count = 0
    for other_mask in three_aps_by_v[v]:
        if (mask & other_mask) == other_mask:
            count += 1
    return count

def construct():
    start_time = time.time()
    random.seed(42)
    
    three_aps, four_aps, four_aps_by_v, three_aps_by_v = _precompute()
    
    best_mask = 0
    best_score = 0
    
    # GRASP + Local Search
    # We try multiple random restarts
    max_restarts = 1000 # Limit by time instead
    
    while (time.time() - start_time) < DEADLINE - 5.0:
        # Phase 1: Randomized Greedy Construction
        # Randomly shuffle elements
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        
        current_mask = 0
        current_set = set()
        
        for v in elements:
            if _can_add(current_mask, v, four_aps_by_v):
                current_mask |= (1 << (v-1))
                current_set.add(v)
        
        current_score = 0
        # Calculate initial score
        # We can sum gains during construction, but let's just count at the end for simplicity
        # Actually, counting is O(M3) which is ~3000. Fast enough.
        for ap_mask in three_aps:
            if (current_mask & ap_mask) == ap_mask:
                current_score += 1
        
        # Phase 2: Local Search
        improved = True
        while improved and (time.time() - start_time) < DEADLINE - 5.0:
            improved = False
            
            # Try Add moves
            best_add_v = -1
            best_add_gain = 0
            for v in range(1, N + 1):
                if v in current_set:
                    continue
                if _can_add(current_mask, v, four_aps_by_v):
                    gain = _gain_add(current_mask, v, three_aps_by_v)
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add_v = v
            
            if best_add_v != -1:
                current_mask |= (1 << (best_add_v-1))
                current_set.add(best_add_v)
                current_score += best_add_gain
                improved = True
                continue # Restart loop to check for more adds
            
            # If no add, try Swap moves
            best_swap_u = -1
            best_swap_v = -1
            best_swap_delta = 0
            
            # To speed up, we might only try swaps if set size is large
            # But swaps are crucial for escaping local optima
            for u in list(current_set):
                # Calculate loss from removing u
                loss = _gain_add(current_mask, u, three_aps_by_v)
                
                # Tentatively remove u
                mask_without_u = current_mask & ~(1 << (u-1))
                
                for v in range(1, N + 1):
                    if v in current_set:
                        continue
                    if _can_add(mask_without_u, v, four_aps_by_v):
                        gain = _gain_add(mask_without_u, v, three_aps_by_v)
                        delta = gain - loss
                        if delta > best_swap_delta:
                            best_swap_delta = delta
                            best_swap_u = u
                            best_swap_v = v
            
            if best_swap_u != -1:
                # Apply swap
                current_mask &= ~(1 << (best_swap_u-1))
                current_set.remove(best_swap_u)
                current_mask |= (1 << (best_swap_v-1))
                current_set.add(best_swap_v)
                current_score += best_swap_delta
                improved = True
                continue # Restart loop to check for more swaps/adds
            
            # If no improvement, break local search
            break
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask

    # Convert best_mask back to sorted list
    result = []
    for v in range(1, N + 1):
        if best_mask & (1 << (v-1)):
            result.append(v)
    
    return sorted(result)
