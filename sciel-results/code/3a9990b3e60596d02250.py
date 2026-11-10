import time
import random

N = 256
DEADLINE = 55.0

def _precompute():
    """Precompute 3-AP and 4-AP bitmasks and element-to-AP mappings."""
    # 3-APs: (a, a+d, a+2d)
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            three_aps.append(mask)
    
    # 4-APs: (a, a+d, a+2d, a+3d)
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps.append(mask)
    
    # Map each element index to list of 4-AP indices containing it
    elem_to_4ap = [[] for _ in range(N)]
    for idx, mask in enumerate(four_aps):
        for i in range(N):
            if mask & (1 << i):
                elem_to_4ap[i].append(idx)
    
    # Map each element index to list of 3-AP indices containing it
    elem_to_3ap = [[] for _ in range(N)]
    for idx, mask in enumerate(three_aps):
        for i in range(N):
            if mask & (1 << i):
                elem_to_3ap[i].append(idx)
    
    return three_aps, four_aps, elem_to_4ap, elem_to_3ap

def _count_3aps(mask, three_aps):
    """Count number of 3-APs fully contained in mask."""
    count = 0
    for ap_mask in three_aps:
        if (mask & ap_mask) == ap_mask:
            count += 1
    return count

def _has_4ap(mask, four_aps):
    """Check if any 4-AP is fully contained in mask."""
    for ap_mask in four_aps:
        if (mask & ap_mask) == ap_mask:
            return True
    return False

def construct():
    start_time = time.time()
    three_aps, four_aps, elem_to_4ap, elem_to_3ap = _precompute()
    
    # Initialize with greedy set (same as seed)
    selected = set()
    for x in range(1, N+1):
        if all(not all(x-j*d in selected for j in (1,2,3)) for d in range(1,(x-1)//3+1)):
            selected.add(x)
    
    # Convert to bitmask
    mask = 0
    for x in selected:
        mask |= (1 << (x-1))
    
    # Count initial 3-APs
    current_score = _count_3aps(mask, three_aps)
    best_mask = mask
    best_score = current_score
    
    # Verify no 4-AP
    if _has_4ap(mask, four_aps):
        # Shouldn't happen with greedy, but safety check
        pass
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    random.seed(42)
    
    while time.time() - start_time < DEADLINE:
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Generate a random move: add, remove, or swap an element
        move_type = random.random()
        
        if move_type < 0.35:
            # Add a random element
            candidates = [i for i in range(N) if not (mask & (1 << i))]
            if not candidates:
                continue
            i = random.choice(candidates)
            new_mask = mask | (1 << i)
            # Check 4-AP constraint
            if _has_4ap(new_mask, four_aps):
                continue
            # Compute score change
            old_count = _count_3aps(mask, three_aps)
            new_count = _count_3aps(new_mask, three_aps)
            delta = new_count - old_count
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                mask = new_mask
                current_score = new_count
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
        elif move_type < 0.55:
            # Remove a random element
            present = [i for i in range(N) if mask & (1 << i)]
            if not present:
                continue
            i = random.choice(present)
            new_mask = mask & ~(1 << i)
            old_count = _count_3aps(mask, three_aps)
            new_count = _count_3aps(new_mask, three_aps)
            delta = new_count - old_count
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                mask = new_mask
                current_score = new_count
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
        else:
            # Swap: remove one present, add one absent
            present = [i for i in range(N) if mask & (1 << i)]
            absent = [i for i in range(N) if not (mask & (1 << i))]
            if not present or not absent:
                continue
            i_out = random.choice(present)
            i_in = random.choice(absent)
            new_mask = (mask & ~(1 << i_out)) | (1 << i_in)
            if _has_4ap(new_mask, four_aps):
                continue
            old_count = _count_3aps(mask, three_aps)
            new_count = _count_3aps(new_mask, three_aps)
            delta = new_count - old_count
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                mask = new_mask
                current_score = new_count
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
    
    # Convert best_mask back to sorted list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)
# EVOLVE-BLOCK-END
