"""Rewritten: density-driven greedy + simulated annealing for max 3-APs without 4-APs in [1,256]."""
import time
import random

_N = 256
_DEADLINE = 80.0  # safety margin before 90s kill

def _has_4ap_bitset(mask):
    """Check if a bitset (int) contains a nonconstant 4-term AP."""
    # For each pair (i,j) with i<j, check if i+d, i+2d, i+3d all present
    # where d = j-i, and i+3d <= N
    # Fast approach: iterate over all 4-APs
    for a in range(1, _N + 1):
        bit_a = 1 << (a - 1)
        if not (mask & bit_a):
            continue
        max_d = (_N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            # Check b, c, e all set
            if (mask & (1 << (b-1))) and (mask & (1 << (c-1))) and (mask & (1 << (e-1))):
                return True
    return False

def _count_3aps_bitset(mask):
    """Count distinct 3-term APs (a, a+d, a+2d) with d>0 in the bitset."""
    count = 0
    for a in range(1, _N + 1):
        bit_a = 1 << (a - 1)
        if not (mask & bit_a):
            continue
        max_d = (_N - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2*d
            if (mask & (1 << (b-1))) and (mask & (1 << (c-1))):
                count += 1
    return count

def _greedy_density_driven():
    """Build a set greedily, prioritizing elements that create many 3-APs while avoiding 4-APs."""
    mask = 0
    # Precompute for each candidate x, the 3-APs it would form with existing elements
    # and whether it would create a 4-AP
    for x in range(1, _N + 1):
        # Check if adding x creates a 4-AP
        creates_4ap = False
        # x as first term: x, x+d, x+2d, x+3d
        for d in range(1, (_N - x) // 3 + 1):
            if (mask & (1 << (x+d-1))) and (mask & (1 << (x+2*d-1))) and (mask & (1 << (x+3*d-1))):
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as second term: x-d, x, x+d, x+2d
        for d in range(1, min(x-1, (_N - x) // 2) + 1):
            if (mask & (1 << (x-d-1))) and (mask & (1 << (x+d-1))) and (mask & (1 << (x+2*d-1))):
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as third term: x-2d, x-d, x, x+d
        for d in range(1, min((x-1)//2, _N - x) + 1):
            if (mask & (1 << (x-2*d-1))) and (mask & (1 << (x-d-1))) and (mask & (1 << (x+d-1))):
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as fourth term: x-3d, x-2d, x-d, x
        for d in range(1, (x-1)//3 + 1):
            if (mask & (1 << (x-3*d-1))) and (mask & (1 << (x-2*d-1))) and (mask & (1 << (x-d-1))):
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # Count 3-APs x would form
        gain = 0
        # x as first: x, x+d, x+2d
        for d in range(1, (_N - x) // 2 + 1):
            if (mask & (1 << (x+d-1))) and (mask & (1 << (x+2*d-1))):
                gain += 1
        # x as second: x-d, x, x+d
        for d in range(1, min(x-1, _N - x) + 1):
            if (mask & (1 << (x-d-1))) and (mask & (1 << (x+d-1))):
                gain += 1
        # x as third: x-2d, x-d, x
        for d in range(1, (x-1)//2 + 1):
            if (mask & (1 << (x-2*d-1))) and (mask & (1 << (x-d-1))):
                gain += 1
        mask |= (1 << (x - 1))
    return mask

def _simulated_annealing(mask, time_budget):
    """Improve mask via simulated annealing: add/remove elements to maximize 3-APs."""
    start = time.time()
    best_mask = mask
    best_score = _count_3aps_bitset(mask)
    current_score = best_score
    temp = 50.0
    cooling = 0.999
    
    # Precompute all valid 3-APs and 4-APs for speed
    all_3aps = []
    for a in range(1, _N + 1):
        for d in range(1, (_N - a) // 2 + 1):
            all_3aps.append((a, a+d, a+2*d))
    
    all_4aps = []
    for a in range(1, _N + 1):
        for d in range(1, (_N - a) // 3 + 1):
            all_4aps.append((a, a+d, a+2*d, a+3*d))
    
    while time.time() - start < time_budget:
        # Pick a random element to toggle
        x = random.randint(1, _N)
        bit = 1 << (x - 1)
        
        if mask & bit:
            # Try removing x
            new_mask = mask & ~bit
            # Check no 4-AP (removing can't create one, so always valid)
            new_score = _count_3aps_bitset(new_mask)
            delta = new_score - current_score
        else:
            # Try adding x
            new_mask = mask | bit
            # Check no 4-AP
            valid = True
            for (a, b, c, e) in all_4aps:
                if x in (a, b, c, e):
                    others = [v for v in (a, b, c, e) if v != x]
                    if all((new_mask & (1 << (v-1))) for v in others):
                        valid = False
                        break
            if not valid:
                continue
            new_score = _count_3aps_bitset(new_mask)
            delta = new_score - current_score
        
        # Accept or reject
        if delta > 0 or random.random() < pow(2.718281828, delta / max(temp, 0.01)):
            mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = mask
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return best_mask, best_score

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    
    # Phase 1: Density-driven greedy
    mask = _greedy_density_driven()
    score = _count_3aps_bitset(mask)
    
    # Phase 2: Simulated annealing with remaining time
    remaining = _DEADLINE - (time.time() - start)
    if remaining > 2:
        mask, score = _simulated_annealing(mask, remaining - 1)
    
    # Convert to sorted list
    result = [i+1 for i in range(_N) if (mask >> i) & 1]
    
    # Safety: ensure nonempty and valid
    if not result:
        result = [1, 2, 4, 8]  # fallback, no 4-APs
        # Verify
        if _has_4ap_bitset(sum(1 << (x-1) for x in result)):
            result = [1, 3, 7, 15]
    
    return sorted(result)
