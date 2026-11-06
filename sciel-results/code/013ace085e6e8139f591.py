"""
Maximize 3-AP count in a 4-AP-free subset of [1,256].
Strategy: Simulated annealing with bitmasks, precomputed 3-AP/4-AP lists,
and a deadline guard.
"""
import time
import random
import numpy as np

# Deadline: must finish within 60s wall-clock; use 55s as safety margin
DEADLINE = 55.0
N = 256

# Precompute all 3-APs (a, b, c) with a < b < c, b-a = c-b > 0, all in [1, N]
# Represent as (a-1, b-1, c-1) zero-indexed
THREE_APS = []
for a in range(1, N + 1):
    for d in range(1, N):
        b = a + d
        c = b + d
        if c > N:
            break
        THREE_APS.append((a - 1, b - 1, c - 1))

# Precompute all 4-APs (a, b, c, d) with a < b < c < d, equal spacing > 0
FOUR_APS = []
for a in range(1, N + 1):
    for d in range(1, N):
        b = a + d
        c = b + d
        e = c + d
        if e > N:
            break
        FOUR_APS.append((a - 1, b - 1, c - 1, e - 1))

# Precompute for each element, which 3-APs it participates in (as a, b, or c)
# And which 4-APs it participates in
elem_to_3ap = [[] for _ in range(N)]
for idx, (a, b, c) in enumerate(THREE_APS):
    elem_to_3ap[a].append(idx)
    elem_to_3ap[b].append(idx)
    elem_to_3ap[c].append(idx)

elem_to_4ap = [[] for _ in range(N)]
for idx, (a, b, c, d) in enumerate(FOUR_APS):
    elem_to_4ap[a].append(idx)
    elem_to_4ap[b].append(idx)
    elem_to_4ap[c].append(idx)
    elem_to_4ap[d].append(idx)

# For fast evaluation: for a given set (bitmask), count 3-APs and 4-APs
def count_3aps_and_4aps(mask):
    """Count 3-APs and 4-APs in the set represented by mask (numpy uint64 array of 4 words)."""
    # mask is a tuple/list of 4 uint64s for 256 bits
    # We'll use numpy for speed
    pass

def _bit_set(mask, pos):
    """Check if bit pos is set in mask (list of 4 uint64s)."""
    return (mask[pos >> 6] >> (pos & 63)) & 1

def count_3aps(mask):
    cnt = 0
    for a, b, c in THREE_APS:
        if _bit_set(mask, a) and _bit_set(mask, b) and _bit_set(mask, c):
            cnt += 1
    return cnt

def count_4aps(mask):
    cnt = 0
    for a, b, c, d in FOUR_APS:
        if _bit_set(mask, a) and _bit_set(mask, b) and _bit_set(mask, c) and _bit_set(mask, d):
            cnt += 1
    return cnt

def objective(mask):
    """
    Objective: maximize 3-APs, penalize 4-APs heavily.
    Return (score, n3, n4)
    """
    n3 = count_3aps(mask)
    n4 = count_4aps(mask)
    # Penalize 4-APs so heavily that any 4-AP-free set with fewer 3-APs still wins
    # Use penalty large enough: max possible 3-APs is ~N^2/6 ≈ 10922 for N=256
    # Actually for a dense set, 3-APs ~ N^2/6. Let's use penalty = 100000
    score = n3 - 100000 * n4
    return score, n3, n4

def greedy_init():
    """
    Greedy: add elements one by one, preferring those that add most 3-APs
    without creating any 4-AP.
    """
    mask = [0, 0, 0, 0]
    # Start with a random-ish seed: take every 3rd or use a known construction
    # Actually, let's use a greedy that tries to maximize 3-APs while avoiding 4-APs
    # We'll iterate through elements in a random order and add if it doesn't create a 4-AP
    # and adds at least one 3-AP
    
    # Better: use a known good starting point. The greedy in the seed adds x if it
    # doesn't complete a 4-AP with already-selected elements. Let's do something similar
    # but also consider 3-AP gains.
    
    # Simple approach: try all elements, add those that don't create 4-APs,
    # prioritizing those with high 3-AP potential
    order = list(range(N))
    random.shuffle(order)
    
    for pos in order:
        # Check if adding pos creates any 4-AP
        creates_4ap = False
        for ap_idx in elem_to_4ap[pos]:
            a, b, c, d = FOUR_APS[ap_idx]
            # Check if the other 3 are already in mask
            others = [x for x in (a, b, c, d) if x != pos]
            if all(_bit_set(mask, x) for x in others):
                creates_4ap = True
                break
        if not creates_4ap:
            # Check 3-AP gain
            gain = 0
            for ap_idx in elem_to_3ap[pos]:
                a, b, c = THREE_APS[ap_idx]
                others = [x for x in (a, b, c) if x != pos]
                if all(_bit_set(mask, x) for x in others):
                    gain += 1
            # Add if gain > 0 or randomly with small prob
            if gain > 0 or random.random() < 0.1:
                mask[pos >> 6] |= (1 << (pos & 63))
    
    return mask

def simulated_annealing(initial_mask, time_budget):
    """
    Simulated annealing to improve the mask.
    """
    mask = list(initial_mask)
    cur_score, cur_n3, cur_n4 = objective(mask)
    best_mask = list(mask)
    best_score = cur_score
    best_n3 = cur_n3
    best_n4 = cur_n4
    
    start_time = time.time()
    T = 10.0  # initial temperature
    cooling = 0.9999  # per-step cooling
    min_T = 0.01
    
    step = 0
    while time.time() - start_time < time_budget:
        step += 1
        # Pick a random element to toggle
        pos = random.randint(0, N - 1)
        
        # Compute delta
        # Count 3-APs and 4-APs involving pos that are currently satisfied
        # and would be affected by toggling pos
        
        is_set = _bit_set(mask, pos)
        
        # Delta in 3-APs
        d3 = 0
        for ap_idx in elem_to_3ap[pos]:
            a, b, c = THREE_APS[ap_idx]
            others = [x for x in (a, b, c) if x != pos]
            # Check if all others are in mask
            if all(_bit_set(mask, x) for x in others):
                if is_set:
                    d3 -= 1  # removing pos breaks this 3-AP
                else:
                    d3 += 1  # adding pos creates this 3-AP
        
        # Delta in 4-APs
        d4 = 0
        for ap_idx in elem_to_4ap[pos]:
            a, b, c, d = FOUR_APS[ap_idx]
            others = [x for x in (a, b, c, d) if x != pos]
            if all(_bit_set(mask, x) for x in others):
                if is_set:
                    d4 -= 1  # removing pos breaks this 4-AP
                else:
                    d4 += 1  # adding pos creates this 4-AP
        
        delta_score = d3 - 100000 * d4
        
        # Accept or reject
        if delta_score > 0:
            accept = True
        else:
            # Metropolis criterion
            if T > min_T:
                ratio = delta_score / T
                if ratio > -50:  # avoid overflow
                    accept = random.random() < np.exp(ratio)
                else:
                    accept = False
            else:
                accept = False
        
        if accept:
            mask[pos >> 6] ^= (1 << (pos & 63))
            cur_score += delta_score
            cur_n3 += d3
            cur_n4 += d4
            
            if cur_score > best_score:
                best_score = cur_score
                best_mask = list(mask)
                best_n3 = cur_n3
                best_n4 = cur_n4
        
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
    
    return best_mask, best_score, best_n3, best_n4

def construct():
    start = time.time()
    time_budget = DEADLINE - (start - start)  # 55s
    time_budget = min(time_budget, 50.0)  # leave 5s for finalization
    
    # Try multiple random initializations and pick the best
    best_mask = None
    best_score = -float('inf')
    best_n3 = 0
    best_n4 = 0
    
    # First, try a few greedy inits
    for i in range(3):
        mask = greedy_init()
        score, n3, n4 = objective(mask)
        if n4 == 0 and score > best_score:
            best_score = score
            best_mask = list(mask)
            best_n3 = n3
            best_n4 = n4
    
    if best_mask is None:
        # Fallback: use the original greedy
        selected = set()
        for x in range(1, 257):
            if all(not all(x - j * d in selected for j in (1, 2, 3)) for d in range(1, (x - 1) // 3 + 1)):
                selected.add(x)
        mask = [0, 0, 0, 0]
        for x in selected:
            pos = x - 1
            mask[pos >> 6] |= (1 << (pos & 63))
        best_mask = mask
        best_score, best_n3, best_n4 = objective(mask)
    
    # Now run simulated annealing with the best initial mask
    elapsed = time.time() - start
    remaining = time_budget - elapsed
    if remaining > 2.0:
        best_mask, best_score, best_n3, best_n4 = simulated_annealing(
            best_mask, remaining
        )
    
    # Verify: ensure no 4-APs
    # If there are 4-APs, remove elements to break them
    while True:
        n4 = count_4aps(best_mask)
        if n4 == 0:
            break
        # Find a 4-AP and remove one element
        for a, b, c, d in FOUR_APS:
            if _bit_set(best_mask, a) and _bit_set(best_mask, b) and \
               _bit_set(best_mask, c) and _bit_set(best_mask, d):
                # Remove the middle element (b or c)
                pos = b
                best_mask[pos >> 6] &= ~(1 << (pos & 63))
                break
        # Check time
        if time.time() - start > DEADLINE - 1.0:
            break
    
    # Final check: if still has 4-APs, fall back to a safe construction
    if count_4aps(best_mask) > 0:
        # Use a simple safe construction: take numbers that are 0 or 1 in base 3
        # This is a classic 4-AP-free set
        mask = [0, 0, 0, 0]
        for x in range(1, 257):
            # Check if x-1 has no digit 2 in base 3
            y = x - 1
            valid = True
            while y > 0:
                if y % 3 == 2:
                    valid = False
                    break
                y //= 3
            if valid:
                pos = x - 1
                mask[pos >> 6] |= (1 << (pos & 63))
        best_mask = mask
        best_score, best_n3, best_n4 = objective(mask)
    
    # Convert mask back to sorted list
    result = []
    for pos in range(N):
        if _bit_set(best_mask, pos):
            result.append(pos + 1)
    
    return sorted(result)
