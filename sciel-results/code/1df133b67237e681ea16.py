import time
import random
import numpy as np

N = 256

def _precompute():
    """
    Precompute:
    1. four_ap_conflicts[x]: List of bitmasks. Each bitmask represents the other 3 elements 
       in a 4-AP containing x. 
       Condition to add x to mask: For all m in four_ap_conflicts[x], (mask & m) != m.
    2. three_ap_pairs[x]: List of bitmasks. Each bitmask represents the other 2 elements 
       in a 3-AP containing x.
       Delta add x to mask: Count m in three_ap_pairs[x] where (mask & m) == m.
    """
    # 4-APs
    four_aps = []
    for d in range(1, N):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            dd = a + 3*d
            four_aps.append((a, b, c, dd))
    
    # 3-APs
    three_aps = []
    for d in range(1, N):
        for a in range(1, N - 2*d + 1):
            b = a + d
            c = a + 2*d
            three_aps.append((a, b, c))

    # Build conflict masks for 4-APs
    # four_ap_conflicts[x] will be a list of ints (bitmasks of the other 3)
    four_ap_conflicts = {i: [] for i in range(1, N+1)}
    for a, b, c, d in four_aps:
        # For a: others are b, c, d
        mask_a = (1 << (b-1)) | (1 << (c-1)) | (1 << (d-1))
        four_ap_conflicts[a].append(mask_a)
        
        # For b: others are a, c, d
        mask_b = (1 << (a-1)) | (1 << (c-1)) | (1 << (d-1))
        four_ap_conflicts[b].append(mask_b)
        
        # For c: others are a, b, d
        mask_c = (1 << (a-1)) | (1 << (b-1)) | (1 << (d-1))
        four_ap_conflicts[c].append(mask_c)
        
        # For d: others are a, b, c
        mask_d = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
        four_ap_conflicts[d].append(mask_d)

    # Build pair masks for 3-APs
    # three_ap_pairs[x] will be a list of ints (bitmasks of the other 2)
    three_ap_pairs = {i: [] for i in range(1, N+1)}
    for a, b, c in three_aps:
        # For a: others are b, c
        mask_a = (1 << (b-1)) | (1 << (c-1))
        three_ap_pairs[a].append(mask_a)
        
        # For b: others are a, c
        mask_b = (1 << (a-1)) | (1 << (c-1))
        three_ap_pairs[b].append(mask_b)
        
        # For c: others are a, b
        mask_c = (1 << (a-1)) | (1 << (b-1))
        three_ap_pairs[c].append(mask_c)

    return four_ap_conflicts, three_ap_pairs

# Global precomputed data
_FOUR_AP_CONFLICTS, _THREE_AP_PAIRS = _precompute()

def _can_add(mask, x):
    """Check if x can be added to the set represented by mask without creating a 4-AP."""
    for m in _FOUR_AP_CONFLICTS[x]:
        if (mask & m) == m:
            return False
    return True

def _delta_add(mask, x):
    """Calculate the increase in 3-AP count when adding x to the set represented by mask."""
    count = 0
    for m in _THREE_AP_PAIRS[x]:
        if (mask & m) == m:
            count += 1
    return count

def _delta_remove(mask, x):
    """Calculate the decrease in 3-AP count when removing x from the set represented by mask."""
    # This is the same as delta_add but on the set *without* x?
    # No, it's the number of 3-APs that contain x and whose other two elements are in the set.
    # Since x is in the set, the other two must be in the set.
    count = 0
    for m in _THREE_AP_PAIRS[x]:
        if (mask & m) == m:
            count += 1
    return count

def _count_3aps(mask):
    """Count total 3-APs in the set represented by mask."""
    # To avoid double counting, we can sum delta_add for all elements in mask?
    # No, delta_add(x, mask_without_x) is the contribution of x.
    # It is easier to just iterate all 3-APs? Or use the precomputed pairs.
    # Each 3-AP (a,b,c) is counted in:
    # delta_add(a, {b,c})
    # delta_add(b, {a,c})
    # delta_add(c, {a,b})
    # So if we sum _delta_add(mask, x) for all x in mask, we count each 3-AP 3 times.
    # So total = sum / 3.
    total = 0
    # Iterate set bits
    m = mask
    while m:
        # Get index of lowest set bit
        # x = (m & -m).bit_length()
        # This is slow in Python loop.
        # Better to iterate 1..N and check bit.
        pass
    
    # Alternative: Precompute total 3-APs is hard to do dynamically fast.
    # But we maintain score in the search loop, so we only need initial count.
    count = 0
    for x in range(1, N+1):
        if mask & (1 << (x-1)):
            count += _delta_add(mask, x)
    return count // 3

def _greedy_maximal(seed=0):
    """Construct a maximal 4-AP-free set using greedy addition."""
    rng = random.Random(seed)
    mask = 0
    elements = list(range(1, N+1))
    rng.shuffle(elements)
    
    # We want to maximize 3-APs. 
    # A simple greedy: pick the element with highest potential?
    # Or just pick random valid elements until no more can be added.
    # To get diverse seeds, we shuffle the order.
    
    # Optimization: We can try to add elements in a specific order.
    # Let's try to add elements that create many 3-APs with already added ones.
    # But "already added" changes.
    
    # Simplest robust greedy:
    # 1. Shuffle candidates.
    # 2. Iterate candidates. If valid, add.
    # This is fast and produces a maximal set.
    
    for x in elements:
        if _can_add(mask, x):
            mask |= (1 << (x-1))
            
    return mask

def _local_search(mask, time_limit, seed=0):
    """
    Local search with swaps.
    State: mask (int)
    Score: maintained externally or calculated.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Initial score
    current_score = _count_3aps(mask)
    best_mask = mask
    best_score = current_score
    
    # List of elements in mask for fast access
    # We can maintain a list, but checking bit is O(1).
    # For swaps, we need to pick x in mask and y not in mask.
    
    while time.time() < time_limit:
        # Try a swap
        # Pick random x in mask
        # To pick random set bit:
        # m = mask
        # if m == 0: break
        # r = rng.randint(0, m.bit_length()-1) # Not uniform over set bits
        # Better: get list of set bits? Expensive to update.
        # We can just iterate 1..N and pick a random one that is set.
        # Or maintain a list. Let's maintain a list for efficiency in swap selection.
        
        # Let's use a list for current elements
        # But we are in a loop. Let's just use the mask.
        
        # Pick x
        x_found = False
        while not x_found:
            x = rng.randint(1, N)
            if mask & (1 << (x-1)):
                x_found = True
        
        # Pick y not in mask
        y_found = False
        while not y_found:
            y = rng.randint(1, N)
            if not (mask & (1 << (y-1))):
                y_found = True
        
        # Check if swap is valid:
        # Remove x, Add y.
        # New mask = mask ^ (1 << (x-1)) | (1 << (y-1))
        # Condition: _can_add(mask_without_x, y)
        
        mask_without_x = mask ^ (1 << (x-1))
        
        if _can_add(mask_without_x, y):
            new_mask = mask_without_x | (1 << (y-1))
            
            # Calculate score change
            # delta = -_delta_remove(mask, x) + _delta_add(mask_without_x, y)
            # Note: _delta_remove(mask, x) counts 3-APs containing x with others in mask.
            # _delta_add(mask_without_x, y) counts 3-APs containing y with others in mask_without_x.
            # This is correct.
            
            loss = _delta_remove(mask, x)
            gain = _delta_add(mask_without_x, y)
            delta = gain - loss
            
            new_score = current_score + delta
            
            # Acceptance criterion (Simulated Annealing style)
            # Since we are maximizing, if delta > 0, accept.
            # If delta < 0, accept with probability exp(delta / T).
            # We need a T. Let's use a simple cooling schedule or fixed T.
            # For now, let's just do strict local search (accept if delta > 0).
            # If stuck, we need SA.
            
            if delta > 0:
                mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
            else:
                # Try SA acceptance
                # T should decrease over time.
                # Let's estimate progress
                elapsed = time.time() - start_time
                total_time = time_limit - start_time
                progress = elapsed / total_time if total_time > 0 else 1
                T = 10.0 * (1.0 - progress) + 0.1
                
                if delta != 0:
                    prob = np.exp(delta / T)
                else:
                    prob = 1.0
                
                if rng.random() < prob:
                    mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                        
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_mask = 0
    best_score = 0
    
    # Strategy:
    # 1. Generate several greedy maximal seeds.
    # 2. Run local search on each for a fixed time.
    # 3. Keep the best.
    
    # Number of seeds to try
    # Each local search should take ~5-10s.
    # Let's try 5 seeds.
    
    num_seeds = 5
    time_per_seed = 15.0
    
    for i in range(num_seeds):
        if time.time() > deadline - time_per_seed - 2:
            break
            
        # Generate seed
        mask_seed = _greedy_maximal(seed=i*1000)
        
        # Initial score
        score_seed = _count_3aps(mask_seed)
        
        if score_seed > best_score:
            best_score = score_seed
            best_mask = mask_seed
            
        # Run local search
        # Adjust time limit to ensure we finish before deadline
        current_time = time.time()
        available_time = deadline - current_time - 2
        ls_time = min(time_per_seed, available_time)
        
        if ls_time > 1.0:
            mask_ls, score_ls = _local_search(mask_seed, current_time + ls_time, seed=i*2000)
            
            if score_ls > best_score:
                best_score = score_ls
                best_mask = mask_ls

    # Convert mask to list
    result = []
    m = best_mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        idx = lsb.bit_length()
        result.append(idx)
        m ^= lsb
        
    return sorted(result)

# Test run
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Size: {len(res)}")
    # Verify 4-AP free
    A = set(res)
    valid = True
    for d in range(1, 256):
        for a in range(1, 256 - 3*d + 1):
            if a in A and (a+d) in A and (a+2*d) in A and (a+3*d) in A:
                valid = False
                break
        if not valid:
            break
    print(f"4-AP Free: {valid}")
    
    # Count 3-APs
    count = 0
    for d in range(1, 256):
        for a in range(1, 256 - 2*d + 1):
            if a in A and (a+d) in A and (a+2*d) in A:
                count += 1
    print(f"3-AP Count: {count}")
