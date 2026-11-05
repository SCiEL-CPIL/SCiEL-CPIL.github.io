import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute reciprocals
    recip = [0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i
    
    # Precompute AP pairs for each x
    # ap_pairs[x] contains (y, z) such that {x, y, z} is a 3-AP
    # We only need to check if y and z are in the set when considering adding x
    ap_pairs = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        # x is the middle: y, x, z -> y + z = 2x
        # y < x < z
        for y in range(1, x):
            z = 2 * x - y
            if z <= n and z > x:
                ap_pairs[x].append((y, z))
        
        # x is an endpoint: x, y, z -> x + z = 2y -> z = 2y - x
        # x < y < z
        for y in range(x + 1, n + 1):
            z = 2 * y - x
            if z <= n and z > y:
                ap_pairs[x].append((y, z))
                
        # x is the other endpoint: y, z, x -> y + x = 2z -> y = 2z - x
        # y < z < x
        for z in range(1, x):
            y = 2 * z - x
            if y >= 1 and y < z:
                ap_pairs[x].append((y, z))

    # Remove duplicates in ap_pairs[x] just in case (though logic above should be disjoint)
    # Actually, (y,z) is unique for a given AP.
    
    def can_add(x, mask):
        # Check if x can be added to the set represented by mask
        for y, z in ap_pairs[x]:
            if ((mask >> y) & 1) and ((mask >> z) & 1):
                return False
        return True

    def get_sum(mask):
        total = 0.0
        # Iterate through set bits
        m = mask
        while m:
            # Get index of least significant set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            total += recip[idx]
            m ^= lsb
        return total

    def greedy_fill(mask):
        # Add elements in increasing order (prioritizing high weight)
        # Note: Increasing index = Decreasing weight (1/i)
        # So we iterate 1 to N
        for x in range(1, n + 1):
            if not ((mask >> x) & 1):
                if can_add(x, mask):
                    mask |= (1 << x)
        return mask

    def local_search(mask):
        improved = True
        current_sum = get_sum(mask)
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try removing each element and refilling
            # To speed up, we can randomize the order of removal or just iterate
            # Iterating 1 to N is fine.
            for rem in range(1, n + 1):
                if not ((mask >> rem) & 1):
                    continue
                
                trial_mask = mask & ~(1 << rem)
                trial_mask = greedy_fill(trial_mask)
                new_sum = get_sum(trial_mask)
                
                if new_sum > current_sum + 1e-12:
                    mask = trial_mask
                    current_sum = new_sum
                    improved = True
                    break # Restart the loop with the new mask
            
        return mask

    # 1. Initial Greedy
    initial_mask = 0
    best_mask = greedy_fill(initial_mask)
    best_sum = get_sum(best_mask)
    
    # 2. Local Search on Initial
    improved_mask = local_search(best_mask)
    if get_sum(improved_mask) > best_sum:
        best_mask = improved_mask
        best_sum = get_sum(best_mask)
        
    # 3. Randomized Restart
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # Perturb best_mask
        perturbed_mask = best_mask
        
        # Remove 1 to 3 random elements
        num_removals = random.randint(1, 3)
        # Get list of elements in set
        elements_in_set = []
        m = perturbed_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            elements_in_set.append(idx)
            m ^= lsb
            
        if not elements_in_set:
            break
            
        for _ in range(num_removals):
            if not elements_in_set:
                break
            idx = random.randint(0, len(elements_in_set) - 1)
            rem_val = elements_in_set[idx]
            perturbed_mask &= ~(1 << rem_val)
            elements_in_set.pop(idx)
        
        # Refill
        perturbed_mask = greedy_fill(perturbed_mask)
        
        # Local Search
        improved_mask = local_search(perturbed_mask)
        
        new_sum = get_sum(improved_mask)
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_mask = improved_mask

    # Extract result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx)
        m ^= lsb
        
    # Validation
    s_set = set(result)
    for i, a in enumerate(result):
        for b in result[i+1:]:
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    raise AssertionError("3-AP found")
                    
    return result
