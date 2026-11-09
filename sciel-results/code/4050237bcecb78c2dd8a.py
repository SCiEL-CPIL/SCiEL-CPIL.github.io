import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute conflicts
    # conflicts[i][j] is a bitmask of k such that {i, j, k} is a bad triple
    # We only need k > j to avoid duplicates, but bitmask is easier with all k
    conflicts = [[0] * (n + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l_ij = lcm_table[i][j]
            mask = 0
            for k in range(j + 1, n + 1):
                l_jk = lcm_table[j][k]
                l_ik = lcm_table[i][k]
                if l_ij == l_jk == l_ik:
                    mask |= (1 << k)
            conflicts[i][j] = mask
            conflicts[j][i] = mask

    # Bitmasks for all numbers 1..128
    # Bit 0 is unused, bit 1 corresponds to number 1, etc.
    # Actually, let's use bit (x-1) for number x to keep it tight?
    # Python ints are arbitrary precision, so 1 << 127 is fine.
    # Let's use bit x for number x for simplicity, ignoring bit 0.
    
    all_mask = 0
    for i in range(1, n + 1):
        all_mask |= (1 << i)
        
    best_size = 0
    best_mask = 0
    
    # Order candidates to improve pruning?
    # We can just pick the lowest index available in the mask.
    
    def popcount(x):
        return bin(x).count('1')
        
    def dfs(candidates_mask, current_mask, danger_mask):
        nonlocal best_size, best_mask
        
        # Pruning
        if popcount(current_mask) + popcount(candidates_mask) <= best_size:
            return
            
        if candidates_mask == 0:
            if popcount(current_mask) > best_size:
                best_size = popcount(current_mask)
                best_mask = current_mask
            return
            
        # Check time
        if time.time() > deadline:
            return

        # Pick a candidate
        # Heuristic: pick the candidate that causes the most conflicts?
        # For simplicity and speed, pick the lowest bit.
        # x = (candidates_mask & -candidates_mask).bit_length() - 1
        # But we need the index.
        low_bit = candidates_mask & -candidates_mask
        x = low_bit.bit_length() - 1
        
        # Branch 1: Include x
        # Check if x is in danger_mask
        if not (danger_mask & low_bit):
            # Calculate new danger_mask
            new_danger = danger_mask
            # Iterate through bits in current_mask
            cm = current_mask
            while cm:
                lb = cm & -cm
                y = lb.bit_length() - 1
                # Add conflicts[x][y] to new_danger
                new_danger |= conflicts[x][y]
                cm -= lb
                
            # Update candidates: remove x and remove any new dangers
            new_candidates = candidates_mask & ~low_bit & ~new_danger
            new_current = current_mask | low_bit
            
            dfs(new_candidates, new_current, new_danger)
            
        # Branch 2: Exclude x
        new_candidates = candidates_mask & ~low_bit
        dfs(new_candidates, current_mask, danger_mask)

    dfs(all_mask, 0, 0)
    
    if best_mask == 0:
        return [1]
        
    result = []
    for i in range(1, n + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    return result
