import time
import sys
from math import gcd

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)
    
    # Precompute compatibility
    # compat[i][j] is True if lcm(vertices[i], vertices[j]) > N
    # We can store this as a list of bitmasks for efficiency
    compat = [0] * n
    for i in range(n):
        a = vertices[i]
        for j in range(i + 1, n):
            b = vertices[j]
            # lcm(a, b) = a * b // gcd(a, b)
            g = gcd(a, b)
            if a * b // g > N:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
                
    weights = [1.0 / v for v in vertices]
    
    # Order vertices by weight descending (index 0 is vertex 2, index 1 is vertex 3, etc.)
    # Since weights are 1/v, larger v means smaller weight.
    # So we want to consider small v first.
    # The vertices are already 2..256, so index 0 is 2, index 1 is 3...
    # This is already sorted by weight descending.
    order = list(range(n))
    
    def get_value(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            s += weights[idx]
            m ^= lsb
        return s
        
    best_mask = 0
    best_val = 0.0
    
    # Try multiple greedy starts with different exclusion prefixes
    # This helps escape local optima caused by locking in small numbers
    max_prefix = 20  # Try skipping first 0 to 20 vertices
    
    for prefix_len in range(max_prefix + 1):
        if time.time() > deadline:
            break
            
        # Create a "forbidden" mask for the first 'prefix_len' vertices
        forbidden_mask = 0
        for i in range(prefix_len):
            forbidden_mask |= (1 << i)
            
        # Greedy Construction
        # S is a bitmask of selected vertices
        S = 0
        for v in order:
            if S & forbidden_mask: # This check is redundant if we just skip in loop
                pass
            # Skip if v is in forbidden mask
            if forbidden_mask & (1 << v):
                continue
                
            # Check if v is compatible with all in S
            # S & ~compat[v] should be 0
            if (S & ~compat[v]) == 0:
                S |= (1 << v)
                
        val = get_value(S)
        if val > best_val:
            best_val = val
            best_mask = S
            
    # Local Search on the best found so far
    # We iterate until no improvement or time limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each vertex in the current best set
        m = best_mask
        while m:
            lsb = m & -m
            u_idx = (lsb.bit_length() - 1)
            m ^= lsb
            
            # Candidate set without u
            S_candidate = best_mask & ~lsb
            val_candidate = best_val - weights[u_idx]
            
            # Greedy Refill
            # Try to add vertices from 'order' that are not in S_candidate
            # and are compatible with S_candidate
            new_mask = S_candidate
            new_val = val_candidate
            
            for v in order:
                if new_mask & (1 << v):
                    continue
                # Check compatibility
                if (new_mask & ~compat[v]) == 0:
                    new_mask |= (1 << v)
                    new_val += weights[v]
                    
            if new_val > best_val:
                best_val = new_val
                best_mask = new_mask
                improved = True
                break # Restart the outer loop with the new best
        
    # Extract result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = (lsb.bit_length() - 1)
        result.append(vertices[idx])
        m ^= lsb
        
    if not result:
        return [2]
        
    return result
