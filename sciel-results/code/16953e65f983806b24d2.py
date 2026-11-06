import random
import time

def construct():
    N = 512
    best_set = set()
    start_time = time.time()
    time_limit = 55.0  # Leave some buffer for 60s limit
    
    # Helper function to check if adding x (0-indexed) to bitset creates a 4-AP
    def is_safe(x, bitset):
        # Case 1: x is the 4th term (a+3d)
        # a = x-3d, a+d = x-2d, a+2d = x-d
        # All must be in bitset
        max_d1 = x // 3
        for d in range(1, max_d1 + 1):
            if ((bitset >> (x - 3*d)) & 1) and ((bitset >> (x - 2*d)) & 1) and ((bitset >> (x - d)) & 1):
                return False
        
        # Case 2: x is the 3rd term (a+2d)
        # a = x-2d, a+d = x-d, a+3d = x+d
        max_d2 = min(x // 2, (N - 1 - x))
        for d in range(1, max_d2 + 1):
            if ((bitset >> (x - 2*d)) & 1) and ((bitset >> (x - d)) & 1) and ((bitset >> (x + d)) & 1):
                return False
        
        # Case 3: x is the 2nd term (a+d)
        # a = x-d, a+2d = x+2d, a+3d = x+3d
        max_d3 = min(x, (N - 1 - x) // 3)
        for d in range(1, max_d3 + 1):
            if ((bitset >> (x - d)) & 1) and ((bitset >> (x + 2*d)) & 1) and ((bitset >> (x + 3*d)) & 1):
                return False
        
        # Case 4: x is the 1st term (a)
        # a+d = x+d, a+2d = x+2d, a+3d = x+3d
        max_d4 = (N - 1 - x) // 3
        for d in range(1, max_d4 + 1):
            if ((bitset >> (x + d)) & 1) and ((bitset >> (x + 2*d)) & 1) and ((bitset >> (x + 3*d)) & 1):
                return False
        
        return True

    # Run multiple random greedy attempts
    while time.time() - start_time < time_limit:
        bitset = 0
        order = list(range(N))
        random.shuffle(order)
        
        for x in order:
            if is_safe(x, bitset):
                bitset |= (1 << x)
        
        # Count bits
        count = bin(bitset).count('1')
        if count > len(best_set):
            best_set = {i for i in range(N) if (bitset >> i) & 1}
            
    # Return as 1-indexed sorted list
    return sorted([x + 1 for x in best_set])
