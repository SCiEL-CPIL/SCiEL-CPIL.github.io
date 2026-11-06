import random
import time

def construct():
    N = 512
    best_set = set()
    start_time = time.time()
    max_time = 58  # Leave some buffer for 60s limit
    
    # Precompute max d limits for each position to avoid recalculating
    # max_d_1[x] for x being 1st term: x+3d <= N => d <= (N-x)//3
    # max_d_2[x] for x being 2nd term: x-d >= 1, x+2d <= N => d <= min(x-1, (N-x)//2)
    # max_d_3[x] for x being 3rd term: x-2d >= 1, x+d <= N => d <= min((x-1)//2, N-x)
    # max_d_4[x] for x being 4th term: x-3d >= 1 => d <= (x-1)//3
    
    max_d_1 = [0] * (N + 1)
    max_d_2 = [0] * (N + 1)
    max_d_3 = [0] * (N + 1)
    max_d_4 = [0] * (N + 1)
    
    for x in range(1, N + 1):
        max_d_1[x] = (N - x) // 3
        max_d_2[x] = min(x - 1, (N - x) // 2)
        max_d_3[x] = min((x - 1) // 2, N - x)
        max_d_4[x] = (x - 1) // 3

    while time.time() - start_time < max_time:
        S_bits = 0
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if x can be added to S_bits
            valid = True
            
            # Case 1: x is 1st term (x, x+d, x+2d, x+3d)
            # Need x+d, x+2d, x+3d in S
            # Bit x in (S>>d) & (S>>2d) & (S>>3d)
            for d in range(1, max_d_1[x] + 1):
                if (S_bits >> d) & (S_bits >> (2 * d)) & (S_bits >> (3 * d)) & (1 << x):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Case 2: x is 2nd term (x-d, x, x+d, x+2d)
            # Need x-d, x+d, x+2d in S
            # Bit x in (S<<d) & (S>>d) & (S>>2d)
            for d in range(1, max_d_2[x] + 1):
                if (S_bits << d) & (S_bits >> d) & (S_bits >> (2 * d)) & (1 << x):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Case 3: x is 3rd term (x-2d, x-d, x, x+d)
            # Need x-2d, x-d, x+d in S
            # Bit x in (S<<2d) & (S<<d) & (S>>d)
            for d in range(1, max_d_3[x] + 1):
                if (S_bits << (2 * d)) & (S_bits << d) & (S_bits >> d) & (1 << x):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Case 4: x is 4th term (x-3d, x-2d, x-d, x)
            # Need x-3d, x-2d, x-d in S
            # Bit x in (S<<3d) & (S<<2d) & (S<<d)
            for d in range(1, max_d_4[x] + 1):
                if (S_bits << (3 * d)) & (S_bits << (2 * d)) & (S_bits << d) & (1 << x):
                    valid = False
                    break
            
            if valid:
                S_bits |= (1 << x)
        
        # Convert S_bits to set for comparison
        S = set()
        temp_bits = S_bits
        while temp_bits:
            # Extract lowest set bit
            lsb = temp_bits & -temp_bits
            idx = lsb.bit_length() - 1
            S.add(idx)
            temp_bits ^= lsb
        
        if len(S) > len(best_set):
            best_set = S
            
    return sorted(best_set)
