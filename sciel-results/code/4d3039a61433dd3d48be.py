import time

def max_ap_discrepancy(signs):
    n = len(signs)
    max_abs = 0
    # signs is 0-indexed in list, but problem is 1-indexed.
    # s[0] corresponds to index 1.
    # AP: start a (1..n), step d (1..n), length k.
    # Indices: a-1, a-1+d, ...
    
    for d in range(1, n + 1):
        for a in range(1, n - d + 1):
            # start index in 0-based list is a-1
            s_val = 0
            idx = a - 1
            while idx < n:
                s_val += signs[idx]
                if abs(s_val) > max_abs:
                    max_abs = abs(s_val)
                idx += d
    return max_abs

def construct():
    n = 256
    # Generate Thue-Morse
    signs = []
    for i in range(1, n + 1):
        s2 = bin(i).count('1')
        signs.append(1 if s2 % 2 == 0 else -1)
        
    current_max = max_ap_discrepancy(signs)
    
    # Local Search
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < 50: # Leave 10s buffer
        improved = False
        for i in range(n):
            # Flip bit i
            signs[i] *= -1
            new_max = max_ap_discrepancy(signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                # Revert
                signs[i] *= -1
                
    return signs
