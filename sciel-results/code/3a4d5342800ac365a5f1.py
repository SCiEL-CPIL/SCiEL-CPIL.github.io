import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 10_000_000
    
    # 1. Compute sigma using NumPy
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve
    # Using a loop over i is standard.
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Check time
    if time.time() > deadline - 10:
        # Fallback: return a known good pair
        # sigma(6)=12, sigma(11)=12. 11/6 = 1.833.
        # sigma(14)=24, sigma(15)=24. 15/14 = 1.07.
        # Let's just return 11, 6
        return [11, 6]

    # 2. Sort indices by sigma value
    # stable sort preserves original index order for equal sigma values
    # which means groups will be sorted by value (since original indices are 1..N)
    sorted_indices = np.argsort(sigma, kind='stable')
    
    best_a = 11
    best_b = 6
    best_num = abs(2*11 - 3*6) # 4
    best_den = 2*6 # 12
    
    # 3. Iterate through groups
    i = 1
    while i < N + 1:
        if time.time() > deadline - 2:
            break
            
        # Find the extent of the group with same sigma
        s_val = sigma[sorted_indices[i]]
        j = i + 1
        while j < N + 1 and sigma[sorted_indices[j]] == s_val:
            j += 1
            
        group_size = j - i
        if group_size >= 2:
            group = sorted_indices[i:j]
            
            # Two-pointer search for best ratio
            p = 0
            for k in range(group_size):
                b = group[k]
                target = 1.5 * b
                
                # Advance p
                while p < group_size and group[p] < target:
                    p += 1
                # p is now the index of the first element >= target
                
                # Check p and p-1
                if p < group_size:
                    a = group[p]
                    if a != b:
                        num = abs(2*a - 3*b)
                        den = 2*b
                        if num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a = a
                            best_b = b
                            
                if p > 0:
                    a = group[p-1]
                    if a != b:
                        num = abs(2*a - 3*b)
                        den = 2*b
                        if num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a = a
                            best_b = b
                            
        i = j

    return [int(best_a), int(best_b)]
