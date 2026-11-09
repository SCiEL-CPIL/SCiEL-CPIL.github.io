import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    
    # We will try to find better M by sieving
    # Let's try a few sizes
    sizes = [100000, 500000, 2000000, 10000000]
    
    for N in sizes:
        if time.time() > deadline - 5.0:
            break
            
        # Compute sigma(k) for k=1..N
        sigma = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                sigma[j] += i
                
        # Group by M
        groups = {}
        for k in range(1, N + 1):
            M = k * sigma[k]
            if M <= 10**18:
                if M not in groups:
                    groups[M] = []
                groups[M].append(k)
                
        # Find best in this batch
        # We can update best_M if we find a group larger than current best
        # But we need to check if the new group is larger
        # Also, we should check if existing best_M got more solutions in this larger N
        # Since we are increasing N, the groups from previous N are subset of current N groups?
        # No, sigma[k] is fixed. So if k <= N_prev, it's in the new list too.
        # So we can just find the max length in `groups`.
        
        max_count = 0
        current_best_M = None
        current_best_list = None
        
        for M, ks in groups.items():
            if len(ks) > max_count:
                max_count = len(ks)
                current_best_M = M
                current_best_list = ks
                
        if current_best_list and len(current_best_list) > len(best_k):
            best_M = current_best_M
            best_k = current_best_list
            
    return {'M': best_M, 'k': best_k}
