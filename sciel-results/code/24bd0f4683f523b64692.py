import time
import random
import numpy as np

# Precompute 4-APs
N = 512
aps = []
aps_containing = {i: [] for i in range(1, N+1)}

for a in range(1, N+1):
    for d in range(1, (N - a)//3 + 1):
        b = a + d
        c = a + 2*d
        e = a + 3*d
        if e > N:
            break
        ap = (a, b, c, e)
        idx = len(aps)
        aps.append(ap)
        for x in ap:
            aps_containing[x].append(idx)

num_aps = len(aps)

def construct():
    best_set = set()
    start_time = time.time()
    time_limit = 50.0  # Safety margin
    
    # Initialize best set with a simple greedy or empty
    # We'll just run randomized greedy
    
    while time.time() - start_time < time_limit:
        # Random permutation
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        
        S = set()
        ap_counts = np.zeros(num_aps, dtype=np.int8)
        
        for x in candidates:
            # Check if adding x creates a 4-AP
            # If any AP containing x already has 3 elements in S, we can't add x
            valid = True
            for idx in aps_containing[x]:
                if ap_counts[idx] == 3:
                    valid = False
                    break
            
            if valid:
                S.add(x)
                for idx in aps_containing[x]:
                    ap_counts[idx] += 1
        
        if len(S) > len(best_set):
            best_set = S.copy()
            
    return sorted(best_set)
