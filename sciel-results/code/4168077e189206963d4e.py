import itertools
import math
import time

def score_set(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    start_time = time.time()
    limit = 120
    
    best_score = 0
    best_set = [0]
    
    # Search subsets of {0..15}
    # 2^15 = 32768 subsets. Very fast.
    # Try {0..20} = 10^6 subsets.
    
    for n in range(1, 21):
        if time.time() - start_time > limit - 10:
            break

    # Correct iteration
    max_val = 20
    for mask in range(1 << max_val):
        if time.time() - start_time > limit - 5:
            break
        U = [0]
        for i in range(1, max_val):
            if (mask >> i) & 1:
                U.append(i)
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    return best_set
