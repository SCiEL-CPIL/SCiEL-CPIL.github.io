import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    p = 101
    denominators = list(range(1, 11))
    invs = [pow(x, -1, p) for x in denominators]
    
    # best_rep[r] = smallest-size list of denominators summing to r mod 101
    best_rep = [None] * p
    max_size_needed = 0
    
    # Iterate by subset size k=1 to 10
    for k in range(1, 11):
        if time.time() > deadline:
            break
        if all(x is not None for x in best_rep):
            break
        
        for combo in combinations(range(10), k):
            s = 0
            for i in combo:
                s = (s + invs[i]) % p
            if best_rep[s] is None:
                best_rep[s] = [denominators[i] for i in combo]
        
        # Track the max size used so far
        if k > max_size_needed:
            max_size_needed = k
    
    # Fallback: if any residue is still None (shouldn't happen), scan all subsets
    if any(x is None for x in best_rep):
        for mask in range(1, 1 << 10):
            if time.time() > deadline:
                break
            denom_list = []
            s = 0
            for i in range(10):
                if mask & (1 << i):
                    denom_list.append(denominators[i])
                    s = (s + invs[i]) % p
            if best_rep[s] is None:
                best_rep[s] = denom_list
    
    # Final safety: fill any remaining None with full set (should never trigger)
    for r in range(p):
        if best_rep[r] is None:
            best_rep[r] = denominators[:]
    
    return best_rep
