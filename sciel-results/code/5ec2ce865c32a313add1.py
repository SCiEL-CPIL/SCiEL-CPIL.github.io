import time
import random
from fractions import Fraction

N = 256

def _has_3ap_bitmask(mask, y):
    """Check if adding y (1-indexed) to the set represented by mask creates a 3-AP.
    Bit i-1 represents element i. Returns True if a 3-AP would be formed."""
    if mask & (1 << (y - 1)):
        return True
    
    # Case 1: y is the middle term. Need a, y, b where a = y-d, b = y+d
    d = 1
    while y - d >= 1 and y + d <= N:
        if (mask & (1 << (y - d - 1))) and (mask & (1 << (y + d - 1))):
            return True
        d += 1
    
    # Case 2: y is the left endpoint. Need y, y+d, y+2d
    d = 1
    while y + 2 * d <= N:
        if (mask & (1 << (y + d - 1))) and (mask & (1 << (y + 2 * d - 1))):
            return True
        d += 1
    
    # Case 3: y is the right endpoint. Need y-2d, y-d, y
    d = 1
    while y - 2 * d >= 1:
        if (mask & (1 << (y - 2 * d - 1))) and (mask & (1 << (y - d - 1))):
            return True
        d += 1
    
    return False

def _calc_score_float(S):
    """Calculate sum of reciprocals as float for fast comparison."""
    return sum(1.0 / x for x in S)

def _greedy_init(order=None):
    """Greedy initialization: add elements in given order, skipping if 3-AP would form."""
    if order is None:
        order = list(range(1, N + 1))
    mask = 0
    S = set()
    for x in order:
        if not _has_3ap_bitmask(mask, x):
            mask |= (1 << (x - 1))
            S.add(x)
    return S, mask

def _local_search(S, mask, deadline, T_initial=5.0, cooling=0.9995, max_iter=50000):
    """Simulated annealing local search with swap and add moves."""
    best_S = set(S)
    best_mask = mask
    best_score = _calc_score_float(S)
    current_score = best_score
    T = T_initial
    all_elements = list(range(1, N + 1))
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose move type: 70% swap, 30% add
        if random.random() < 0.7:
            # Swap: remove x from S, add y not in S
            if not S:
                continue
            x = random.choice(list(S))
            candidates = [y for y in all_elements if y not in S]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # Check if swap is valid
            new_mask = mask & ~(1 << (x - 1))
            if _has_3ap_bitmask(new_mask, y):
                continue
            new_mask |= (1 << (y - 1))
            new_S = S - {x} | {y}
            new_score = _calc_score_float(new_S)
            delta = new_score - current_score
        else:
            # Add: add y not in S
            candidates = [y for y in all_elements if y not in S]
            if not candidates:
                continue
            y = random.choice(candidates)
            if _has_3ap_bitmask(mask, y):
                continue
            new_mask = mask | (1 << (y - 1))
            new_S = S | {y}
            new_score = _calc_score_float(new_S)
            delta = new_score - current_score
        
        # Accept or reject
        if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 1e-10)):
            S = new_S
            mask = new_mask
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_S = set(S)
                best_mask = mask
        
        T *= cooling
        if T < 0.001:
            T = 0.001
    
    return best_S, best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Multi-start with different orderings
    best_S = set()
    best_mask = 0
    best_score = 0.0
    
    # Start 1: Greedy in natural order (1, 2, 3, ...)
    S1, mask1 = _greedy_init(list(range(1, N + 1)))
    score1 = _calc_score_float(S1)
    if score1 > best_score:
        best_S, best_mask, best_score = S1, mask1, score1
    
    # Start 2: Greedy in random order
    random.seed(42)
    order2 = list(range(1, N + 1))
    random.shuffle(order2)
    S2, mask2 = _greedy_init(order2)
    score2 = _calc_score_float(S2)
    if score2 > best_score:
        best_S, best_mask, best_score = S2, mask2, score2
    
    # Start 3: Greedy prioritizing small numbers but with some randomization
    order3 = list(range(1, N + 1))
    for i in range(len(order3) - 1, 0, -1):
        j = random.randint(0, i)
        order3[i], order3[j] = order3[j], order3[i]
    # Bias towards small numbers: sort by (value, random)
    order3.sort(key=lambda x: (x, random.random()))
    S3, mask3 = _greedy_init(order3)
    score3 = _calc_score_float(S3)
    if score3 > best_score:
        best_S, best_mask, best_score = S3, mask3, score3
    
    # Run local search from each start, keeping the best
    for S_init, mask_init in [(S1, mask1), (S2, mask2), (S3, mask3)]:
        if time.time() > deadline - 10:
            break
        S_res, mask_res, score_res = _local_search(
            S_init, mask_init, 
            deadline=deadline - 5,
            T_initial=5.0,
            cooling=0.9995,
            max_iter=50000
        )
        if score_res > best_score:
            best_S, best_mask, best_score = S_res, mask_res, score_res
    
    # Final local search from the best solution found
    if time.time() < deadline - 5:
        S_final, mask_final, score_final = _local_search(
            best_S, best_mask,
            deadline=deadline - 2,
            T_initial=2.0,
            cooling=0.9998,
            max_iter=30000
        )
        if score_final > best_score:
            best_S, best_mask, best_score = S_final, mask_final, score_final
    
    return sorted(best_S)
