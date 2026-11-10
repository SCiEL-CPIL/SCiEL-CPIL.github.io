"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from bisect import bisect_left

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    # Precompute multiples for each divisor
    multiples = []
    for d in divisors:
        multiples.append(list(range(d, max_val + 1, d)))
    
    # Helper: find smallest multiple >= target not in used, within [target, target+limit)
    def find_mult(mults, target, limit, used):
        lo = bisect_left(mults, target)
        for idx in range(lo, len(mults)):
            v = mults[idx]
            if v >= target + limit:
                break
            if v not in used:
                return v
        return None
    
    # Greedy assignment with a given order and minimum anchor
    def greedy_assign(anchor_div_idx, anchor_val, order, span_limit):
        used = {anchor_val}
        result = [0] * n
        result[anchor_div_idx] = anchor_val
        max_assigned = anchor_val
        
        for j in order:
            if j == anchor_div_idx:
                continue
            v = find_mult(multiples[j], anchor_val, span_limit, used)
            if v is None:
                return None
            used.add(v)
            result[j] = v
            if v > max_assigned:
                max_assigned = v
        return result, max_assigned - anchor_val
    
    # Initial solution: the divisors themselves
    best_result = divisors[:]
    best_span = max(best_result) - min(best_result)
    
    deadline = time.time() + 75
    
    # Generate several assignment orders to try
    # Use a few heuristics: by divisor size, reverse, etc.
    orders = [
        list(range(n)),
        list(range(n-1, -1, -1)),
        sorted(range(n), key=lambda i: divisors[i]),
        sorted(range(n), key=lambda i: -divisors[i]),
    ]
    # Add a few random permutations
    import random
    random.seed(42)
    for _ in range(4):
        perm = list(range(n))
        random.shuffle(perm)
        orders.append(perm)
    
    # Search over candidate minimums for each anchor divisor
    for anchor_i, d in enumerate(divisors):
        if time.time() > deadline:
            break
        
        # Try multiples of d as anchor values
        # Limit: only try values where we could potentially beat best_span
        m = d
        while m <= max_val:
            if time.time() > deadline:
                break
            
            for order in orders:
                res = greedy_assign(anchor_i, m, order, best_span)
                if res is not None:
                    result, span = res
                    if span < best_span:
                        best_span = span
                        best_result = result[:]
            
            m += d
    
    # Local search: try to improve by moving individual elements
    if best_result is not None and time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                d = divisors[i]
                cur = best_result[i]
                # Try several offsets
                for offset in range(-3, 4):
                    if offset == 0:
                        continue
                    new_val = cur + offset * d
                    if new_val < 1 or new_val > max_val:
                        continue
                    # Check distinctness
                    distinct = True
                    for j in range(n):
                        if j != i and best_result[j] == new_val:
                            distinct = False
                            break
                    if not distinct:
                        continue
                    trial = best_result[:]
                    trial[i] = new_val
                    trial_span = max(trial) - min(trial)
                    if trial_span < best_span:
                        best_span = trial_span
                        best_result = trial
                        improved = True
                        break
    
    # Final high-precision polish: try all pairs of adjacent moves
    if best_result is not None and time.time() < deadline:
        for i in range(n):
            if time.time() > deadline:
                break
            d_i = divisors[i]
            for j in range(n):
                if i == j or time.time() > deadline:
                    continue
                d_j = divisors[j]
                cur_i = best_result[i]
                cur_j = best_result[j]
                for off_i in [-1, 1]:
                    for off_j in [-1, 1]:
                        new_i = cur_i + off_i * d_i
                        new_j = cur_j + off_j * d_j
                        if new_i < 1 or new_i > max_val or new_j < 1 or new_j > max_val:
                            continue
                        # Check all distinct
                        trial = best_result[:]
                        trial[i] = new_i
                        trial[j] = new_j
                        if len(set(trial)) < n:
                            continue
                        trial_span = max(trial) - min(trial)
                        if trial_span < best_span:
                            best_span = trial_span
                            best_result = trial
    
    # Ensure return type is list of Python ints
    result = [int(x) for x in best_result]
    return result

# EVOLVE-BLOCK-END
