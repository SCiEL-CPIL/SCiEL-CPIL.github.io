import time
import random
import math

def _calc_score(A):
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _generate_smooth(primes, max_val, count):
    vals = set([1])
    for p in primes:
        new_vals = set()
        for v in vals:
            x = v
            while x * p <= max_val:
                x *= p
                new_vals.add(x)
        vals.update(new_vals)
    vals = sorted(vals)
    return vals[:count]

def _local_search(A, timeout):
    A = sorted(A)
    current_score = _calc_score(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    n = len(A)
    max_val = 10000
    
    while time.time() - start_time < timeout:
        A_set = set(A)
        best_new_score = current_score
        best_swap = None
        
        for i in range(n):
            old_val = A[i]
            # We can break early if we find a score of 0? No, min score is > 0.
            # We can break inner loop if time is up? 
            # Better to check time periodically.
            
            for v in range(1, max_val + 1):
                if v == old_val or v in A_set:
                    continue
                
                A[i] = v
                new_score = _calc_score(A)
                A[i] = old_val
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_swap = (i, v)
                    
            # Check time inside i-loop
            if time.time() - start_time > timeout:
                break
                
        if best_swap is not None:
            i, v = best_swap
            A[i] = v
            current_score = best_new_score
            if current_score < best_score:
                best_score = current_score
                best_A = list(A)
        else:
            break
            
    return sorted(best_A), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    candidates = []
    
    # 1. 5-smooth
    s5 = _generate_smooth([2, 3, 5, 7], 10000, 20)
    if len(s5) == 20:
        candidates.append(s5)
        
    # 2. 4-smooth
    s4 = _generate_smooth([2, 3, 5], 10000, 20)
    if len(s4) == 20:
        candidates.append(s4)
        
    # 3. Powers of 2
    p2 = [2**i for i in range(20)]
    if max(p2) <= 10000:
        candidates.append(p2)
        
    # 4. Random
    random.seed(42)
    for _ in range(5):
        candidates.append(sorted(random.sample(range(1, 10001), 20)))
        
    best_A = None
    best_score = float('inf')
    
    # Allocate time to each candidate
    # If we have 5 candidates and 50s, we give 10s each.
    # But we should prioritize the promising ones (smooth).
    # Let's run smooth ones first.
    
    # Reorder candidates: Smooth first
    # candidates[0] is 5-smooth, [1] is 4-smooth.
    
    time_per_candidate = (deadline - time.time()) / len(candidates)
    
    for cand in candidates:
        if time.time() > deadline - 5.0:
            break
            
        cand = sorted(set(x for x in cand if 1 <= x <= 10000))
        if len(cand) < 20:
            continue
            
        # Evaluate initial
        score = _calc_score(cand)
        if score < best_score:
            best_score = score
            best_A = list(cand)
            
        # Local search
        # Use remaining time for this candidate, but cap it to not starve others?
        # Actually, just use a fixed chunk.
        chunk_time = min(10.0, deadline - time.time() - 2.0)
        if chunk_time > 1.0:
            A_res, score_res = _local_search(cand, chunk_time)
            if score_res < best_score:
                best_score = score_res
                best_A = A_res
                
    if best_A is None:
        best_A = list(range(1, 21))
        
    return best_A
