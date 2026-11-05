import time
import random
import numpy as np

def _set_sizes(A):
    A = list(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def _objective(A):
    s, p = _set_sizes(A)
    return max(s, p)

def _incremental_cost(A, candidate):
    """Compute how much adding 'candidate' to set A increases max(|A+A|, |A*A|)."""
    A = list(A)
    n = len(A)
    
    # Current sizes
    cur_sums = set()
    cur_prods = set()
    for i in range(n):
        for j in range(i, n):
            cur_sums.add(A[i] + A[j])
            cur_prods.add(A[i] * A[j])
    
    cur_s = len(cur_sums)
    cur_p = len(cur_prods)
    cur_obj = max(cur_s, cur_p)
    
    # New sizes with candidate added
    new_sums = set(cur_sums)
    new_prods = set(cur_prods)
    
    # Add sums: candidate + each existing element, and candidate + candidate
    new_sums.add(candidate + candidate)
    for x in A:
        new_sums.add(candidate + x)
    
    # Add products: candidate * each existing element, and candidate * candidate
    new_prods.add(candidate * candidate)
    for x in A:
        new_prods.add(candidate * x)
    
    new_s = len(new_sums)
    new_p = len(new_prods)
    new_obj = max(new_s, new_p)
    
    return new_obj - cur_obj

def _greedy_construct(deadline):
    """Build a set of 20 elements greedily, choosing each to minimize incremental cost."""
    # Start with a reasonable base
    base_elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    A = list(base_elements)
    
    # Candidate pool: try various ranges
    # Focus on small values first (they tend to create fewer distinct sums/products)
    candidates = list(range(11, 60))
    random.shuffle(candidates)
    
    while len(A) < 20 and time.time() < deadline - 5:
        best_cand = None
        best_cost = float('inf')
        
        # Try candidates in order, but also add some random ones
        search_list = candidates[:30] + [random.randint(1, 10000) for _ in range(10)]
        
        for cand in search_list:
            if cand in A or cand < 1 or cand > 10000:
                continue
            cost = _incremental_cost(A, cand)
            if cost < best_cost:
                best_cost = cost
                best_cand = cand
        
        if best_cand is None:
            # Fallback: pick a random unused value
            while True:
                best_cand = random.randint(1, 10000)
                if best_cand not in A:
                    break
        
        A.append(best_cand)
        A.sort()
    
    return A

def _local_refine(A, deadline):
    """Local search refinement: try replacing each element with a better one."""
    A = list(A)
    cur_obj = _objective(A)
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(len(A)):
            old_val = A[i]
            best_cand = old_val
            best_obj = cur_obj
            
            # Generate candidates: nearby values, multiples, random
            candidates = set()
            for d in range(-10, 11):
                candidates.add(old_val + d)
            for factor in [0.5, 0.7, 1.3, 1.5, 2.0, 3.0]:
                candidates.add(int(old_val * factor))
            candidates.add(random.randint(1, 10000))
            
            for cand in candidates:
                if cand < 1 or cand > 10000 or cand in A:
                    continue
                A[i] = cand
                new_obj = _objective(A)
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_cand = cand
                A[i] = old_val
            
            if best_cand != old_val:
                A[i] = best_cand
                cur_obj = best_obj
                improved = True
    
    return A

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s
    
    random.seed(42)
    
    best_set = None
    best_obj = float('inf')
    
    # Strategy 1: Greedy construction from different starting bases
    starting_bases = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23],  # primes
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 12],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]
    
    for base in starting_bases:
        if time.time() > deadline - 15:
            break
        
        # Build greedily from this base
        A = list(base)
        candidates = list(range(1, 100))
        random.shuffle(candidates)
        
        while len(A) < 20 and time.time() < deadline - 10:
            best_cand = None
            best_cost = float('inf')
            
            search_list = candidates[:40] + [random.randint(1, 10000) for _ in range(15)]
            
            for cand in search_list:
                if cand in A or cand < 1 or cand > 10000:
                    continue
                cost = _incremental_cost(A, cand)
                if cost < best_cost:
                    best_cost = cost
                    best_cand = cand
            
            if best_cand is None:
                while True:
                    best_cand = random.randint(1, 10000)
                    if best_cand not in A:
                        break
            
            A.append(best_cand)
            A.sort()
        
        if len(A) == 20:
            # Refine locally
            A = _local_refine(A, deadline - 5)
            obj = _objective(A)
            if obj < best_obj:
                best_obj = obj
                best_set = A[:]
    
    # Strategy 2: Also try the simple geometric seeds as fallback
    geometric_seeds = []
    for r in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        gp = [int(round(r ** i)) for i in range(20)]
        seen = set()
        uniq = []
        for x in gp:
            if x not in seen and 1 <= x <= 10000:
                seen.add(x)
                uniq.append(x)
        while len(uniq) < 20:
            uniq.append(uniq[-1] + 1)
        geometric_seeds.append(uniq[:20])
    
    for s in geometric_seeds:
        if time.time() > deadline - 10:
            break
        if len(set(s)) != 20 or any(x < 1 or x > 10000 for x in s):
            continue
        s = _local_refine(s, deadline - 5)
        obj = _objective(s)
        if obj < best_obj:
            best_obj = obj
            best_set = s[:]
    
    # Final polish
    if best_set is not None and time.time() < deadline - 3:
        best_set = _local_refine(best_set, deadline)
    
    if best_set is None:
        # Absolute fallback
        best_set = list(range(1, 21))
    
    return best_set
