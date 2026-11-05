import time
import random
import math

def _set_sizes(A):
    """Compute |A+A| and |A*A| with equal summands/factors allowed."""
    A = list(A)
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return len(sums), len(prods)

def _objective(A):
    s, p = _set_sizes(A)
    return max(s, p)

def _validate(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _greedy_construct(n, max_val, deadline, start_elements=None):
    """
    Greedy constructive: build set element by element,
    choosing each new element to minimize incremental max(|A+A|,|A*A|).
    """
    if start_elements is None:
        start_elements = []
    
    A = sorted(start_elements)
    if not A:
        return None
    
    sums = set()
    prods = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    
    used = set(A)
    
    while len(A) < n:
        best_candidate = None
        best_cost = float('inf')
        
        # Candidate pool: small numbers (for sum compactness) and 
        # numbers that create product overlap
        candidates = set()
        
        # Small integers 1..80
        for v in range(1, 81):
            if v not in used:
                candidates.add(v)
        
        # Numbers related to existing elements (products, sums)
        for a in A:
            for b in A:
                p = a * b
                if 1 <= p <= max_val and p not in used:
                    candidates.add(p)
                s = a + b
                if 1 <= s <= max_val and s not in used:
                    candidates.add(s)
            # Multiples
            for m in range(2, 12):
                v = a * m
                if 1 <= v <= max_val and v not in used:
                    candidates.add(v)
            # Divisors
            for m in range(2, 12):
                if a % m == 0:
                    v = a // m
                    if 1 <= v <= max_val and v not in used:
                        candidates.add(v)
        
        # Log-spaced candidates
        for e in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            for mult in range(1, 15):
                v = e * mult
                if 1 <= v <= max_val and v not in used:
                    candidates.add(v)
        
        # Some random candidates for diversity
        for _ in range(50):
            v = random.randint(1, max_val)
            if v not in used:
                candidates.add(v)
        
        # Evaluate candidates (limit to keep time bounded)
        cand_list = list(candidates)
        if len(cand_list) > 300:
            random.shuffle(cand_list)
            cand_list = cand_list[:300]
        
        for v in cand_list:
            if v in used:
                continue
            # Compute new sums and prods incrementally
            new_sum_vals = 0
            new_prod_vals = 0
            
            for a in A:
                s = a + v
                if s not in sums:
                    new_sum_vals += 1
                p = a * v
                if p not in prods:
                    new_prod_vals += 1
            
            # Also v+v and v*v (diagonal)
            s_diag = v + v
            if s_diag not in sums:
                new_sum_vals += 1
            p_diag = v * v
            if p_diag not in prods:
                new_prod_vals += 1
            
            total_sums = len(sums) + new_sum_vals
            total_prods = len(prods) + new_prod_vals
            cost = max(total_sums, total_prods)
            
            if cost < best_cost:
                best_cost = cost
                best_candidate = v
        
        if best_candidate is None:
            # Fallback: pick smallest unused
            v = 1
            while v in used:
                v += 1
            best_candidate = v
        
        A.append(best_candidate)
        used.add(best_candidate)
        A.sort()
        
        # Update sums and prods
        for i in range(len(A) - 1):
            ai = A[i]
            sums.add(ai + best_candidate)
            prods.add(ai * best_candidate)
        sums.add(best_candidate + best_candidate)
        prods.add(best_candidate * best_candidate)
        
        if time.time() > deadline:
            break
    
    if len(A) < n:
        # Fill remaining
        v = 1
        while len(A) < n:
            if v not in used:
                A.append(v)
                used.add(v)
            v += 1
        A.sort()
    
    return A

def _local_refine(A, deadline, max_rounds=5000):
    """
    Local refinement: try replacing each element with a better one.
    """
    n = len(A)
    current_cost = _objective(A)
    best_A = A[:]
    best_cost = current_cost
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(n):
            if time.time() > deadline:
                break
            
            old_val = A[idx]
            # Try nearby values and structured candidates
            trial_vals = set()
            # Nearby
            for d in range(-10, 11):
                v = old_val + d
                if 1 <= v <= 10000:
                    trial_vals.add(v)
            # Small values
            for v in range(1, 30):
                trial_vals.add(v)
            # Products of pairs in A (excluding idx)
            others = [A[i] for i in range(n) if i != idx]
            for a in others[:15]:
                for m in range(2, 10):
                    v = a * m
                    if 1 <= v <= 10000:
                        trial_vals.add(v)
                    if a % m == 0:
                        v = a // m
                        if 1 <= v <= 10000:
                            trial_vals.add(v)
            # Sums of pairs
            for i in range(len(others)):
                for j in range(i, len(others)):
                    v = others[i] + others[j]
                    if 1 <= v <= 10000:
                        trial_vals.add(v)
            
            # Limit trial values
            trial_list = list(trial_vals)
            if len(trial_list) > 100:
                random.shuffle(trial_list)
                trial_list = trial_list[:100]
            
            for v in trial_list:
                if v == old_val or v in [A[i] for i in range(n) if i != idx]:
                    continue
                A[idx] = v
                new_cost = _objective(A)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = A[:]
                    break
                else:
                    A[idx] = old_val
        
        if not improved:
            break
    
    return best_A, best_cost

def _generate_grid(base1, base2, i_max, j_max):
    """Generate {base1^i * base2^j : i in 0..i_max, j in 0..j_max}"""
    A = []
    for i in range(i_max + 1):
        for j in range(j_max + 1):
            v = (base1 ** i) * (base2 ** j)
            if v <= 10000:
                A.append(v)
    return A

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 20
    max_val = 10000
    
    random.seed(42)
    
    best_A = list(range(1, 21))
    best_cost = _objective(best_A)
    
    # Strategy 1: Grid-based initial candidates
    grid_configs = [
        (2, 3, 3, 4),
        (2, 3, 4, 3),
        (2, 5, 3, 4),
        (2, 5, 4, 3),
        (3, 5, 3, 4),
        (2, 7, 3, 4),
        (2, 7, 4, 3),
        (2, 3, 1, 9),
        (2, 3, 9, 1),
        (3, 4, 3, 4),
        (2, 4, 4, 3),
        (4, 5, 3, 4),
        (2, 3, 2, 6),
        (2, 3, 6, 2),
    ]
    
    candidates = []
    for b1, b2, im, jm in grid_configs:
        A = _generate_grid(b1, b2, im, jm)
        if len(A) == n and _validate(A):
            candidates.append(A)
    
    # Arithmetic progression
    candidates.append(list(range(1, 21)))
    candidates.append(list(range(1, 21, 1)))
    candidates.append([2*i+1 for i in range(20)])
    
    # Evaluate all grid candidates
    for cand in candidates:
        if not _validate(cand):
            continue
        obj = _objective(cand)
        if obj < best_cost:
            best_cost = obj
            best_A = list(cand)
    
    # Strategy 2: Greedy construction with different seeds
    seed_sets = [
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 5, 7],
        [1, 3, 5, 7],
        [1, 2, 4, 8],
        [1, 2, 3, 5, 7],
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
        [1, 2, 4, 8, 16],
    ]
    
    for seed in seed_sets:
        if time.time() > deadline:
            break
        A = _greedy_construct(n, max_val, deadline, start_elements=seed)
        if A is None or len(A) != n:
            continue
        cost = _objective(A)
        if cost < best_cost:
            best_cost = cost
            best_A = A[:]
    
    # Strategy 3: Refine the best result so far
    if time.time() < deadline:
        refined_A, refined_cost = _local_refine(best_A, deadline, max_rounds=3000)
        if refined_cost < best_cost:
            best_cost = refined_cost
            best_A = refined_A
    
    # Strategy 4: Try refining other good candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if not _validate(cand):
            continue
        obj = _objective(cand)
        if obj <= best_cost + 30:
            new_set, new_obj = _local_refine(cand, deadline, max_rounds=1500)
            if new_obj < best_cost:
                best_cost = new_obj
                best_A = new_set
    
    # Strategy 5: Random restarts with greedy + local search
    while time.time() < deadline - 5.0:
        A = random.sample(range(1, 10001), 20)
        obj = _objective(A)
        if obj < best_cost:
            best_A, best_cost = _local_refine(A, deadline, max_rounds=2000)
        else:
            # Quick local search
            new_set, new_obj = _local_refine(A, deadline, max_rounds=500)
            if new_obj < best_cost:
                best_A = new_set
                best_cost = new_obj
    
    return best_A
