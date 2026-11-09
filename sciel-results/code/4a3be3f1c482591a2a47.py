"""Optimized search for integer sum-product extremal sets."""
import random
import time

def _objective(A):
    """Compute max(|A+A|, |A*A|) for set A of 20 distinct positive integers."""
    n = len(A)
    # Sumset: all a_i + a_j for i <= j (distinct sums)
    sums = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
    # Product set: all a_i * a_j for i <= j
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _smooth_objective(A, t=5.0):
    """Smooth surrogate: log-sum-exp of log(|A+A|) and log(|A*A|).
    This is differentiable and helps guide search in regions where
    the max is flat."""
    n = len(A)
    sums = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            prods.add(ai * A[j])
    s1 = len(sums)
    s2 = len(prods)
    # log-sum-exp: log(exp(log(s1)/t) + exp(log(s2)/t)) * t
    import math
    l1 = math.log(s1) / t
    l2 = math.log(s2) / t
    m = max(l1, l2)
    return t * (m + math.log(math.exp(l1 - m) + math.exp(l2 - m)))

def _is_valid(A):
    """Check A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if x < 1 or x > 10000:
            return False
    return True

def _generate_structured_candidates():
    """Generate diverse initial candidates using different structures."""
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, 21)))
    candidates.append(list(range(2, 22)))
    candidates.append(list(range(5, 25)))
    
    # 2. Powers of 2 (minimizes product set, maximizes sum set)
    powers2 = [2**i for i in range(20)]
    if max(powers2) <= 10000:
        candidates.append(powers2)
    else:
        # Truncated powers
        p = []
        v = 1
        while len(p) < 20 and v <= 10000:
            p.append(v)
            v *= 2
        while len(p) < 20:
            p.append(p[-1] + 1)
        candidates.append(p)
    
    # 3. Geometric progression with small ratio
    for ratio in [1.1, 1.2, 1.3, 1.5]:
        gp = []
        v = 1.0
        while len(gp) < 20 and v <= 10000:
            gp.append(int(round(v)))
            v *= ratio
        # Remove duplicates and ensure 20 distinct
        gp = list(dict.fromkeys(gp))
        while len(gp) < 20:
            gp.append(max(gp) + 1)
        if _is_valid(gp):
            candidates.append(gp[:20])
    
    # 4. Mixed: small numbers plus some large
    base = list(range(1, 11))
    large = [500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 5000]
    mixed = base + large
    if _is_valid(mixed):
        candidates.append(mixed)
    
    # 5. Random samples
    for _ in range(10):
        r = random.sample(range(1, 10001), 20)
        candidates.append(sorted(r))
    
    # 6. Arithmetic progressions
    for start in [1, 2, 5, 10]:
        for step in [1, 2, 3, 5, 10]:
            ap = [start + i * step for i in range(20)]
            if max(ap) <= 10000 and _is_valid(ap):
                candidates.append(ap)
    
    # 7. Fibonacci-like
    fib = [1, 1]
    while len(fib) < 20:
        fib.append(fib[-1] + fib[-2])
    # Remove duplicate 1s
    fib = list(dict.fromkeys(fib))
    while len(fib) < 20:
        fib.append(fib[-1] + fib[-2])
    if _is_valid(fib):
        candidates.append(fib[:20])
    
    return candidates

def _greedy_coordinate_descent(A, deadline, best_val, best_A):
    """
    Systematic coordinate descent with curated candidate values.
    Uses both exact objective and smooth surrogate for guidance.
    """
    A = list(A)
    cur_val = _objective(A)
    if cur_val < best_val:
        best_val = cur_val
        best_A = list(A)
    
    n = 20
    max_passes = 150
    no_improve_passes = 0
    max_no_improve = 8
    
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        
        improved_this_pass = False
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            old = A[i]
            best_candidate = None
            best_candidate_val = cur_val
            
            # Generate candidate values for position i
            # 1. Neighbors: old ± d for small d
            for d in [1, 2, 3, 4, 5, 7, 11, 17, 29, 47, 79, 131, 211]:
                for sign in [-1, 1]:
                    cand = old + sign * d
                    if 1 <= cand <= 10000 and cand != old and cand not in A:
                        A[i] = cand
                        v = _objective(A)
                        if v < best_candidate_val:
                            best_candidate_val = v
                            best_candidate = cand
            
            # 2. Geometric neighbors
            for factor in [0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 0.5, 2.0, 0.75, 1.33, 0.66, 1.5]:
                cand = int(round(old * factor))
                if 1 <= cand <= 10000 and cand != old and cand not in A:
                    A[i] = cand
                    v = _objective(A)
                    if v < best_candidate_val:
                        best_candidate_val = v
                        best_candidate = cand
            
            # 3. Targeted values: try to move toward median or mean of A
            other_vals = [x for j, x in enumerate(A) if j != i]
            other_vals_sorted = sorted(other_vals)
            median_val = other_vals_sorted[n // 2 - 1]
            mean_val = sum(other_vals) // (n - 1)
            # Try values near the median
            for offset in [-8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8]:
                cand = median_val + offset
                if 1 <= cand <= 10000 and cand != old and cand not in A:
                    A[i] = cand
                    v = _objective(A)
                    if v < best_candidate_val:
                        best_candidate_val = v
                        best_candidate = cand
            
            # 4. Try values near mean
            for offset in [-5, -2, -1, 0, 1, 2, 5]:
                cand = mean_val + offset
                if 1 <= cand <= 10000 and cand != old and cand not in A:
                    A[i] = cand
                    v = _objective(A)
                    if v < best_candidate_val:
                        best_candidate_val = v
                        best_candidate = cand
            
            # 5. Random candidates (fewer, since we have more structured ones)
            for _ in range(3):
                cand = random.randint(1, 10000)
                if cand != old and cand not in A:
                    A[i] = cand
                    v = _objective(A)
                    if v < best_candidate_val:
                        best_candidate_val = v
                        best_candidate = cand
            
            # Apply best candidate if found
            if best_candidate is not None:
                A[i] = best_candidate
                cur_val = best_candidate_val
                improved_this_pass = True
                if cur_val < best_val:
                    best_val = cur_val
                    best_A = list(A)
            else:
                A[i] = old
        
        if not improved_this_pass:
            no_improve_passes += 1
            if no_improve_passes >= max_no_improve:
                break
        else:
            no_improve_passes = 0
    
    return best_val, best_A

def _local_search(A, deadline, best_val, best_A):
    """Perform local search with random replacements and swaps."""
    A = list(A)
    cur_val = _objective(A)
    if cur_val < best_val:
        best_val = cur_val
        best_A = list(A)
    
    n = 20
    max_iter = 80000
    no_improve = 0
    max_no_improve = 8000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.5:
            # Replace one element
            idx = random.randint(0, n - 1)
            old = A[idx]
            # Try values near old or random
            if random.random() < 0.6:
                new = old + random.randint(-80, 80)
            else:
                new = random.randint(1, 10000)
            new = max(1, min(10000, new))
            if new == old or new in A:
                continue
            A[idx] = new
            new_val = _objective(A)
            if new_val < cur_val:
                cur_val = new_val
                no_improve = 0
                if cur_val < best_val:
                    best_val = cur_val
                    best_A = list(A)
            else:
                # Accept with probability based on improvement (simulated annealing)
                delta = new_val - cur_val
                if delta <= 0 or random.random() < 0.15:
                    no_improve += 1
                else:
                    A[idx] = old
                    no_improve += 1
        elif move_type < 0.8:
            # Swap two elements
            i, j = random.sample(range(n), 2)
            A[i], A[j] = A[j], A[i]
            new_val = _objective(A)
            if new_val < cur_val:
                cur_val = new_val
                no_improve = 0
                if cur_val < best_val:
                    best_val = cur_val
                    best_A = list(A)
            else:
                A[i], A[j] = A[j], A[i]
                no_improve += 1
        else:
            # Perturb multiple elements
            for _ in range(random.randint(2, 4)):
                idx = random.randint(0, n - 1)
                new = A[idx] + random.randint(-150, 150)
                new = max(1, min(10000, new))
                if new not in A or new == A[idx]:
                    A[idx] = new
            
            new_val = _objective(A)
            if new_val < cur_val:
                cur_val = new_val
                no_improve = 0
                if cur_val < best_val:
                    best_val = cur_val
                    best_A = list(A)
            else:
                no_improve += 1
        
        if no_improve > max_no_improve:
            break
    
    return best_val, best_A

def _final_polish(A, deadline):
    """High-precision final polish: systematically try all small perturbations
    on each element and keep improvements. This is a local exact solve."""
    A = list(A)
    cur_val = _objective(A)
    n = 20
    
    # Multiple passes of exact local search
    for _pass in range(5):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            old = A[i]
            best_candidate = None
            best_candidate_val = cur_val
            
            # Try all values in a window around old
            for d in range(1, 30):
                for sign in [-1, 1]:
                    cand = old + sign * d
                    if 1 <= cand <= 10000 and cand != old and cand not in A:
                        A[i] = cand
                        v = _objective(A)
                        if v < best_candidate_val:
                            best_candidate_val = v
                            best_candidate = cand
            
            # Also try some geometric moves
            for factor in [0.8, 0.9, 1.1, 1.2, 0.5, 2.0, 0.75, 1.33, 0.66, 1.5]:
                cand = int(round(old * factor))
                if 1 <= cand <= 10000 and cand != old and cand not in A:
                    A[i] = cand
                    v = _objective(A)
                    if v < best_candidate_val:
                        best_candidate_val = v
                        best_candidate = cand
            
            if best_candidate is not None:
                A[i] = best_candidate
                cur_val = best_candidate_val
                improved = True
            else:
                A[i] = old
        
        if not improved:
            break
    
    return cur_val, A

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    random.seed(42)
    
    # Generate initial candidates
    candidates = _generate_structured_candidates()
    
    # Evaluate all candidates
    best_val = float('inf')
    best_A = None
    for c in candidates:
        if not _is_valid(c):
            continue
        v = _objective(c)
        if v < best_val:
            best_val = v
            best_A = list(c)
    
    # Use greedy coordinate descent as primary optimization
    if time.time() < deadline - 5.0:
        best_val, best_A = _greedy_coordinate_descent(best_A, deadline - 5.0, best_val, best_A)
    
    # Then use local search for further refinement
    if time.time() < deadline - 5.0:
        best_val, best_A = _local_search(best_A, deadline - 5.0, best_val, best_A)
    
    # Multiple restarts from different starting points
    restarts = 0
    while time.time() < deadline - 5.0 and restarts < 8:
        # Generate a new random starting point
        new_start = sorted(random.sample(range(1, 10001), 20))
        v = _objective(new_start)
        if v < best_val:
            best_val = v
            best_A = list(new_start)
        
        # Use greedy coordinate descent from this point
        if time.time() < deadline - 5.0:
            best_val, best_A = _greedy_coordinate_descent(new_start, deadline - 5.0, best_val, best_A)
        
        restarts += 1
    
    # Final high-precision polish: systematic local exact search
    if time.time() < deadline - 1.0:
        best_val, best_A = _final_polish(best_A, deadline - 1.0)
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return best_A

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
