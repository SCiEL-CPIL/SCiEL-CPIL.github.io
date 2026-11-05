import time
import random
import numpy as np

def set_sizes(A):
    """Compute sizes of A+A and A*A with equal summands/factors allowed."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # Use upper triangle including diagonal for sums and products
    i_idx, j_idx = np.triu_indices(n)
    sums = A[i_idx] + A[j_idx]
    prods = A[i_idx] * A[j_idx]
    return len(np.unique(sums)), len(np.unique(prods))

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s
    
    best_set = None
    best_obj = float('inf')
    
    def try_candidate(A):
        nonlocal best_set, best_obj
        if len(A) != 20:
            return
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        obj = objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = A[:]

    # === STRUCTURAL CHANGE: Focus on 7-smooth numbers ===
    # Generate all 7-smooth numbers <= 10000
    smooth_numbers = set()
    for a in range(0, 14):  # 2^13 = 8192
        for b in range(0, 9):  # 3^8 = 6561
            for c in range(0, 6):  # 5^5 = 3125
                for d in range(0, 4):  # 7^3 = 343
                    val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if 1 <= val <= 10000:
                        smooth_numbers.add(val)
    
    smooth_list = sorted(smooth_numbers)
    n_smooth = len(smooth_list)
    
    # Strategy 1: Select 20 smooth numbers that form a "dense" cluster
    # to maximize collisions in both sumset and productset
    # Try sliding windows of 20 consecutive smooth numbers
    for i in range(0, min(n_smooth - 20, 100)):
        A = smooth_list[i:i+20]
        if len(set(A)) == 20:
            try_candidate(A)
    
    # Strategy 2: Select smooth numbers from a small range (dense in log space)
    # Try different log-space windows
    log_smooth = [(x, np.log(x)) for x in smooth_list]
    for log_min in np.arange(0, 8, 0.1):
        for log_max in np.arange(log_min + 1, min(log_min + 4, 10), 0.5):
            A = [x for x, lx in log_smooth if log_min <= lx <= log_max]
            if len(A) >= 20:
                # Take first 20 or random sample
                A = A[:20]
                if len(set(A)) == 20:
                    try_candidate(A)
    
    # Strategy 3: Greedy selection - start with a seed, add smooth numbers
    # that minimize the increase in max(|A+A|, |A*A|)
    random.seed(42)
    for seed_val in smooth_list[:50]:
        if time.time() > deadline - 10:
            break
        A = [seed_val]
        used = {seed_val}
        cur_obj = objective(A)
        
        for _ in range(19):
            if time.time() > deadline - 5:
                break
            best_add = None
            best_new_obj = cur_obj + 1000  # high threshold
            
            # Try adding each unused smooth number
            for cand in smooth_list:
                if cand in used:
                    continue
                new_A = A + [cand]
                new_obj = objective(new_A)
                if new_obj < best_new_obj:
                    best_new_obj = new_obj
                    best_add = cand
            
            if best_add is not None:
                A.append(best_add)
                used.add(best_add)
                cur_obj = best_new_obj
            else:
                break
        
        if len(A) == 20 and len(set(A)) == 20:
            try_candidate(A)
    
    # Strategy 4: Exponent lattice selection
    # Represent smooth numbers as (a,b,c,d) where value = 2^a * 3^b * 5^c * 7^d
    # Select 20 exponent vectors that cluster together
    exponent_vectors = []
    for a in range(0, 14):
        for b in range(0, 9):
            for c in range(0, 6):
                for d in range(0, 4):
                    val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if 1 <= val <= 10000:
                        exponent_vectors.append((a, b, c, d, val))
    
    # Try selecting vectors from a small box in exponent space
    for center in [(2,2,1,0), (3,2,1,0), (4,3,1,0), (5,3,2,0), (6,4,2,1)]:
        for radius in [1, 2, 3]:
            A = []
            for a, b, c, d, val in exponent_vectors:
                if (abs(a - center[0]) <= radius and 
                    abs(b - center[1]) <= radius and 
                    abs(c - center[2]) <= radius and 
                    abs(d - center[3]) <= radius):
                    A.append(val)
            if len(A) >= 20:
                A = sorted(A[:20])
                if len(set(A)) == 20:
                    try_candidate(A)
    
    # Strategy 5: Random search within smooth numbers with local improvement
    if best_set is None:
        best_set = smooth_list[:20]
        best_obj = objective(best_set)
    
    # Local search: replace elements with other smooth numbers
    random.seed(123)
    
    def local_search_smooth(start_set, max_iters=20000):
        nonlocal best_set, best_obj
        A = list(start_set)
        cur_obj = objective(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
        it = 0
        no_improve = 0
        while it < max_iters:
            if time.time() > deadline - 5:
                return
            it += 1
            i = random.randrange(20)
            x = A[i]
            
            # Candidate replacements: other smooth numbers
            # Prefer nearby values in log space
            log_x = np.log(x)
            cands = []
            for y in smooth_list:
                if y == x or y in A:
                    continue
                if abs(np.log(y) - log_x) < 2.0:  # within factor of ~7
                    cands.append(y)
            
            if not cands:
                cands = [y for y in smooth_list if y != x and y not in A]
            
            if not cands:
                continue
            
            # Sample a few candidates
            sample_size = min(10, len(cands))
            chosen = random.sample(cands, sample_size)
            
            for nv in chosen:
                old = A[i]
                A[i] = nv
                new_obj = objective(A)
                if new_obj <= cur_obj:
                    cur_obj = new_obj
                    no_improve = 0
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_set = A[:]
                    break
                else:
                    A[i] = old
            
            no_improve += 1
            if no_improve > 2000:
                break
    
    # Run local search from best candidate
    local_search_smooth(best_set, max_iters=15000)
    
    # Run from a few other structured starts
    other_starts = []
    
    # Dense cluster starts
    for i in [0, 10, 20, 30]:
        if i + 20 < n_smooth:
            other_starts.append(smooth_list[i:i+20])
    
    # Small number starts
    small_smooth = [x for x in smooth_list if x <= 100]
    if len(small_smooth) >= 20:
        other_starts.append(small_smooth[:20])
    
    for sp in other_starts:
        if time.time() > deadline - 8:
            break
        if sp == best_set:
            continue
        local_search_smooth(sp, max_iters=10000)
    
    # Final polish: fine-tune each element
    if time.time() < deadline - 3:
        A = list(best_set)
        cur_obj = best_obj
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(20):
                if time.time() > deadline - 2:
                    break
                x = A[i]
                # Try nearby smooth numbers
                log_x = np.log(x)
                nearby = [y for y in smooth_list 
                         if y != x and y not in A and abs(np.log(y) - log_x) < 0.5]
                for nv in nearby:
                    old = A[i]
                    A[i] = nv
                    new_obj = objective(A)
                    if new_obj < cur_obj:
                        cur_obj = new_obj
                        improved = True
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_set = A[:]
                        break
                    else:
                        A[i] = old
            if not improved:
                break
    
    return best_set
