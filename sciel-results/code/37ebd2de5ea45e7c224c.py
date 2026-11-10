import time
import random
import math

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sumset = set()
    prodset = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def generate_smooth_numbers(limit, primes, max_count=50000):
    """Generate smooth numbers up to limit using given primes."""
    smooth = [1]
    i = 0
    while i < len(smooth) and len(smooth) < max_count:
        for p in primes:
            val = smooth[i] * p
            if val <= limit:
                smooth.append(val)
            else:
                break
        i += 1
    return sorted(set(smooth))

def generate_lattice_candidates(limit, max_exp=14):
    """
    Generate candidates as 2^x * 3^y for x,y in [0, max_exp].
    Returns sorted list of distinct values <= limit.
    """
    vals = set()
    for x in range(max_exp + 1):
        for y in range(max_exp + 1):
            v = (2 ** x) * (3 ** y)
            if v <= limit:
                vals.add(v)
    return sorted(vals)

def get_exponents(val):
    """Return (x, y) such that val = 2^x * 3^y, or None if not of that form."""
    x = 0
    y = 0
    v = val
    while v % 2 == 0 and v > 1:
        v //= 2
        x += 1
    while v % 3 == 0 and v > 1:
        v //= 3
        y += 1
    if v != 1:
        return None
    return (x, y)

def lattice_neighbors(val, limit, pool_set, max_dist=2):
    """
    Find lattice neighbors of val in the pool, within Manhattan distance max_dist.
    Returns list of (distance, neighbor_val) sorted by distance.
    """
    exp = get_exponents(val)
    if exp is None:
        return []
    x0, y0 = exp
    results = []
    for dx in range(-max_dist, max_dist + 1):
        for dy in range(-max_dist, max_dist + 1):
            if dx == 0 and dy == 0:
                continue
            x = x0 + dx
            y = y0 + dy
            if x < 0 or y < 0:
                continue
            v = (2 ** x) * (3 ** y)
            if 1 <= v <= limit and v in pool_set:
                dist = abs(dx) + abs(dy)
                results.append((dist, v))
    results.sort(key=lambda t: t[0])
    return results

def local_search_lattice(A, time_limit, rng, candidate_pool, pool_set):
    """
    Local search using lattice-aware single swaps with fast incremental updates.
    """
    A = list(A)
    n = len(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    # Precompute sumset and prodset
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    current_vals = set(A)
    
    improved = True
    no_improve_count = 0
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        no_improve_count += 1
        
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            # Build ordered candidate list: lattice neighbors first, then random pool
            neighbors = lattice_neighbors(old_val, 10000, pool_set, max_dist=2)
            neighbor_vals = [v for _, v in neighbors if v not in current_vals and v != old_val]
            
            # Add some random pool candidates for diversity
            random_candidates = []
            for _ in range(10):
                v = rng.choice(candidate_pool)
                if v not in current_vals and v != old_val and v not in neighbor_vals:
                    random_candidates.append(v)
            
            all_candidates = neighbor_vals + random_candidates
            
            best_new_val = None
            best_est_score = current_score
            
            for new_val in all_candidates:
                if new_val == old_val or new_val in current_vals:
                    continue
                if not (1 <= new_val <= 10000):
                    continue
                
                # Fast delta computation
                new_sumset_add = set()
                new_prodset_add = set()
                old_sumset_remove = set()
                old_prodset_remove = set()
                
                for other in A:
                    if other != old_val:
                        new_sumset_add.add(new_val + other)
                        new_prodset_add.add(new_val * other)
                        old_sumset_remove.add(old_val + other)
                        old_prodset_remove.add(old_val * other)
                
                new_sumset_add.add(2 * new_val)
                new_prodset_add.add(new_val * new_val)
                old_sumset_remove.add(2 * old_val)
                old_prodset_remove.add(old_val * old_val)
                
                # Estimate new sizes
                est_sumset_size = len(sumset) + len(new_sumset_add - sumset) - len(old_sumset_remove & (sumset - new_sumset_add))
                est_prodset_size = len(prodset) + len(new_prodset_add - prodset) - len(old_prodset_remove & (prodset - new_prodset_add))
                
                est_score = max(est_sumset_size, est_prodset_size)
                
                if est_score < best_est_score:
                    best_est_score = est_score
                    best_new_val = new_val
            
            if best_new_val is not None and best_est_score < current_score:
                new_A = A.copy()
                new_A[i] = best_new_val
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    no_improve_count = 0
                    current_vals = set(A)
                    sumset = set(a + b for a in A for b in A)
                    prodset = set(a * b for a in A for b in A)
                    
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
        
        # If no single swap improved, try a few random pair swaps
        if not improved and (time.time() - start_time) < time_limit:
            for _ in range(5):
                if (time.time() - start_time) > time_limit:
                    break
                i, j = rng.sample(range(n), 2)
                old_i, old_j = A[i], A[j]
                
                # Use lattice neighbors for pair swaps too
                neighbors_i = lattice_neighbors(old_i, 10000, pool_set, max_dist=2)
                neighbors_j = lattice_neighbors(old_j, 10000, pool_set, max_dist=2)
                
                new_i = rng.choice([v for _, v in neighbors_i]) if neighbors_i else rng.choice(candidate_pool)
                new_j = rng.choice([v for _, v in neighbors_j]) if neighbors_j else rng.choice(candidate_pool)
                
                if new_i == old_i or new_j == old_j or new_i == new_j:
                    continue
                if new_i in current_vals or new_j in current_vals:
                    continue
                if not (1 <= new_i <= 10000 and 1 <= new_j <= 10000):
                    continue
                
                new_A = A.copy()
                new_A[i] = new_i
                new_A[j] = new_j
                
                score = calc_score(new_A)
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    no_improve_count = 0
                    current_vals = set(A)
                    sumset = set(a + b for a in A for b in A)
                    prodset = set(a * b for a in A for b in A)
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    break
    
    return best_A, best_score

def lattice_local_search(A, time_limit, rng, candidate_pool, pool_set):
    """
    Specialized local search that navigates the 2D exponent lattice.
    Tries to maintain geometric clustering in exponent space.
    """
    A = list(A)
    n = len(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    for iteration in range(300):
        if (time.time() - start_time) > time_limit:
            break
        
        i = rng.randint(0, n - 1)
        old_val = A[i]
        
        # Try lattice moves: shift exponents by small amounts
        exp = get_exponents(old_val)
        if exp is not None:
            x0, y0 = exp
            # Try all small shifts
            for dx in [-2, -1, 1, 2]:
                for dy in [-2, -1, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    x = x0 + dx
                    y = y0 + dy
                    if x < 0 or y < 0:
                        continue
                    new_val = (2 ** x) * (3 ** y)
                    if 1 <= new_val <= 10000 and new_val not in A:
                        new_A = A.copy()
                        new_A[i] = new_val
                        score = calc_score(new_A)
                        if score < current_score:
                            current_score = score
                            A = new_A
                            if score < best_score:
                                best_score = score
                                best_A = list(A)
        
        # Also try multiplicative perturbations
        for k in [2, 3, 5]:
            new_val = old_val * k
            if 1 <= new_val <= 10000 and new_val not in A:
                new_A = A.copy()
                new_A[i] = new_val
                score = calc_score(new_A)
                if score < current_score:
                    current_score = score
                    A = new_A
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    break
            
            if old_val % k == 0:
                new_val = old_val // k
                if new_val >= 1 and new_val not in A:
                    new_A = A.copy()
                    new_A[i] = new_val
                    score = calc_score(new_A)
                    if score < current_score:
                        current_score = score
                        A = new_A
                        if score < best_score:
                            best_score = score
                            best_A = list(A)
                        break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    rng = random.Random(42)
    
    # Generate lattice-based candidate pool: 2^x * 3^y
    lattice_candidates = generate_lattice_candidates(10000, max_exp=14)
    
    # Also generate smooth numbers with more primes for diversity
    primes = [2, 3, 5, 7, 11]
    smooth_numbers = generate_smooth_numbers(10000, primes, max_count=50000)
    
    # Combined candidate pool
    candidate_pool = sorted(set(lattice_candidates + smooth_numbers + list(range(1, 51))))
    pool_set = set(candidate_pool)
    
    best_A = None
    best_score = float('inf')
    
    # Multiple initial candidates
    candidates = []
    
    # 1. Smallest lattice points (2^x * 3^y)
    cand1 = lattice_candidates[:20]
    candidates.append(cand1)
    
    # 2. Powers of 2 and 3
    cand2 = []
    for e2 in range(15):
        for e3 in range(10):
            val = (2 ** e2) * (3 ** e3)
            if val <= 10000 and val not in cand2:
                cand2.append(val)
            if len(cand2) >= 20:
                break
        if len(cand2) >= 20:
            break
    cand2 = sorted(cand2[:20])
    candidates.append(cand2)
    
    # 3. Include 1 and next lattice points
    cand3 = [1] + lattice_candidates[1:19]
    candidates.append(cand3)
    
    # 4. Random selection from lattice
    cand4 = sorted(rng.sample(lattice_candidates[:200], 20))
    candidates.append(cand4)
    
    # 5. Dense small
    cand5 = list(range(1, 21))
    candidates.append(cand5)
    
    # 6. Geometric-like in lattice space
    cand6 = []
    val = 1
    while val <= 10000 and len(cand6) < 20:
        cand6.append(val)
        if val % 2 == 0:
            val = val * 3
        elif val % 3 == 0:
            val = val * 2
        else:
            val = val * 2
    candidates.append(cand6)
    
    # 7. Highly composite
    cand7 = sorted([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36])
    candidates.append(cand7)
    
    # 8. Log-spaced in lattice
    cand8 = []
    for target_log in [i * 0.3 for i in range(20)]:
        target = int(round(2 ** target_log))
        best_lattice = min(lattice_candidates, key=lambda x: abs(x - target))
        if best_lattice not in cand8:
            cand8.append(best_lattice)
    cand8 = sorted(cand8[:20])
    candidates.append(cand8)
    
    # 9. Geometric progression candidates
    for base in [2, 3, 4, 5]:
        gp = []
        val = 1
        while val <= 10000 and len(gp) < 20:
            gp.append(val)
            val *= base
        if len(gp) >= 20:
            candidates.append(gp[:20])
    
    # 10. Lattice diagonal: 2^x * 3^x = 6^x
    cand10 = []
    val = 1
    while val <= 10000 and len(cand10) < 20:
        cand10.append(val)
        val *= 6
    if len(cand10) >= 20:
        candidates.append(cand10[:20])
    
    # 11. Lattice with small perturbations around diagonal
    cand11 = []
    for x in range(20):
        v = (2 ** x) * (3 ** x)
        if v <= 10000:
            cand11.append(v)
            # Add a neighbor
            if x > 0:
                v2 = (2 ** (x-1)) * (3 ** (x+1))
                if v2 <= 10000 and v2 not in cand11:
                    cand11.append(v2)
    cand11 = sorted(set(cand11)[:20])
    if len(cand11) == 20:
        candidates.append(cand11)
    
    # Validate
    valid_candidates = []
    for c in candidates:
        if len(set(c)) == 20 and all(1 <= x <= 10000 for x in c):
            valid_candidates.append(c)
    
    # Evaluate initial
    initial_scores = []
    for c in valid_candidates:
        s = calc_score(c)
        initial_scores.append((s, c))
    
    initial_scores.sort(key=lambda x: x[0])
    
    # Run local search
    time_per_candidate = min(12.0, (deadline - time.time()) * 0.9 / max(1, len(valid_candidates)))
    
    for score, cand in initial_scores:
        if time.time() > deadline - 5.0:
            break
        
        if score < best_score:
            best_score = score
            best_A = list(cand)
        
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        alloc_time = min(time_per_candidate, remaining * 0.8)
        
        improved_A, ls_score = local_search_lattice(cand, alloc_time, rng, candidate_pool, pool_set)
        
        if ls_score < best_score:
            best_score = ls_score
            best_A = improved_A
        
        # Also try lattice-specific local search if time permits
        if (time.time() - start_time) < deadline - 10.0:
            geo_A, geo_score = lattice_local_search(improved_A, min(5.0, (deadline - time.time()) * 0.3), rng, candidate_pool, pool_set)
            if geo_score < best_score:
                best_score = geo_score
                best_A = geo_A
    
    # Final random restarts if time permits
    if time.time() < deadline - 10.0:
        for attempt in range(8):
            if time.time() > deadline - 5.0:
                break
            
            if best_A is not None:
                start_A = best_A.copy()
                # Perturb with lattice awareness
                for _ in range(3):
                    idx = rng.randint(0, 19)
                    old_val = start_A[idx]
                    exp = get_exponents(old_val)
                    if exp is not None:
                        x0, y0 = exp
                        # Try a random lattice shift
                        dx = rng.randint(-2, 2)
                        dy = rng.randint(-2, 2)
                        if dx == 0 and dy == 0:
                            dx = rng.choice([-1, 1])
                        x = x0 + dx
                        y = y0 + dy
                        if x >= 0 and y >= 0:
                            new_val = (2 ** x) * (3 ** y)
                            if 1 <= new_val <= 10000 and new_val not in start_A:
                                start_A[idx] = new_val
                                continue
                    # Fallback: random pool
                    new_val = rng.choice(candidate_pool)
                    while new_val in start_A:
                        new_val = rng.choice(candidate_pool)
                    start_A[idx] = new_val
            else:
                start_A = list(range(1, 21))
            
            if len(set(start_A)) != 20:
                continue
            
            s = calc_score(start_A)
            if s < best_score:
                best_score = s
                best_A = start_A
            
            remaining = deadline - time.time()
            if remaining < 5.0:
                break
            
            improved_A, ls_score = local_search_lattice(start_A, min(10.0, remaining * 0.8), rng, candidate_pool, pool_set)
            if ls_score < best_score:
                best_score = ls_score
                best_A = improved_A
            
            # Try lattice polish
            if (time.time() - start_time) < deadline - 3.0:
                geo_A, geo_score = lattice_local_search(improved_A, min(3.0, (deadline - time.time()) * 0.5), rng, candidate_pool, pool_set)
                if geo_score < best_score:
                    best_score = geo_score
                    best_A = geo_A
    
    if best_A is None:
        best_A = list(range(1, 21))
    
    return best_A
