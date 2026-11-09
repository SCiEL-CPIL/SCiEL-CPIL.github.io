"""Program using a model-and-optimize approach for large integer sets with unique triple sums."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_all_sums(selected):
    """Compute set of all unordered triple sums (with replacement). Returns set or None on collision."""
    n = len(selected)
    if n == 0:
        return set()
    arr = np.array(selected, dtype=np.int64)
    # Build all unordered triples with replacement: i <= j <= k
    idx_i = np.repeat(np.arange(n), n * (n + 1) // 2)
    # For each i, j ranges from i to n-1, and for each j, k ranges from j to n-1
    # Total count = C(n+2, 3) = n(n+1)(n+2)/6
    total = n * (n + 1) * (n + 2) // 6
    sums_arr = np.empty(total, dtype=np.int64)
    pos = 0
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums_arr[pos] = arr[i] + arr[j] + arr[k]
                pos += 1
    # Check for duplicates
    if len(np.unique(sums_arr)) != total:
        return None
    return set(sums_arr.tolist())

def _check_valid(selected):
    """Return True if all unordered triple sums (with replacement) are unique."""
    return _compute_all_sums(selected) is not None

def _incremental_check_add(selected, x, existing_sums):
    """Check if adding x to selected keeps all triple sums unique.
    x must be >= max(selected). Returns (True, updated_sums) if valid, (False, None) otherwise."""
    n = len(selected)
    new_sums = set()
    
    # Check (x,x,x)
    s = 3 * x
    if s in existing_sums or s in new_sums:
        return (False, None)
    new_sums.add(s)
    
    # Check (a,x,x) for each a in selected
    for a in selected:
        s = a + 2 * x
        if s in existing_sums or s in new_sums:
            return (False, None)
        new_sums.add(s)
    
    # Check (a,b,x) for a<=b in selected
    for i in range(n):
        sa = selected[i]
        for j in range(i, n):
            sb = selected[j]
            s = sa + sb + x
            if s in existing_sums or s in new_sums:
                return (False, None)
            new_sums.add(s)
    
    combined = existing_sums | new_sums
    return (True, combined)

def _base3_seed(limit, max_size=80):
    """Generate a base-3 inspired seed set that guarantees unique triple sums.
    Use digits 1 and 2 in base-3 positions to create a Sidon-like structure.
    For triples (with repetition), we need all sums of 3 elements to be unique.
    Using powers of 3: a_i = 3^i. Sum of any triple 3^i + 3^j + 3^k is unique
    by base-3 expansion (each digit is 0,1,2,3 but since we pick from {1,2} * 3^i,
    actually let's use a_i = 3^i and check).
    
    Actually for B_3[1], we need that the multiset sum of any 3 elements (with repetition allowed)
    is unique. The standard construction uses a_i = 3^i, but let's verify:
    3^0=1, 3^1=3, 3^2=9, 3^3=27, ...
    Triple sums: 1+1+1=3, 1+1+3=5, 1+3+3=7, 3+3+3=9, 1+1+9=11, etc.
    In base 3, 3^i + 3^j + 3^k where i<=j<=k. The base-3 representation has at most
    one digit > 1 (at position k if i=j=k, giving digit 3 which carries).
    Actually this isn't straightforward. Let's use a_i = 3^i and verify computationally.
    """
    base3 = []
    val = 1
    while val <= limit and len(base3) < max_size:
        base3.append(val)
        val *= 3
    return base3

def _greedy_incremental(start_list, limit, deadline):
    """Greedy construction with incremental sum tracking."""
    base = sorted(start_list)
    if not base:
        base = [1]
    
    # Compute initial sums
    sums = _compute_all_sums(base)
    if sums is None:
        return base
    
    while len(base) < 80:
        if time.time() > deadline:
            break
        next_min = max(base) + 1
        found = False
        for x in range(next_min, limit + 1):
            ok, new_sums = _incremental_check_add(base, x, sums)
            if ok:
                base.append(x)
                sums = new_sums
                found = True
                break
        if not found:
            break
    return base

def _local_search_swap(base, limit, deadline, max_iter=200):
    """Try to improve by swapping elements to allow more additions."""
    base = sorted(base)
    if not base:
        return base
    
    sums = _compute_all_sums(base)
    if sums is None:
        return base
    
    best = list(base)
    best_size = len(base)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try removing one element and adding two
        improved = False
        for remove_idx in range(len(base)):
            if time.time() > deadline:
                break
            candidate = [x for i, x in enumerate(base) if i != remove_idx]
            
            # Recompute sums without removed element
            new_sums = _compute_all_sums(candidate)
            if new_sums is None:
                continue
            
            # Try to add two new elements
            next_min = max(candidate) + 1 if candidate else 1
            added_count = 0
            temp_base = list(candidate)
            temp_sums = new_sums
            
            for x in range(next_min, limit + 1):
                if added_count >= 2:
                    break
                ok, updated = _incremental_check_add(temp_base, x, temp_sums)
                if ok:
                    temp_base.append(x)
                    temp_sums = updated
                    added_count += 1
            
            if len(temp_base) > best_size:
                best = temp_base
                best_size = len(best)
                base = temp_base
                sums = temp_sums
                improved = True
                break
        
        if not improved:
            break
    
    return best

def _dense_packing_greedy(base, limit, deadline):
    """After initial construction, try to pack more elements by inserting between existing ones.
    This is a local search that tries to find gaps where new elements can fit."""
    base = sorted(base)
    if not base:
        return base
    
    sums = _compute_all_sums(base)
    if sums is None:
        return base
    
    best = list(base)
    best_size = len(base)
    
    # Try inserting elements in gaps
    improved = True
    while improved and len(best) < 80 and time.time() < deadline:
        improved = False
        # Generate candidate positions: between each pair, and after max
        candidates = []
        for i in range(len(best) - 1):
            mid = (best[i] + best[i+1]) // 2
            if mid > best[i] and mid < best[i+1]:
                candidates.append(mid)
        candidates.append(max(best) + 1)
        candidates.append(max(best) + 2)
        candidates.append(max(best) + 3)
        
        # Also try positions near existing elements
        for x in best:
            for offset in [1, 2, 3]:
                cand = x + offset
                if cand <= limit and cand not in best:
                    candidates.append(cand)
        
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in best or x < 1 or x > limit:
                continue
            # Check if x can be added
            new_base = sorted(best + [x])
            new_sums = _compute_all_sums(new_base)
            if new_sums is not None:
                best = new_base
                sums = new_sums
                best_size = len(best)
                improved = True
                if len(best) >= 80:
                    break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s margin (90s kill)
    
    limit = 4096
    max_size = 80
    
    best_result = []
    
    # Phase 1: Base-3 inspired seed + greedy
    seed_set = _base3_seed(limit, max_size)
    result = _greedy_incremental(seed_set, limit, deadline)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 2: Deterministic greedy from 1
    result = _greedy_incremental([1], limit, deadline)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 3: Local search swap
    result = _local_search_swap(best_result, limit, deadline, max_iter=50)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 4: Dense packing
    result = _dense_packing_greedy(best_result, limit, deadline)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 5: Try different starting points with random seeds
    for seed in range(2, 20):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        # Try small random starting sets
        k = random.randint(2, 8)
        initial_set = sorted(random.sample(range(1, min(200, limit+1)), k))
        
        if _check_valid(initial_set):
            result = _greedy_incremental(initial_set, limit, deadline)
            if len(result) > len(best_result):
                best_result = result
                # Try local search on the new best
                if time.time() < deadline - 10:
                    result2 = _local_search_swap(best_result, limit, deadline, max_iter=30)
                    if len(result2) > len(best_result):
                        best_result = result2
                if time.time() < deadline - 5:
                    result3 = _dense_packing_greedy(best_result, limit, deadline)
                    if len(result3) > len(best_result):
                        best_result = result3
    
    return best_result

# EVOLVE-BLOCK-END
