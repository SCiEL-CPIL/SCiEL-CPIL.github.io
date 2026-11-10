import time
import random

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _generate_smooth(primes, max_val=10000):
    """Generate smooth numbers with given prime bases."""
    result = set()
    def rec(idx, current):
        if current > max_val:
            return
        result.add(current)
        if idx < len(primes):
            val = current
            while val * primes[idx] <= max_val:
                val *= primes[idx]
                rec(idx + 1, val)
    rec(0, 1)
    return sorted(result)

def _build_candidate_pool(rng, A):
    """Build a rich candidate pool for local search replacements."""
    pool = set()
    # Small integers 1-120
    pool.update(range(1, 121))
    # 2,3-smooth numbers
    for i in range(1, 15):
        for j in range(1, 15):
            val = (2 ** i) * (3 ** j)
            if val <= 10000:
                pool.add(val)
    # 2,3,5-smooth numbers
    for i in range(1, 14):
        for j in range(1, 14):
            for k in range(1, 14):
                val = (2 ** i) * (3 ** j) * (5 ** k)
                if val <= 10000:
                    pool.add(val)
    # Powers of 2, 3, 5
    for base in [2, 3, 5, 7]:
        val = base
        while val <= 10000:
            pool.add(val)
            val *= base
    # Collision-targeting: for each pair in A, try values that would create
    # sum/product collisions with existing pairs
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s = A[i] + A[j]
            p = A[i] * A[j]
            # Try values that could create collisions in sums
            for k in range(n):
                if k != i and k != j:
                    candidate = s - A[k]
                    if 1 <= candidate <= 10000 and candidate not in A:
                        pool.add(candidate)
            # Try values that could create collisions in products
            for k in range(n):
                if k != i and k != j and A[k] != 1:
                    if p % A[k] == 0:
                        candidate = p // A[k]
                        if 1 <= candidate <= 10000 and candidate not in A:
                            pool.add(candidate)
    # Remove elements already in A
    pool.difference_update(set(A))
    pool_list = list(pool)
    rng.shuffle(pool_list)
    return pool_list

def _incremental_score(A, i_old, old_val, new_val):
    """
    Fast score calculation when replacing A[i_old] with new_val.
    Reuses pairs that don't involve index i_old.
    """
    n = len(A)
    sums = set()
    prods = set()
    
    # Add all pairs not involving i_old
    for i in range(n):
        if i == i_old:
            continue
        for j in range(i + 1, n):
            if j == i_old:
                continue
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    
    # Add pairs involving i_old with new_val
    for j in range(n):
        if j == i_old:
            continue
        sums.add(new_val + A[j])
        prods.add(new_val * A[j])
    
    # Add the diagonal pair (new_val, new_val)
    sums.add(2 * new_val)
    prods.add(new_val * new_val)
    
    return max(len(sums), len(prods))

def local_search(A, time_limit, rng, candidate_pool=None):
    """
    Perform local search to minimize the score.
    Tries to replace elements in A with other integers to improve the score.
    """
    A = list(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    if candidate_pool is None:
        candidate_pool = _build_candidate_pool(rng, A)
    
    # Ensure A is sorted for consistent indexing in local search
    A = sorted(A)
    best_A = sorted(best_A)
    
    # Precompute set for O(1) membership
    A_set = set(A)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try swapping each element in A with a candidate from the pool
        for i in range(len(A)):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            for new_val in candidate_pool:
                if new_val == old_val or new_val in A_set:
                    continue
                
                # Fast incremental score
                score = _incremental_score(A, i, old_val, new_val)
                
                if score < current_score:
                    # Update A and A_set
                    A_set.discard(old_val)
                    A[i] = new_val
                    A_set.add(new_val)
                    A = sorted(A)
                    current_score = score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    break
        
        # Polish pass: try swapping pairs within A
        if (time.time() - start_time) < time_limit:
            for i in range(len(A)):
                if (time.time() - start_time) > time_limit:
                    return best_A, best_score
                for j in range(i + 1, len(A)):
                    if (time.time() - start_time) > time_limit:
                        return best_A, best_score
                    # Try replacing A[i] with A[j] and A[j] with A[i] is just a swap, no benefit
                    # Instead, try replacing A[i] with some other value from pool
                    # This is already covered above. Skip.
                    pass
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    rng = random.Random(42)

    candidates = []

    # Candidate 1: Arithmetic Progression 1..20
    candidates.append(list(range(1, 21)))

    # Candidate 2: Geometric Progression 1, 2, 4, ...
    gp = [2**i for i in range(14)]  # 16 elements
    for x in range(1, 100):
        if x not in gp and len(gp) < 20:
            gp.append(x)
    candidates.append(gp)

    # Candidate 3: 2,3-smooth numbers
    smooth = []
    for a in range(14):
        for b in range(14):
            val = (2**a) * (3**b)
            if val <= 10000:
                smooth.append(val)
    smooth = sorted(set(smooth))
    candidates.append(smooth[:20])

    # Candidate 4: Random set of small integers
    rand_set = random.sample(range(1, 101), 20)
    candidates.append(rand_set)

    # Candidate 5: Mixed structure - low AP + high sparse
    mixed = list(range(1, 11)) + [12, 14, 16, 20, 25, 30, 40, 50, 60, 80]
    candidates.append(mixed)

    # Candidate 6: 2,3,5-smooth numbers
    smooth_235 = _generate_smooth([2, 3, 5])
    candidates.append(smooth_235[:20])

    # Candidate 7: 2,3,5,7-smooth numbers
    smooth_2357 = _generate_smooth([2, 3, 5, 7])
    candidates.append(smooth_2357[:20])

    # Candidate 8: Geometric progression with ratio ~1.3
    geo = []
    val = 1
    for i in range(20):
        val = int(round(val * 1.3))
        if val > 10000:
            break
        if val not in geo:
            geo.append(val)
    if len(geo) >= 20:
        candidates.append(geo[:20])

    # Candidate 9: Geometric progression with ratio ~1.5
    geo2 = []
    val = 1
    for i in range(20):
        val = int(round(val * 1.5))
        if val > 10000:
            break
        if val not in geo2:
            geo2.append(val)
    if len(geo2) >= 20:
        candidates.append(geo2[:20])

    # Candidate 10: Mix: small numbers + powers of 2
    mix = list(range(1, 11)) + [2**i for i in range(4, 14)]
    mix = sorted(set(x for x in mix if x <= 10000))
    if len(mix) >= 20:
        candidates.append(mix[:20])

    # Candidate 11: Random sample from smooth pool
    if len(smooth_235) >= 20:
        candidates.append(sorted(random.sample(smooth_235, 20)))

    # Candidate 12: 2,3,5,7-smooth, pick 20 with good spread
    if len(smooth_2357) >= 20:
        # Pick evenly spaced from the smooth list
        step = len(smooth_2357) // 20
        spread = [smooth_2357[i * step] for i in range(20)]
        if len(set(spread)) == 20:
            candidates.append(sorted(spread))

    best_A = None
    best_score = float('inf')

    # Sort candidates by initial score to prioritize good starting points
    scored_candidates = []
    for A in candidates:
        if len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
        s = calc_score(A)
        scored_candidates.append((s, A))
    scored_candidates.sort(key=lambda x: x[0])

    # Run local search on each candidate with allocated time
    n_candidates = len(scored_candidates)
    for idx, (init_score, A) in enumerate(scored_candidates):
        if time.time() > deadline:
            break

        time_remaining = deadline - time.time()
        if time_remaining < 2:
            break

        # Allocate more time to better initial candidates
        if idx == 0:
            alloc_time = min(25.0, time_remaining * 0.5)
        elif idx == 1:
            alloc_time = min(15.0, time_remaining * 0.3)
        else:
            alloc_time = min(8.0, time_remaining * 0.2)

        improved_A, score = local_search(A, alloc_time, rng)

        if score < best_score:
            best_score = score
            best_A = improved_A

        # If we've already found a very good score, spend less time on later candidates
        if best_score <= 55 and idx >= 2:
            alloc_time = min(3.0, time_remaining * 0.15)

    if best_A is None:
        best_A = list(range(1, 21))

    return best_A
