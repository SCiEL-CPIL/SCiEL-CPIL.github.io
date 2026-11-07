"""Rewritten program: stronger algorithm for large sets with distinct subset products."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 40, 'maximum_size': 18}

# Precompute factorizations for numbers 2..40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES)

def _factorize(n):
    """Return exponent vector as tuple for n (2..40)."""
    exponents = [0] * _NUM_PRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            exponents[i] += 1
            n //= p
    return tuple(exponents)

# Precompute factorizations for all numbers 2..40
_FACTORS = {i: _factorize(i) for i in range(2, 41)}

def _product_distinct(nums):
    """Check if all 2^k subset products are distinct.
    Uses prime exponent vectors. Two subset products are equal iff
    they have the same prime exponent vector.
    """
    k = len(nums)
    if k == 0:
        return True
    
    # Pre-compute factorization of each number in the set
    factors = [_FACTORS[x] for x in nums]
    
    # For k up to 18, there are 2^18 = 262144 subsets
    # Use DP: for each mask, exponent vector = sum of factors of set bits
    # We'll track the exponent vector as a tuple for hashing
    
    num_subsets = 1 << k
    seen = set()
    seen.add((0,) * _NUM_PRIMES)  # empty subset
    
    # Compute exponent vectors using DP
    # exp_vector[mask] = exp_vector[mask without lowest set bit] + factors[lsb_index]
    # To save memory, we can compute on the fly
    
    # For efficiency, use the fact that:
    # mask & (mask-1) removes the lowest set bit
    # So: exp_vector[mask] = exp_vector[mask & (mask-1)] + factors[lsb_index]
    
    # Store only the current vectors we need
    # Actually, for k=18, 2^18 = 262144 vectors, each of length 12
    # That's manageable in Python
    
    # Let's use a list to store vectors
    exp_vectors = [None] * num_subsets
    exp_vectors[0] = (0,) * _NUM_PRIMES
    
    for mask in range(1, num_subsets):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        prev_vec = exp_vectors[prev_mask]
        f = factors[idx]
        
        # Add the two vectors
        new_vec = tuple(prev_vec[i] + f[i] for i in range(_NUM_PRIMES))
        exp_vectors[mask] = new_vec
        
        if new_vec in seen:
            return False
        seen.add(new_vec)
    
    return True

def _try_greedy(order, max_size=None):
    """Greedy: add numbers in the given order if they keep the set valid."""
    if max_size is None:
        max_size = P['maximum_size']
    
    current = []
    for candidate in order:
        if len(current) >= max_size:
            break
        new_set = current + [candidate]
        if _product_distinct(new_set):
            current = new_set
    return current

def _local_search(nums, time_limit=5.0):
    """Perform local search to improve the set."""
    current = list(nums)
    if not _product_distinct(current):
        return None
    
    best = current
    best_size = len(current)
    start_time = time.time()
    
    while time.time() - start_time < time_limit and best_size < P['maximum_size']:
        improved = False
        
        # Try adding a new element
        for candidate in range(2, P['n'] + 1):
            if candidate not in best:
                new_set = best + [candidate]
                if _product_distinct(new_set):
                    best = new_set
                    best_size = len(best)
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try single-swap: remove one, add another
        for i in range(len(best)):
            for candidate in range(2, P['n'] + 1):
                if candidate == best[i] or candidate in best:
                    continue
                new_set = [x for j, x in enumerate(best) if j != i]
                new_set.append(candidate)
                if _product_distinct(new_set):
                    best = new_set
                    best_size = len(best)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try double-swap: remove two, add two
            if time.time() - start_time < time_limit * 0.7:
                found = False
                for i in range(len(best)):
                    for j in range(i + 1, len(best)):
                        for c1 in range(2, P['n'] + 1):
                            if c1 in best:
                                continue
                            for c2 in range(c1 + 1, P['n'] + 1):
                                if c2 in best or c2 == c1:
                                    continue
                                new_set = [x for k, x in enumerate(best) if k != i and k != j]
                                new_set.extend([c1, c2])
                                if _product_distinct(new_set):
                                    best = new_set
                                    best_size = len(best)
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    continue
        
        if not improved:
            break
    
    return best

def construct():
    """Construct a set of distinct integers in 2..40 with distinct subset products.
    Maximize the number of selected integers."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_set = None
    best_size = 0
    
    # Strategy 1: Deterministic greedy with different orderings
    # Try ascending order
    greedy1 = _try_greedy(range(2, 41))
    if len(greedy1) > best_size:
        best_size = len(greedy1)
        best_set = greedy1[:]
    
    # Try descending order
    greedy2 = _try_greedy(range(40, 1, -1))
    if len(greedy2) > best_size:
        best_size = len(greedy2)
        best_set = greedy2[:]
    
    # Try prime-priority order
    primes_order = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 9, 25, 8, 27, 16, 12, 15, 21, 22, 26, 33, 34, 35, 38, 39, 6, 10, 14, 18, 20, 24, 28, 30, 32, 36]
    greedy3 = _try_greedy(primes_order)
    if len(greedy3) > best_size:
        best_size = len(greedy3)
        best_set = greedy3[:]
    
    # Strategy 2: Randomized greedy with multiple trials
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
        order = list(range(2, 41))
        random.shuffle(order)
        result = _try_greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Strategy 3: Local search from multiple starting points
    starting_points = []
    
    # Add the best greedy result
    if best_set:
        starting_points.append(best_set[:])
    
    # Add some random starting points
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
        size = random.randint(8, min(15, P['n'] - 1))
        candidate = random.sample(range(2, 41), size)
        if _product_distinct(candidate):
            starting_points.append(candidate)
    
    # Run local search from each starting point
    for i, start in enumerate(starting_points):
        if time.time() > deadline - 5.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Give each local search a portion of the time
        per_search_time = max(0.5, min(remaining / max(1, len(starting_points) - i), 5.0))
        
        result = _local_search(start, time_limit=per_search_time)
        if result and len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Strategy 4: More random restarts if time remains
    if best_size < P['maximum_size']:
        attempts = 0
        while time.time() < deadline - 2.0 and attempts < 30:
            attempts += 1
            size = random.randint(10, min(16, P['n'] - 1))
            candidate = random.sample(range(2, 41), size)
            if not _product_distinct(candidate):
                continue
            
            improved = _local_search(candidate, time_limit=1.0)
            if improved and len(improved) > best_size:
                best_size = len(improved)
                best_set = improved[:]
            
            if best_size >= P['maximum_size']:
                break
    
    # Final: ensure we have a valid result
    if best_set is None:
        best_set = [2]
        best_size = 1
    elif not _product_distinct(best_set):
        best_set = [2]
        best_size = 1
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
