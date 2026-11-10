import time
import random
from math import isqrt

P = {'n': 40, 'maximum_size': 18}

# Precompute primes up to 40
def _get_primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

N = P['n']
MAX_SIZE = P['maximum_size']
PRIMES = _get_primes(N)
N_PRIMES = len(PRIMES)
CANDIDATES = list(range(2, N + 1))

# Precompute prime-exponent vectors for all candidates
def _factorize(n):
    fac = {}
    for p in PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            fac[p] = fac.get(p, 0) + 1
            n //= p
    if n > 1:
        fac[n] = fac.get(n, 0) + 1
    return fac

ALL_VECS = {}
for num in CANDIDATES:
    fac = _factorize(num)
    vec = [0] * N_PRIMES
    for i, p in enumerate(PRIMES):
        if p in fac:
            vec[i] = fac[p]
    ALL_VECS[num] = tuple(vec)

ZERO_VEC = (0,) * N_PRIMES

def _check_add(vec, existing_products):
    """
    Check if adding vec to the set of existing products creates a collision.
    Returns (True, new_products) if valid, (False, None) otherwise.
    """
    new_products = set()
    for p in existing_products:
        np = tuple(p[i] + vec[i] for i in range(N_PRIMES))
        if np in existing_products:
            return (False, None)
        new_products.add(np)
    return (True, new_products)

def _greedy_fill(ordered, max_size, deadline, start_set=None, start_products=None):
    """
    Greedily fill a set based on the given order.
    """
    if start_set is not None:
        selected = list(start_set)
        selected_set = set(selected)
        products = start_products.copy() if start_products else {ZERO_VEC}
    else:
        selected = []
        selected_set = set()
        products = {ZERO_VEC}

    for cand in ordered:
        if time.time() > deadline:
            break
        if len(selected) >= max_size:
            break
        if cand in selected_set:
            continue
        
        v = ALL_VECS[cand]
        ok, new_prods = _check_add(v, products)
        if ok:
            selected.append(cand)
            selected_set.add(cand)
            products |= new_prods
            
    return selected, products

def _swap_improve(selected, products, deadline):
    """
    Attempt to improve the current set by removing one element and greedily adding others.
    This helps escape local optima where a "heavy" element blocks multiple "light" ones.
    """
    best_selected = list(selected)
    best_products = products
    best_size = len(selected)
    n = len(selected)
    
    # We want to try removing each element and see if we can fill up more
    for i in range(n):
        if time.time() > deadline:
            break
        if best_size >= MAX_SIZE:
            break
            
        removed = best_selected[i]
        trial = best_selected[:i] + best_selected[i+1:]
        trial_set = set(trial)
        
        # Recompute products for the trial set (without 'removed')
        trial_products = {ZERO_VEC}
        for x in trial:
            v = ALL_VECS[x]
            ok, new_prods = _check_add(v, trial_products)
            if not ok:
                # This shouldn't happen if the original set was valid, but safe to check
                break
            trial_products |= new_prods
            
        # Greedily add from the global order, skipping those already in trial
        # We prioritize adding small/prime numbers first as they are more likely to fit
        order = sorted(CANDIDATES, key=lambda x: (x in PRIMES and 0 or 1, x))
        
        for cand in order:
            if time.time() > deadline:
                break
            if len(trial) >= MAX_SIZE:
                break
            if cand in trial_set:
                continue
            
            v = ALL_VECS[cand]
            ok, new_prods = _check_add(v, trial_products)
            if ok:
                trial.append(cand)
                trial_set.add(cand)
                trial_products |= new_prods
                
        if len(trial) > best_size:
            best_selected = trial
            best_products = trial_products
            best_size = len(trial)
            
    return best_selected, best_products

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Initial deterministic greedy with multiple orderings
    best_set = []
    best_products = {ZERO_VEC}
    best_size = 0
    
    # Ordering 1: Primes first, then composites by value
    order1 = sorted(CANDIDATES, key=lambda x: (x in PRIMES and 0 or 1, x))
    s1, p1 = _greedy_fill(order1, MAX_SIZE, deadline)
    if len(s1) > best_size:
        best_set, best_products, best_size = s1, p1, len(s1)
        
    # Ordering 2: Ascending value
    order2 = sorted(CANDIDATES)
    s2, p2 = _greedy_fill(order2, MAX_SIZE, deadline)
    if len(s2) > best_size:
        best_set, best_products, best_size = s2, p2, len(s2)

    # Ordering 3: Descending value (sometimes helps? unlikely but cheap)
    order3 = sorted(CANDIDATES, reverse=True)
    s3, p3 = _greedy_fill(order3, MAX_SIZE, deadline)
    if len(s3) > best_size:
        best_set, best_products, best_size = s3, p3, len(s3)
        
    # Local search improvement on the best found so far
    if time.time() < deadline - 5:
        improved_set, improved_products = _swap_improve(best_set, best_products, deadline)
        if len(improved_set) > best_size:
            best_set, best_products, best_size = improved_set, improved_products, len(improved_set)
            
    # Random restarts if time permits
    # We use the improved set as a seed for some restarts, or pure random
    attempts = 0
    while time.time() < deadline - 2:
        attempts += 1
        # Create a random order
        random_order = CANDIDATES[:]
        random.shuffle(random_order)
        
        # Occasionally bias towards primes
        if attempts % 3 == 0:
            primes_list = [x for x in random_order if x in PRIMES]
            composites_list = [x for x in random_order if x not in PRIMES]
            random_order = primes_list + composites_list
            
        s, p = _greedy_fill(random_order, MAX_SIZE, deadline)
        
        # Quick local search on this random result
        if len(s) > best_size:
            s, p = _swap_improve(s, p, deadline)
            
        if len(s) > best_size:
            best_set, best_products, best_size = s, p, len(s)
            if best_size >= MAX_SIZE:
                break
                
    # Final verification to ensure validity (crucial for scoring)
    # We must ensure all subset products are distinct.
    # Since we built it via incremental check, it should be valid, but let's double check
    # if the set is small enough or if we want to be 100% sure.
    # For k=18, 2^18 = 262144 products. Checking this is fast.
    if best_set:
        products = {ZERO_VEC}
        valid = True
        for x in best_set:
            v = ALL_VECS[x]
            ok, new_prods = _check_add(v, products)
            if not ok:
                valid = False
                break
            products |= new_prods
        
        if not valid:
            # Fallback to a known valid small set (e.g., just primes)
            # This should never happen if _check_add is correct, but safety first.
            fallback = [p for p in PRIMES if p <= N]
            return fallback
            
    # Ensure return type is list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
