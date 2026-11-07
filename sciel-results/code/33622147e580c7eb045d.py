"""Combined program: fast incremental check + deterministic greedy + many randomized restarts."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

P = {'n': 40, 'maximum_size': 18}

def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0 or a % 3 == 0:
        return False
    d = 5
    while d * d <= a:
        if a % d == 0 or a % (d + 2) == 0:
            return False
        d += 6
    return True

def _largest_prime_factor(n):
    lpf = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            lpf = d
            n //= d
        d += 1
    if n > 1:
        lpf = n
    return lpf

def _check_valid(S):
    """
    Validate that all subset products of S are distinct.
    Uses the incremental set expansion method.
    """
    current = {1}
    for x in S:
        new_vals = {p * x for p in current}
        if current & new_vals:
            return False
        current |= new_vals
    return True

def _try_greedy(order):
    """
    Greedily add elements from order to build a valid set.
    Returns the list of elements added.
    """
    current_set = []
    prods = {1}
    for v in order:
        new_prods = []
        valid = True
        for p in prods:
            np = v * p
            if np in prods:
                valid = False
                break
            new_prods.append(np)
        if valid:
            current_set.append(v)
            prods.update(new_prods)
    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    candidates = list(range(2, P['n'] + 1))
    primes = [a for a in candidates if _is_prime(a)]
    composites = [a for a in candidates if not _is_prime(a)]
    
    # Initial best: primes
    best_set = primes[:]
    best_size = len(best_set)
    
    # 1. Deterministic Greedy Attempts
    # Order 1: Primes + Composites sorted by largest prime factor (descending)
    composites_lpf_desc = sorted(composites, key=_largest_prime_factor, reverse=True)
    order1 = primes + composites_lpf_desc
    sol1 = _try_greedy(order1)
    if len(sol1) > best_size:
        best_size = len(sol1)
        best_set = sol1[:]
        
    # Order 2: Primes + Composites sorted by value (ascending)
    order2 = primes + sorted(composites)
    sol2 = _try_greedy(order2)
    if len(sol2) > best_size:
        best_size = len(sol2)
        best_set = sol2[:]
        
    # Order 3: Primes + Composites sorted by value (descending)
    order3 = primes + sorted(composites, reverse=True)
    sol3 = _try_greedy(order3)
    if len(sol3) > best_size:
        best_size = len(sol3)
        best_set = sol3[:]

    # 2. Randomized Restart Greedy
    # We use a fixed seed for reproducibility, but the task allows randomness.
    # We will iterate many times.
    rng = random.Random(42)
    
    # Pre-compute a few static orders to reuse? No, shuffling is cheap.
    # We'll shuffle the entire candidate list.
    # To bias towards "good" elements, we can sort by some heuristic first, then shuffle?
    # Or just pure shuffle. Pure shuffle is robust.
    # Let's try a mix: sometimes shuffle all, sometimes shuffle composites and keep primes first.
    
    # Strategy A: Shuffle all candidates
    # Strategy B: Primes first (in random order?), then shuffle composites
    
    # Let's stick to Strategy A for simplicity and coverage, but maybe weight it.
    # Actually, keeping primes first is often good because they are "independent".
    # Let's do 50% Strategy A, 50% Strategy B.
    
    while time.time() < deadline:
        if rng.random() < 0.5:
            order = candidates[:]
            rng.shuffle(order)
        else:
            p_shuffled = primes[:]
            rng.shuffle(p_shuffled)
            c_shuffled = composites[:]
            rng.shuffle(c_shuffled)
            order = p_shuffled + c_shuffled
            
        sol = _try_greedy(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
            
            # If we hit a high number, we can stop early?
            # Max possible is limited by 2^k products distinct.
            # With 40 candidates, max k is small. 18 is the upper bound in task description.
            # If best_size >= 16, it's very good.
            # But let's keep searching until deadline to be safe.
            
    # Final validation
    if not _check_valid(best_set):
        # Fallback to primes if something went wrong (shouldn't happen)
        best_set = primes[:]
        
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
