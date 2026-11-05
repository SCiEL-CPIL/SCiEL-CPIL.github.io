"""Construct the largest admissible tuple in [0, 5000]."""
import time
import math
from collections import defaultdict

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to a reasonable bound (max tuple size <= N+1 = 5001)
_PRIMES_CACHE = {}

def _get_primes(limit):
    if limit not in _PRIMES_CACHE:
        _PRIMES_CACHE[limit] = _primes(limit)
    return _PRIMES_CACHE[limit]

def _is_admissible(points, primes_list):
    """Check if points form an admissible tuple for the given primes."""
    n = len(points)
    for p in primes_list:
        if p > n:
            break
        # Check if all p residue classes are covered
        covered = set()
        for v in points:
            covered.add(v % p)
            if len(covered) == p:
                return False
    return True

def _repair(points, primes_list):
    """Repair points to make them admissible by removing elements."""
    points = list(points)
    changed = True
    while changed:
        changed = False
        n = len(points)
        for p in primes_list:
            if p > n:
                break
            # Count elements in each residue class
            classes = defaultdict(list)
            for v in points:
                classes[v % p].append(v)
            if len(classes) == p:
                # All classes covered; remove the smallest class
                smallest = min(classes.values(), key=len)
                remove_set = set(smallest)
                points = [v for v in points if v not in remove_set]
                changed = True
                break
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    # Strategy: Try to build the largest admissible set.
    # The maximum possible size is N+1 = 5001, but admissibility constraints limit it.
    # 
    # Key insight: For a set of size k, we need primes p <= k to each miss a class.
    # The constraint is most binding for small primes.
    #
    # Approach: Start with all numbers [0, N] and repair, but that gives a specific answer.
    # Instead, try multiple strategies and keep the best.
    
    best_result = []
    best_size = 0
    
    # Strategy 1: Start with all numbers and repair
    all_points = list(range(N + 1))
    primes_for_repair = _get_primes(N + 1)
    repaired = _repair(all_points, primes_for_repair)
    if len(repaired) > best_size:
        best_size = len(repaired)
        best_result = repaired
    
    # Strategy 2: Start with primes > sqrt(N) plus 1 (like seed)
    sqrt_N = math.isqrt(N)
    large_primes = [p for p in _get_primes(N) if p > sqrt_N]
    points2 = [1] + large_primes
    repaired2 = _repair(points2, _get_primes(len(points2)))
    if len(repaired2) > best_size:
        best_size = len(repaired2)
        best_result = repaired2
    
    # Strategy 3: Try to maximize by starting with a large set and being smarter about removals
    # The key is that for each prime p <= k, we need to miss at least one class.
    # To maximize k, we want to remove as few elements as possible.
    # 
    # Greedy approach: For each prime p from smallest to largest (up to potential k),
    # ensure we miss a class. The class we miss should be the one with fewest elements.
    
    # Let's try: start with all numbers, and for each prime p <= current_size,
    # if all classes are covered, remove the least-populated class.
    # But do this in order of increasing p to minimize total removals.
    
    # Actually, the repair function already does this iteratively. Let me try a 
    # different initial set that might lead to a larger admissible set.
    
    # Strategy 4: Use a heuristic - numbers that are 1 mod small primes
    # or numbers in specific residue classes to naturally miss classes.
    
    # Try: all numbers that are NOT 0 mod 2 (i.e., all odd numbers)
    # This automatically misses class 0 mod 2.
    odd_points = [x for x in range(1, N + 1, 2)]
    repaired4 = _repair(odd_points, _get_primes(len(odd_points)))
    if len(repaired4) > best_size:
        best_size = len(repaired4)
        best_result = repaired4
    
    # Try: all numbers that are 1 mod 2 (same as above)
    
    # Try: numbers that avoid 0 mod 2 and 0 mod 3
    # i.e., x % 6 in {1, 5}
    points5 = [x for x in range(1, N + 1) if x % 6 in (1, 5)]
    repaired5 = _repair(points5, _get_primes(len(points5)))
    if len(repaired5) > best_size:
        best_size = len(repaired5)
        best_result = repaired5
    
    # Try: numbers that avoid 0 mod 2, 0 mod 3, 0 mod 5
    # x % 30 in {1, 7, 11, 13, 17, 19, 23, 29}
    residues_30 = [1, 7, 11, 13, 17, 19, 23, 29]
    points6 = [x for x in range(1, N + 1) if x % 30 in residues_30]
    repaired6 = _repair(points6, _get_primes(len(points6)))
    if len(repaired6) > best_size:
        best_size = len(repaired6)
        best_result = repaired6
    
    # Try: avoid 0 mod 2, 0 mod 3, 0 mod 5, 0 mod 7
    # x % 210 in residues coprime to 210
    residues_210 = [x for x in range(1, 210) if math.gcd(x, 210) == 1]
    points7 = [x for x in range(1, N + 1) if x % 210 in residues_210]
    repaired7 = _repair(points7, _get_primes(len(points7)))
    if len(repaired7) > best_size:
        best_size = len(repaired7)
        best_result = repaired7
    
    # Try: avoid 0 mod 2, 0 mod 3, 0 mod 5, 0 mod 7, 0 mod 11
    # x % 2310 in residues coprime to 2310
    residues_2310 = [x for x in range(1, 2310) if math.gcd(x, 2310) == 1]
    points8 = [x for x in range(1, N + 1) if x % 2310 in residues_2310]
    repaired8 = _repair(points8, _get_primes(len(points8)))
    if len(repaired8) > best_size:
        best_size = len(repaired8)
        best_result = repaired8
    
    # Check time
    if time.time() > deadline:
        return best_result
    
    # Strategy 5: More systematic - for each "avoid set" of small primes,
    # build the set of numbers avoiding 0 mod p for p in that set, then repair.
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Try avoiding 0 mod p for various subsets of small primes
    # Use the product of the avoided primes as the modulus
    
    # Avoid 0 mod 2,3,5,7,11,13 (product = 30030)
    # But 30030 > 5000, so just check gcd with the product
    prod = 1
    for p in [2, 3, 5, 7, 11, 13]:
        prod *= p
    # prod = 30030
    points9 = [x for x in range(1, N + 1) if math.gcd(x, prod) == 1]
    repaired9 = _repair(points9, _get_primes(len(points9)))
    if len(repaired9) > best_size:
        best_size = len(repaired9)
        best_result = repaired9
    
    if time.time() > deadline:
        return best_result
    
    # Try avoiding 0 mod 2,3,5,7,11,13,17 (product = 510510)
    prod2 = 1
    for p in [2, 3, 5, 7, 11, 13, 17]:
        prod2 *= p
    points10 = [x for x in range(1, N + 1) if math.gcd(x, prod2) == 1]
    repaired10 = _repair(points10, _get_primes(len(points10)))
    if len(repaired10) > best_size:
        best_size = len(repaired10)
        best_result = repaired10
    
    if time.time() > deadline:
        return best_result
    
    # Now try to improve: for the best result, see if we can add back elements
    # that were removed, as long as admissibility is maintained.
    
    # Actually, let's try a completely different approach:
    # Binary search on the size k. For a given k, can we find an admissible set of size k?
    # This is hard, but we can try: start with all numbers, repair, and see what size we get.
    # The repair is deterministic given the initial set, so different initial sets give different results.
    
    # Let me try: start with a random-ish set and repair, keeping the best.
    import random
    random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Create a random subset of [0, N]
        # Size around what we expect
        target_size = best_size + random.randint(-100, 500)
        if target_size < 10:
            target_size = 10
        if target_size > N + 1:
            target_size = N + 1
        
        # Sample target_size elements
        sample = random.sample(range(N + 1), min(target_size, N + 1))
        repaired_r = _repair(sample, _get_primes(len(sample)))
        if len(repaired_r) > best_size:
            best_size = len(repaired_r)
            best_result = repaired_r
    
    # Final: try to add elements to the best result
    # Check if we can add any element from [0, N] not in best_result while maintaining admissibility
    if time.time() < deadline - 5:
        best_set = set(best_result)
        current_list = list(best_result)
        primes_needed = _get_primes(len(current_list))
        
        # Try adding each missing element
        for candidate in range(N + 1):
            if time.time() > deadline - 5:
                break
            if candidate in best_set:
                continue
            # Try adding candidate
            test_list = current_list + [candidate]
            test_primes = _get_primes(len(test_list))
            if _is_admissible(test_list, test_primes):
                current_list = test_list
                best_set.add(candidate)
                primes_needed = test_primes
    
    return current_list if 'current_list' in dir() and len(current_list) >= best_size else best_result
