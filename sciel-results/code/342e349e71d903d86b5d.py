"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _parity_vector(num, primes):
    """Return tuple of parities (exponent mod 2) for each prime in primes."""
    v = []
    n = num
    for p in primes:
        c = 0
        while n % p == 0:
            n //= p
            c += 1
        v.append(c % 2)
        if n == 1:
            v.extend([0] * (len(primes) - len(v)))
            break
    else:
        # If n > 1 after processing all primes, there's a large prime factor
        # This means our prime list wasn't sufficient, but for our range it should be fine
        pass
    return tuple(v)

def _xor(a, b):
    """XOR two parity vectors."""
    return tuple((x ^ y) for x, y in zip(a, b))

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin out of 60s
    
    n = 210  # base number
    min_val = n + 1  # must be > 210
    max_val = 10210  # at most 10210
    max_size = 100
    
    # First, let's factorize n to get its parity vector
    # n = 210 = 2 * 3 * 5 * 7
    # So the target parity vector is (1,1,1,1,0,0,...) for primes 2,3,5,7
    
    # We need to collect enough primes. Let's use primes up to max_val
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, max_val + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, max_val + 1) if sieve[i]]
    
    # Get parity vector of n
    target = _parity_vector(n, primes)
    
    # Strategy: Binary search on M (max selected value)
    # For a given M, check if there exists a subset of {n+1, ..., M} 
    # whose XOR of parity vectors equals target, with size <= max_size
    # 
    # Actually, we can use Gaussian elimination to find a subset.
    # The set of all possible XOR sums of subsets of {v(n+1), ..., v(M)} 
    # forms a subspace. We need target to be in this subspace.
    # Moreover, we need a small subset (ideally small for minimality).
    #
    # Let's do a BFS/BFS-like approach for small M, and Gaussian elimination for feasibility.
    
    # First, compute parity vectors for all numbers from min_val to max_val
    parity_map = {}
    for x in range(min_val, max_val + 1):
        if time.time() > deadline:
            break
        parity_map[x] = _parity_vector(x, primes)
    
    # Check if we ran out of time
    if time.time() > deadline:
        return [4 * n]
    
    # Try to find the minimal M by searching
    # Start from small M and increase
    
    best_answer = [4 * n]  # fallback
    best_raw = 4 * n - n  # 630
    
    # We'll search M from min_val upward
    # For efficiency, use a frontier-based BFS in the XOR space
    
    # Let's collect vectors in order
    numbers = list(range(min_val, max_val + 1))
    
    # For each M, we have vectors v(n+1), ..., v(M)
    # We want to know if target is in the span, and find a small subset
    
    # Use incremental Gaussian elimination
    # As we add vectors, we maintain a basis and track which original vectors contribute
    
    # Actually, let's try a different approach: 
    # For small M, do BFS over subsets (up to some size)
    # For larger M, use the fact that the span grows quickly
    
    # Let's try: for M from min_val to max_val, maintain a set of reachable XOR values
    # with the minimum number of elements to reach each value, and track the actual subset
    
    # But the XOR space can be large. Let's cap the BFS depth.
    
    # Alternative: Use the following insight.
    # The parity vectors are in GF(2)^k where k = len(primes).
    # But most vectors will have small support (few primes).
    
    # Let's try a simple approach first:
    # 1. Check if any single number has parity == target
    # 2. Check if any pair has XOR == target  
    # 3. Check if any triple has XOR == target
    # etc.
    
    # Build lookup
    vec_to_num = {}
    for x in numbers:
        if x not in parity_map:
            continue
        v = parity_map[x]
        if v not in vec_to_num:
            vec_to_num[v] = []
        vec_to_num[v].append(x)
    
    # Single element
    if target in vec_to_num:
        selected = [vec_to_num[target][0]]
        # Verify
        prod = n
        for x in selected:
            prod *= x
        if math.isqrt(prod) ** 2 == prod:
            return selected
    
    # Pair: v1 ^ v2 == target => v2 = v1 ^ target
    for x in numbers:
        if x not in parity_map:
            continue
        if time.time() > deadline:
            break
        v1 = parity_map[x]
        v2_needed = _xor(v1, target)
        if v2_needed in vec_to_num:
            for y in vec_to_num[v2_needed]:
                if y != x:
                    selected = [x, y]
                    prod = n
                    for z in selected:
                        prod *= z
                    if math.isqrt(prod) ** 2 == prod:
                        raw = max(selected) - n
                        if raw < best_raw:
                            best_raw = raw
                            best_answer = selected
                        return selected
    
    # Triple: v1 ^ v2 ^ v3 == target
    # This is O(n^3) which is too slow. Let's use a smarter approach.
    # For each pair (x,y), compute v1^v2, and look for v3 = target ^ v1 ^ v2
    # But that's still O(n^2) pairs times lookup.
    
    # Let's precompute all pair XORs
    pair_xor = defaultdict(list)  # xor_val -> list of (x, y)
    
    # Actually, let's be smarter. We only need to check pairs where both x,y <= M
    # for the current M. Let's iterate M and maintain state.
    
    # Let's do incremental: for M from min_val upward
    # Maintain:
    # - set of single vectors seen
    # - set of pair XORs seen (with the pair)
    # When adding a new number z with vector vz:
    #   Check if vz == target (single)
    #   Check if there's x seen with vx ^ vz == target (pair)
    #   Check if there's pair (x,y) seen with vx^vy^vz == target (triple)
    
    seen_singles = {}  # vector -> number
    seen_pairs = defaultdict(list)  # xor -> [(x, y)]
    
    for M in range(min_val, max_val + 1):
        if time.time() > deadline:
            break
        
        if M not in parity_map:
            continue
        
        vz = parity_map[M]
        
        # Check single
        if vz == target:
            selected = [M]
            prod = n * M
            if math.isqrt(prod) ** 2 == prod:
                return selected
        
        # Check pair with previous singles
        needed = _xor(vz, target)
        if needed in seen_singles:
            x = seen_singles[needed]
            selected = [x, M]
            prod = n * x * M
            if math.isqrt(prod) ** 2 == prod:
                return selected
        
        # Check triple with previous pairs
        needed_pair_xor = _xor(vz, target)
        if needed_pair_xor in seen_pairs:
            for (x, y) in seen_pairs[needed_pair_xor]:
                selected = [x, y, M]
                prod = n * x * y * M
                if math.isqrt(prod) ** 2 == prod:
                    return selected
        
        # Add M to singles
        if vz not in seen_singles:
            seen_singles[vz] = M
        
        # Add pairs involving M
        for x, vx in seen_singles.items():
            if x == M:
                continue
            pair_x = _xor(vx, vz)
            seen_pairs[pair_x].append((x, M))
        
        # If we've found a good answer, we can stop early if we want
        # But we want minimal M, so continue until deadline or max_val
    
    # If we didn't find anything with 1-3 elements, try a different approach
    # Use Gaussian elimination to find any subset
    
    if time.time() > deadline:
        return best_answer
    
    # Build matrix from all numbers min_val to max_val
    # Use Gaussian elimination over GF(2)
    
    # Actually, let's try quadruples too, but more efficiently
    # Or use a randomized approach
    
    # Let's try: pick a random subset and check
    import random
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        size = random.randint(4, 10)
        subset = random.sample(numbers, size)
        xor_acc = (0,) * len(primes)
        for x in subset:
            xor_acc = _xor(xor_acc, parity_map[x])
        if xor_acc == target:
            selected = subset
            prod = n
            for x in subset:
                prod *= x
            if math.isqrt(prod) ** 2 == prod:
                return selected
    
    return best_answer

# EVOLVE-BLOCK-END
