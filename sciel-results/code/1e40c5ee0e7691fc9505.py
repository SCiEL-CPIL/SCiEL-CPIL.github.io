"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict
import random

def _is_powerful(n):
    """Check if n is powerful: for every prime p|n, p^2|n."""
    if n <= 1:
        return True
    temp = n
    for p in range(2, isqrt(temp) + 1):
        if temp % p == 0:
            # Check p^2 divides original n
            if n % (p * p) != 0:
                return False
            while temp % p == 0:
                temp //= p
    if temp > 1:
        # temp is a prime factor with exponent 1
        if n % (temp * temp) != 0:
            return False
    return True

def _generate_small_powerful(limit):
    """Generate powerful numbers up to limit using direct check for small limits."""
    result = [1]
    for n in range(2, limit + 1):
        if _is_powerful(n):
            result.append(n)
    return result

def _factorize(n):
    """Factorize n, returning dict of prime->exponent. For n up to 10^18."""
    factors = {}
    temp = n
    for p in range(2, isqrt(temp) + 1):
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            factors[p] = exp
    if temp > 1:
        factors[temp] = 1
    return factors

def _verify_powerful(n):
    """Verify n is powerful using exact factorization."""
    if n <= 1:
        return True
    factors = _factorize(n)
    for p, e in factors.items():
        if e < 2:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate powerful numbers up to a moderate limit for use as 'a' values
    # We need powerful a such that a <= N/2 for various N
    # Let's generate powerful numbers up to 200000
    SMALL_LIMIT = 200000
    powerful_small = _generate_small_powerful(SMALL_LIMIT)
    powerful_small_set = set(powerful_small)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Strategy: Try to find N with many representations.
    # For each candidate N, count how many a in powerful_small have N-a also powerful.
    # We'll search over a range of N values.
    
    # First, let's do a systematic search over N up to N_SEARCH
    # For each N, we check a in powerful_small with a <= N/2 and verify N-a is powerful
    N_SEARCH = 500000
    
    # Pre-compute: for each a in powerful_small, the set of N values where N-a might be powerful
    # is hard to predict. Instead, let's just iterate over candidate N values.
    
    # Better approach: use the structure. N = a + b where a, b powerful.
    # For small a, b = N - a. If we fix a set of small powerful numbers A,
    # then for each N, count = |{a in A : N-a is powerful and N-a >= a}|
    
    # Let's do a randomized + systematic hybrid:
    # 1. Systematically check N from 2 to N_CHECK (small range)
    # 2. Then randomly sample larger N values
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Systematic search for small N
    N_CHECK = 200000
    for N in range(2, N_CHECK + 1):
        count = 0
        pairs = []
        for a in powerful_small:
            if a > N // 2:
                break
            b = N - a
            if b < a:
                continue
            # Check if b is powerful
            if b in powerful_small_set:
                count += 1
                pairs.append([a, b])
            elif b <= SMALL_LIMIT:
                # Already checked via set
                pass
            else:
                # b > SMALL_LIMIT, need to check directly
                if _is_powerful(b):
                    count += 1
                    pairs.append([a, b])
        
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
        
        if time.time() > deadline:
            break
    
    # Randomized search for larger N
    random.seed(42)
    N_MAX_RANDOM = 10_000_000
    attempts = 0
    max_attempts = 500
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        N = random.randint(N_CHECK + 1, N_MAX_RANDOM)
        
        count = 0
        pairs = []
        for a in powerful_small:
            if a > N // 2:
                break
            b = N - a
            if b < a:
                continue
            if b in powerful_small_set:
                count += 1
                pairs.append([a, b])
            elif b <= SMALL_LIMIT:
                pass
            else:
                if _is_powerful(b):
                    count += 1
                    pairs.append([a, b])
        
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
    
    # Verify the best result
    verified_pairs = []
    for a, b in best_pairs:
        if _verify_powerful(a) and _verify_powerful(b) and a + b == best_N and a <= b:
            verified_pairs.append([a, b])
    
    if not verified_pairs:
        verified_pairs = [[1, 1]]
        best_N = 2
    
    return {'N': best_N, 'pairs': verified_pairs}
# EVOLVE-BLOCK-END
