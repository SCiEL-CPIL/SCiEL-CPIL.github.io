"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _is_powerful(n):
    """Check if n is a powerful number using trial division."""
    if n <= 1:
        return True
    d = 2
    while d * d <= n:
        if n % d == 0:
            count = 0
            while n % d == 0:
                n //= d
                count += 1
            if count == 1:
                return False
        d += 1 if d == 2 else 2
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    # b^3 <= limit, so b <= limit^(1/3)
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    # Sieve for square-free numbers up to max_b
    is_sq_free = [True] * (max_b + 1)
    is_sq_free[0] = False
    is_sq_free[1] = True
    for i in range(2, isqrt(max_b) + 1):
        if is_sq_free[i]:
            sq = i * i
            for j in range(sq, max_b + 1, sq):
                is_sq_free[j] = False
    # Generate powerful numbers a^2 * b^3 <= limit
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b * b * b
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Phase 1: Search in a moderate range to find a good candidate
    limit = 100000
    powerful = _generate_powerful(limit)
    powerful_set = set(powerful)
    
    best_n = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    for n in range(2, limit + 1):
        if time.time() > deadline:
            break
        count = 0
        pairs = []
        for a in powerful:
            if 2 * a > n:
                break
            b = n - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        if count > best_count:
            best_count = count
            best_n = n
            best_pairs = pairs
    
    # Phase 2: Try to find N with more representations using a larger range
    # Focus on N that are sums of many powerful numbers
    # Use a heuristic: N should be even and have many divisors
    limit2 = 500000
    powerful2 = _generate_powerful(limit2)
    powerful2_set = set(powerful2)
    
    # Try specific promising N values
    # N = 2 * k where k is highly composite
    for k in range(1000, 250000):
        if time.time() > deadline:
            break
        n = 2 * k
        if n > limit2:
            break
        count = 0
        pairs = []
        for a in powerful2:
            if 2 * a > n:
                break
            b = n - a
            if b in powerful2_set:
                count += 1
                pairs.append([a, b])
        if count > best_count:
            best_count = count
            best_n = n
            best_pairs = pairs
    
    # Phase 3: Try to extend to larger N using the structure
    # For N up to 10^18, we can't enumerate all powerful numbers
    # But we can try N = a + b where a and b are powerful
    # Focus on N that are sums of small powerful numbers
    # Use the fact that 1 is powerful, so N-1 must be powerful
    # Also N-4, N-8, etc.
    
    # Try N values that are known to have many representations
    # Based on the seed, N around 20000 had ~3 representations
    # Let's try to find N with more
    
    # Use a different approach: for each pair of powerful numbers, compute their sum
    # and count how many times each sum appears
    # This is O(P^2) where P is the number of powerful numbers
    # For limit=500000, P is about 10000, so P^2 = 100M which is too slow
    
    # Instead, use the fact that we want N = a + b with a <= b
    # For a fixed a, b = N - a must be powerful
    # So for a fixed N, we check how many a in powerful have N-a in powerful
    
    # Let's try a smarter search: focus on N that are multiples of small numbers
    # N = 2^k * m where m is odd
    for k in range(1, 20):
        if time.time() > deadline:
            break
        base = 2 ** k
        for m in range(1, 500000 // base):
            n = base * m
            if n > limit2:
                break
            count = 0
            pairs = []
            for a in powerful2:
                if 2 * a > n:
                    break
                b = n - a
                if b in powerful2_set:
                    count += 1
                    pairs.append([a, b])
            if count > best_count:
                best_count = count
                best_n = n
                best_pairs = pairs
    
    # Phase 4: Try to find N > limit2 using a different strategy
    # For large N, we can use the fact that if N is powerful and N-1 is powerful,
    # that's one representation. But we need many.
    
    # Try N = x^2 + y^2 where x^2 and y^2 are powerful (always true)
    # But we need x^2 + y^2 = N, so we need many representations as sum of two squares
    # that are also powerful (which they always are)
    # Wait, x^2 is always powerful, so any N that is a sum of two squares has
    # at least one representation. We need N to be a sum of many pairs of squares.
    
    # The number of representations of N as a sum of two squares is related to
    # the number of divisors of N. N with many divisors have many representations.
    
    # Let's find N with many divisors that are also sums of two squares
    # N = 2^a * p1^b1 * p2^b2 * ... where pi are primes congruent to 1 mod 4
    # The number of representations is (1 + b1)(1 + b2)... * 2^k where k is the
    # number of primes congruent to 1 mod 4
    
    # For N up to 10^18, we can try N = 2^a * 5^b * 13^c * 17^d * 29^e * ...
    
    # Let's try a few specific N values
    candidate_ns = []
    for a in range(0, 60):
        for b in range(0, 26):
            for c in range(0, 14):
                n = (2 ** a) * (5 ** b) * (13 ** c)
                if n > 10**18:
                    break
                if n > 1000000:
                    candidate_ns.append(n)
                if n > 10**18:
                    break
            if (2 ** a) * (5 ** b) * (13 ** c) > 10**18:
                break
        if (2 ** a) * (5 ** b) > 10**18:
            break
    
    # For each candidate N, check how many powerful numbers a have N-a powerful
    # But for large N, we can't enumerate all powerful numbers
    # Instead, use the fact that a must be <= N/2 and powerful
    # For large N, the number of powerful numbers <= N/2 is about (N/2)^(1/3) * constant
    # For N = 10^18, that's about 10^6 * constant, which is feasible
    
    # Generate powerful numbers up to N/2 for the best candidate
    if candidate_ns:
        max_n = min(candidate_ns[-1], 10**18)
        # We can't generate all powerful numbers up to 5*10^17
        # Instead, use a probabilistic check
    
    # For now, return the best found so far
    return {'N': best_n, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
