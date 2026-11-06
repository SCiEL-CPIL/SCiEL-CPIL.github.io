"""Find N <= 10^18 with many representations as sum of two powerful numbers."""
import time
import random
from math import isqrt

def _is_square_free(n):
    """Check if n is square-free using trial division up to cbrt(n) then check remainder."""
    if n < 2:
        return True
    # Check small primes first
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n == 1:
        return True
    # Check if n is a perfect square
    r = isqrt(n)
    if r * r == n:
        return False
    # Check if n is a perfect cube (would mean it has a squared prime factor)
    c = round(n ** (1/3))
    for cc in range(c-2, c+3):
        if cc >= 2 and cc**3 == n:
            return False
    # n is now either 1, prime, or product of two primes > 97
    # If it's a product of two primes p*q, it's square-free
    # If it's p^2, we already caught perfect squares
    # If it's p^3 for some prime, we need to check
    # Actually for n after dividing out small primes, if n > 1 and not a perfect square,
    # and n < (100)^2 = 10000 roughly... let's be more careful
    # After removing all factors of primes <= 97, n is either 1, prime, p*q, or p^2 (caught), or p^3...
    # p^3 for p > 97 means p^3 > 97^3 = 912673. If n <= 10^6 or so we're fine.
    # For larger n, if n = p^3, then isqrt(n) won't catch it. But p^3 being square-free? No! p^3 has p appearing 3 times, so it's NOT square-free.
    # Wait: square-free means no prime divides it more than once. p^3 is not square-free.
    # So we need to check if n is a perfect cube.
    # But for large n, cube root check is tricky. Let's use integer arithmetic.
    # Actually, if n = p^k for k >= 2, then n is not square-free.
    # After removing small prime factors, n could be p, p*q, p^2 (caught), p^3, p^4 (caught as square), etc.
    # For n > 1 after small prime removal:
    # - If n is prime: square-free
    # - If n = p*q (distinct primes > 97): square-free
    # - If n = p^2: caught by isqrt check
    # - If n = p^3: NOT square-free
    # - If n = p^k for k >= 4: caught by isqrt if k even, or... p^4 = (p^2)^2 caught, p^5 = p^4 * p, isqrt(p^5) = p^2*sqrt(p) not integer unless p is square... 
    # This is getting complex. Let me use a different approach.
    
    # Simpler: after removing small prime factors completely, check if remaining n is a perfect power with exponent >= 2
    # For our purposes (b up to ~10^6), n after small prime removal is small enough that we can check.
    # Actually, let's just do trial division up to isqrt(n) for the remaining part if it's small enough.
    if n < 10**8:
        i = 98  # start after 97
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 2 if i > 2 else 1
            # Actually this is too slow for large n. Let me reconsider.
            break  # Don't do this, it's wrong
    
    # For the range we care about (b <= ~2*10^6), after removing primes <= 97,
    # the remaining n is at most ~2*10^6 / (2*3*5*7*11*13*17*19*23*29*31*37*41*43*47*53*59*61*67*71*73*79*83*89*97)
    # which is tiny. So n after small prime removal is small.
    # Let me just do a proper check.
    
    # Redo: proper square-free check
    return True  # Placeholder - will fix below


def _is_square_free_proper(n):
    """Proper square-free check."""
    if n < 2:
        return True
    # Trial divide by primes up to cbrt(n)
    # For n up to ~10^6, this is fast
    i = 2
    while i * i * i <= n:
        if n % (i * i) == 0:
            return False
        while n % i == 0:
            n //= i
        i += 1 if i == 2 else 2
    # Now n has no prime factor <= cbrt(original n) with multiplicity >= 2
    # n is either 1, prime, product of two primes, or a square of a prime > cbrt
    r = isqrt(n)
    if r * r == n:
        return False
    return True


def _generate_powerful(limit):
    """Generate all powerful numbers <= limit.
    Powerful numbers are of the form a^2 * b^3 where b is square-free.
    """
    powerful = set([1])
    
    # Max b: b^3 <= limit, so b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    
    # For each square-free b, a^2 * b^3 <= limit => a <= sqrt(limit / b^3)
    for b in range(1, max_b + 1):
        if not _is_square_free_proper(b):
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)


def _count_reps(n, powerful_set, powerful_list):
    """Count the number of pairs (a, b) with a <= b, a + b = n, a, b in powerful_set."""
    count = 0
    pairs = []
    for a in powerful_list:
        if 2 * a > n:
            break
        b = n - a
        if b in powerful_set:
            count += 1
            pairs.append([a, b])
    return count, pairs


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer before 120s limit
    
    # Generate powerful numbers up to a reasonable limit
    # We need N <= 10^18, so powerful numbers up to 10^18
    # But generating all powerful numbers up to 10^18 is too many
    # Instead, generate up to 10^12 or so, and use random sampling for candidates
    
    # Let's generate powerful numbers up to 10^11
    # This should give us a good set to work with
    limit_pw = 10**11
    
    # Generate powerful numbers
    powerful = _generate_powerful(limit_pw)
    powerful_set = set(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 30:
        # Not enough time, use a smaller approach
        limit_pw = 10**9
        powerful = _generate_powerful(limit_pw)
        powerful_set = set(powerful)
    
    # Strategy: Find N with many representations
    # N = a + b where a, b are powerful
    # To find N with many reps, we want many pairs (a, b) from our powerful set that sum to the same N
    
    # Approach: Use a dictionary to count how many ways each sum can be formed
    # But this is O(|powerful|^2) which is too slow for large sets
    
    # Better approach: Random sampling of candidates
    # Pick random powerful numbers a, compute b = N - a for random N, check if b is powerful
    
    # Alternative: For each powerful number a, for each powerful number b, add a+b to a counter
    # But this is O(n^2)
    
    # Let's use a smarter approach:
    # 1. Generate powerful numbers up to limit_pw
    # 2. For a set of candidate N values, count representations
    # 3. Use random sampling to find good N values
    
    # Candidate N values: sums of two powerful numbers
    # Let's sample some sums and count their frequencies
    
    # First, let's try a direct approach for smaller sets
    n_pw = len(powerful)
    
    if n_pw < 500000:
        # Use dictionary to count sums
        sum_count = {}
        for i in range(n_pw):
            a = powerful[i]
            for j in range(i, n_pw):
                b = powerful[j]
                s = a + b
                if s > 10**18:
                    break
                sum_count[s] = sum_count.get(s, 0) + 1
        
        # Find the N with maximum count
        if sum_count:
            best_n = max(sum_count, key=sum_count.get)
            best_count = sum_count[best_n]
            
            # Verify and collect pairs
            pairs = []
            for a in powerful:
                if 2 * a > best_n:
                    break
                b = best_n - a
                if b in powerful_set:
                    pairs.append([a, b])
            
            if pairs:
                return {'N': best_n, 'pairs': pairs}
    
    # Fallback: Random sampling
    random.seed(42)
    best_n = None
    best_pairs = []
    best_count = 0
    
    # Sample candidate N values
    for _ in range(100000):
        if time.time() > deadline:
            break
        
        # Pick a random N in a reasonable range
        # N should be sum of two powerful numbers, so N >= 2 (1+1)
        # Let's sample N from sums of random powerful pairs
        if len(powerful) < 2:
            break
        idx1 = random.randint(0, len(powerful) - 1)
        idx2 = random.randint(0, len(powerful) - 1)
        n_candidate = powerful[idx1] + powerful[idx2]
        
        if n_candidate > 10**18 or n_candidate < 2:
            continue
        
        # Count representations for this N
        count = 0
        pairs = []
        for a in powerful:
            if 2 * a > n_candidate:
                break
            b = n_candidate - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        
        if count > best_count:
            best_count = count
            best_n = n_candidate
            best_pairs = pairs
    
    if best_n is not None and best_pairs:
        return {'N': best_n, 'pairs': best_pairs}
    
    # Last resort: return a simple valid answer
    return {'N': 2, 'pairs': [[1, 1]]}
