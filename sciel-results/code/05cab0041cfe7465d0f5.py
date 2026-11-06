"""Find N <= 10^18 with many representations as sum of two powerful numbers."""
import time
import random
from math import isqrt

def _is_square_free(n):
    """Check if n is square-free."""
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
    # Check if n is a perfect cube
    c = round(n ** (1/3))
    for cc in range(max(2, c-2), c+3):
        if cc**3 == n:
            return False
    return True


def _generate_powerful(limit):
    """Generate all powerful numbers <= limit.
    Powerful numbers are of the form a^2 * b^3 where b is square-free.
    """
    powerful = set([1])
    
    # Max b: b^3 <= limit, so b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    
    # Precompute square-free status for b up to max_b
    sq_free = [True] * (max_b + 1)
    sq_free[0] = False
    if max_b >= 1:
        sq_free[1] = True
    
    # Sieve for square-free
    for p in range(2, int(limit ** (1/6)) + 2):
        if sq_free[p]:
            p2 = p * p
            for k in range(p2, max_b + 1, p2):
                sq_free[k] = False
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    return sorted(powerful)


def _count_reps(N, powerful_list, powerful_set):
    """Count the number of pairs (a, b) with a <= b, a + b = N, a, b in powerful_set."""
    count = 0
    pairs = []
    for a in powerful_list:
        if 2 * a > N:
            break
        b = N - a
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
