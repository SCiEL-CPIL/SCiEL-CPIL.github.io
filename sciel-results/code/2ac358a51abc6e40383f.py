"""Find N <= 10^18 with many representations as sum of two powerful numbers."""
import time
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit.
    Powerful numbers are of the form a^2 * b^3 where b is square-free.
    """
    powerful = set()
    
    # Max b: b^3 <= limit, so b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    
    # Precompute square-free status for b up to max_b
    sq_free = [True] * (max_b + 1)
    if max_b >= 0:
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

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s buffer before 120s limit
    
    # Generate powerful numbers up to 10^7
    # This keeps the count of powerful numbers small enough for O(N^2) summation
    limit_pw = 10**7
    
    powerful = _generate_powerful(limit_pw)
    n_pw = len(powerful)
    
    # If generation took too long or list is huge, reduce limit
    if time.time() - start_time > 5:
        limit_pw = 10**6
        powerful = _generate_powerful(limit_pw)
        n_pw = len(powerful)
        
    # If n_pw is still too large for O(N^2) within time, reduce further
    # 10^6 limit -> ~1000-2000 numbers -> 10^6-4*10^6 ops (fast)
    # 10^7 limit -> ~4000-5000 numbers -> 16*10^6-25*10^6 ops (fast)
    # 10^8 limit -> ~10000 numbers -> 10^8 ops (might be slow, ~10-20s)
    
    # We use a list for counting sums because max sum is 2 * limit_pw
    # This is faster and uses less memory than a dict for dense ranges
    max_sum = 2 * limit_pw
    sum_count = [0] * (max_sum + 1)
    
    # Calculate all pair sums
    for i in range(n_pw):
        a = powerful[i]
        for j in range(i, n_pw):
            b = powerful[j]
            s = a + b
            if s > max_sum:
                break
            sum_count[s] += 1
    
    # Find the N with the maximum count
    best_n = 0
    best_count = 0
    for s in range(2, max_sum + 1):
        if sum_count[s] > best_count:
            best_count = sum_count[s]
            best_n = s
    
    # If no sums found (should not happen), fallback
    if best_n == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Collect pairs for the best N
    powerful_set = set(powerful)
    pairs = []
    for a in powerful:
        if 2 * a > best_n:
            break
        b = best_n - a
        if b in powerful_set:
            pairs.append([a, b])
            
    # Verify
    if not pairs:
         return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_n, 'pairs': pairs}
