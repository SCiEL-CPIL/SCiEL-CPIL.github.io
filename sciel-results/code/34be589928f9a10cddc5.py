import math
import time
import bisect
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Strategy: 
    # 1. Generate powerful numbers up to a large limit L (e.g., 10^12).
    # 2. Use FFT convolution to count pairs (a, b) such that a + b = N for all N.
    # 3. Find N with max count.
    # 4. Retrieve actual pairs for that N by scanning the powerful list.
    
    # Powerful numbers count M(L) ~ L^(1/3) * constant.
    # For L = 10^12, M ~ 10^4 * 20 ~ 200,000.
    # FFT size ~ 2 * L = 2 * 10^12 is too big for memory (2^41).
    # Wait, FFT size needs to be at least max_N. 
    # If we want N up to 10^12, FFT size is 2^41 ~ 2 * 10^12 complex numbers.
    # Memory: 2 * 10^12 * 16 bytes (complex128) = 32 TB. Impossible.
    
    # Correction:
    # We cannot FFT over the full range 1..10^12 directly.
    # However, we can use the fact that powerful numbers are sparse.
    # But convolution of sparse vectors is still dense in the result space.
    
    # Alternative:
    # The current program scores 2.86 with L=10^9.
    # M(10^9) ~ 10^3 * 20 = 20,000.
    # O(M^2) = 4 * 10^8 operations. In Python, this is slow (~10-20s).
    # The current program likely runs close to the deadline or uses optimizations.
    
    # Let's look at the "Inspiration 1" which mentions Number-Theoretic Optimization.
    # And "Inspiration 3" notes the current program scores 2.87.
    
    # If I can't do FFT on 10^12, can I do better with L=10^10 or 10^11?
    # M(10^10) ~ 10^(10/3) ~ 10^3.33 * C ~ 2000 * 20 = 40,000.
    # M(10^11) ~ 10^(11/3) ~ 10^3.66 * C ~ 5000 * 20 = 100,000.
    # M(10^12) ~ 10^4 * 20 = 200,000.
    
    # O(M^2) for M=200,000 is 4 * 10^10. Too slow in Python.
    # O(M^2) for M=100,000 is 10^10. Too slow.
    # O(M^2) for M=50,000 is 2.5 * 10^9. Might be borderline in C, too slow in Python.
    
    # So we are stuck with M ~ 20,000 - 30,000 for Python loops.
    # This corresponds to L ~ 10^9 - 10^10.
    
    # Is there a smarter way?
    # Maybe use NumPy vectorization for the inner loop?
    # For each a, we need to count how many b in powerful_list satisfy a <= b <= L - a.
    # This is just `bisect` range counting.
    # The current program does this.
    
    # Let's re-read the current program.
    # It iterates `a` in `p_list`.
    # For each `a`, it finds `left_idx` and `right_idx`.
    # Then it iterates `i` from `left_idx` to `right_idx` and updates `counts[a+b]`.
    # This is the bottleneck: the inner loop `for i in range(left_idx, right_idx)`.
    # Total operations is sum over a of (number of b's).
    # This is roughly M^2 / 2.
    
    # Can we optimize the counting phase?
    # We want `counts[N]` = number of pairs (a, b) with a+b=N.
    # This is exactly the coefficient of x^N in P(x)^2 where P(x) = sum x^a for a in powerful.
    # We can use FFT if the range of N is small enough.
    # If we restrict N to a smaller range, say N <= 10^8, we can FFT.
    # But we want to maximize count, which usually happens for larger N.
    
    # However, note that `counts[N]` grows with N.
    # If we search N up to 10^9, we might find a better N than searching N up to 10^8.
    
    # Let's try a hybrid approach:
    # 1. Generate powerful numbers up to L = 10^10.
    # 2. Instead of O(M^2) counting, we can use a dictionary of counts, but that's what the current program does.
    # 3. The current program's bottleneck is the Python loop.
    
    # What if we use NumPy to accelerate the counting?
    # We can't easily vectorize the update `counts[a+b] += 1` for varying a+b.
    
    # Alternative:
    # Use the fact that we only need the MAX count.
    # We don't need the exact count for every N, just the max.
    # But we need to know which N has the max count to return the pairs.
    
    # Let's try to increase L to 10^10 and see if it runs in time.
    # M(10^10) is approx 40,000.
    # M^2 / 2 = 8 * 10^8 operations.
    # In Python, 10^8 simple ops take ~1s? No, usually ~0.1-0.5s for simple C-optimized, but Python loops are ~10^7/s.
    # So 8 * 10^8 Python loop iterations would take ~80 seconds.
    # This is risky but might fit in 115s if optimized.
    
    # Let's try L = 10^10.
    
    L = 10**10
    
    # Generate powerful numbers up to L
    # Powerful numbers are of form a^2 * b^3 where b is square-free.
    # b <= L^(1/3) = 10^(10/3) = 10^3.33 = 2154.
    # a <= sqrt(L / b^3).
    
    # Generate primes up to L^(1/3) for square-free check?
    # No, we need to check if b is square-free.
    # b is small (<= 2154). We can precompute square-free numbers up to 2154.
    
    limit_b = int(L ** (1.0/3.0)) + 2
    is_prime = [True] * (limit_b + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_b ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_b + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def is_square_free(n):
        if n < 2:
            return True
        for p in primes:
            if p * p > n:
                break
            if n % (p * p) == 0:
                return False
        return True
        
    powerful_set = set()
    for b in range(1, limit_b + 1):
        if not is_square_free(b):
            continue
        b3 = b ** 3
        if b3 > L:
            break
        max_a = int(math.isqrt(L // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= L:
                powerful_set.add(val)
                
    powerful_list = sorted(powerful_set)
    powerful_set = set(powerful_list)
    M = len(powerful_list)
    
    # Counting phase
    # To speed up, we can break early if time is running out.
    # But we need a complete count to find the max.
    # If we stop early, the max might be wrong.
    # However, if we assume the max is likely to be found in the "middle" or "upper" range,
    # and we iterate a in order, we might miss some.
    
    # Let's stick to the full loop but optimize the inner part.
    # Using a local variable for the list and set.
    p_list = powerful_list
    p_set = powerful_set
    L_half = L // 2
    
    counts = {}
    best_N = 0
    best_count = 0
    
    # Optimization: 
    # Instead of checking time in every inner iteration, check every 1000 outer iterations.
    # The inner loop is the heavy part.
    
    for idx_a, a in enumerate(p_list):
        if a > L_half:
            break
            
        if idx_a % 100 == 0 and time.time() > deadline - 5.0:
            # If we are running out of time, we must return what we have.
            # But the count is incomplete. 
            # Is it better to return a partial result?
            # The task says "certified lower bound". 
            # If we stop early, the pairs we return are valid, but the count might be lower than the true max.
            # However, if we return an incomplete set of pairs for the best_N found so far,
            # is it valid? 
            # "pairs: [[a, b], ...] with ... a + b = N, where a and b are powerful"
            # It doesn't say the list of pairs must be COMPLETE.
            # "The score is the number of pairs".
            # So if we return a subset of pairs for the best N found, the score is the size of that subset.
            # This is a valid certified lower bound.
            # So we can safely return early with the best_N and the pairs we've found for it.
            # BUT, we need to collect the pairs for best_N.
            # The current code collects pairs at the end.
            # If we return early, we need to have the pairs for best_N.
            # We can store the pairs for the current best_N as we go?
            # No, that's expensive.
            
            # Better: If time is up, break and return the best_N found so far.
            # Then in the retrieval phase, we only find pairs that are in the powerful_set.
            # Since we didn't count ALL pairs, the `best_count` might be an overestimate?
            # No, `best_count` is the count of pairs we *did* find.
            # If we break early, we haven't found all pairs for best_N.
            # So the score will be lower than the true max.
            # But it's better than 0 or timeout.
            break
            
        left_idx = bisect.bisect_left(p_list, a)
        right_idx = bisect.bisect_right(p_list, L - a)
        
        # Inner loop
        # We can try to speed this up by not using a dictionary for every update?
        # No, we need to count sums.
        
        for i in range(left_idx, right_idx):
            b = p_list[i]
            N = a + b
            if N in counts:
                c = counts[N] + 1
                counts[N] = c
            else:
                counts[N] = 1
                c = 1
                
            if c > best_count:
                best_count = c
                best_N = N

    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Retrieve pairs for best_N
    # We must ensure we find ALL pairs for best_N to maximize the score.
    # Since we might have broken out of the counting loop early,
    # we should do a fresh scan for pairs for best_N.
    # This scan is O(M) and fast.
    
    pairs = []
    for a in p_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    # If we broke early, `best_count` might be less than len(pairs).
    # We return len(pairs) as the score.
    # This is a valid lower bound.
    
    return {'N': best_N, 'pairs': pairs}

