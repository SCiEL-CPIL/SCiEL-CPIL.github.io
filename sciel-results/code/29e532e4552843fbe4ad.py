import math
import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def time_left():
        return deadline - time.time()

    # 1. Generate powerful numbers up to a limit L.
    # A powerful number is one where every prime factor appears with exponent >= 2.
    # We can generate them by iterating primes and their powers p^2, p^3, ...
    # and multiplying them together.
    
    # Estimate L based on time.
    # If L = 10^6, count of powerfuls is ~400. Pairs ~160k. Very fast.
    # If L = 10^7, count is ~1000. Pairs ~1M. ~1s.
    # If L = 10^8, count is ~2000. Pairs ~4M. ~4s.
    # If L = 10^9, count is ~4000. Pairs ~16M. ~16s.
    
    # We want to maximize the count. The count of representations for a random N
    # grows with L. But we also need N to be valid (<= 10^18).
    # Since a,b <= L, N <= 2L. So 2L <= 10^18 is easily satisfied for L up to 10^18.
    # The bottleneck is the O(K^2) pair generation, where K is count of powerfuls.
    
    # Let's aim for L such that K^2 is manageable within time.
    # Let's try to go as high as possible.
    
    # Generate primes up to sqrt(L)
    # Actually, to generate powerfuls up to L, we only need primes p such that p^2 <= L, i.e., p <= sqrt(L).
    
    # Let's pick L dynamically or try a few L values.
    # Let's start with L = 10^8.
    # Primes up to 10^4.
    
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    # Try increasing L
    # L = 10^6, 10^7, 10^8, 10^9
    
    best_pairs = []
    best_N = 0
    
    # We will try L = 10^9 if time permits, else lower.
    # Let's just do L = 10^8 first as a safe bet, then try higher if fast.
    # Actually, let's just do one large L = 10^9 and see.
    # K ~ 4000. Pairs ~ 8 million (unique a<=b). 
    # Python loop for 8M pairs might take ~2-4 seconds.
    # Counter update is fast.
    
    L = 10**9
    # Check time. If we are already close to deadline, reduce L.
    # Let's just run with L=10^9.
    
    max_prime = int(math.isqrt(L))
    primes = get_primes(max_prime)
    
    powerfuls = [1]
    
    # Generate powerful numbers
    # Recursive/iterative generation
    # For each prime p, we can multiply existing powerfuls by p^2, p^3, ... as long as <= L
    
    # This is a standard "multiple knapsack" style generation.
    # To ensure uniqueness and avoid duplicates, we process primes in order and only multiply by p^e
    # where e >= 1, and the resulting number is >= the current number's smallest prime factor?
    # No, simpler:
    # Start with [1].
    # For each prime p:
    #   new_powerfuls = []
    #   for x in current_powerfuls:
    #      if x * p^2 > L: break
    #      val = x * p^2
    #      while val <= L:
    #          new_powerfuls.append(val)
    #          val *= p
    #   current_powerfuls += new_powerfuls
    # This generates all powerful numbers uniquely.
    
    current_powerfuls = [1]
    for p in primes:
        p2 = p * p
        if p2 > L:
            break
        new_adds = []
        for x in current_powerfuls:
            if x * p2 > L:
                break
            val = x * p2
            while val <= L:
                new_adds.append(val)
                val *= p
        if new_adds:
            current_powerfuls.extend(new_adds)
            
    # current_powerfuls contains all powerful numbers <= L
    # Sort them to handle a <= b easily
    current_powerfuls.sort()
    K = len(current_powerfuls)
    
    # If K is too large, we might TLE.
    # Estimate: 10^9 -> K ~ 4000. 4000^2 / 2 = 8M pairs.
    # 8M iterations in Python is ~1-2s. Safe.
    # 10^10 -> K ~ 10000. 50M pairs. ~10-20s. Risky.
    # Let's stick to L=10^9 or maybe 2*10^9.
    
    # Counter for sums
    sums = Counter()
    
    # Optimize inner loop
    pow_list = current_powerfuls
    n_pow = K
    
    # We can use numpy for speed?
    # Numpy broadcasting for 4000x4000 is 16M floats/ints. 128MB. Doable.
    # But summing and counting unique pairs...
    # Pure Python loop is safer for memory and often faster for sparse logic if optimized,
    # but Counter update in Python is slower than numpy histogram.
    
    # Let's try pure Python first with tight loop.
    # If time_left < 5, we might skip.
    
    # To speed up, we can break early if x^2 > L? No, x is powerful, not x^2.
    # Condition: a + b <= 2L.
    # We want to find N with max count.
    
    # Optimization:
    # For i in range(K):
    #   a = pow_list[i]
    #   for j in range(i, K):
    #     b = pow_list[j]
    #     s = a + b
    #     sums[s] += 1
    #     # Optional: if a + pow_list[j] > some_limit? No limit on N except 10^18.
    
    # This is O(K^2).
    
    # Let's run it.
    if time_left() > 10:
        for i in range(K):
            a = pow_list[i]
            # We can break inner loop if a + b > 10^18?
            # Max a ~ 10^9, max b ~ 10^9. Sum ~ 2*10^9. Well below 10^18.
            # So no need to break based on upper limit of N.
            for j in range(i, K):
                b = pow_list[j]
                sums[a + b] += 1
                
            if i % 500 == 0 and time_left() < 5:
                break
                
    # Find max
    if sums:
        best_N, best_count = sums.most_common(1)[0]
        # Reconstruct pairs
        # We need to find all (a,b) such that a+b == best_N
        # We can do this by iterating a in pow_list, checking if b = best_N - a is in pow_list and a <= b.
        # To be efficient, use a set of powerfuls.
        pow_set = set(pow_list)
        pairs = []
        for a in pow_list:
            if a > best_N // 2:
                break
            b = best_N - a
            if b in pow_set and a <= b:
                pairs.append([a, b])
        
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = best_N

    # If we have time, try a larger L?
    # The score is the number of pairs.
    # As L increases, the number of powerfuls increases, and the max count of sums typically increases.
    # Let's see if we have time for L = 5 * 10^9.
    # K would be ~ 2x larger. Time ~ 4x.
    # If L=10^9 took 2s, L=5*10^9 might take 8s.
    # Total time used so far: gen primes + gen powerfuls + pairs.
    # If time_left > 20, we can try again with larger L.
    
    if time_left() > 20:
        L2 = 5 * 10**9
        max_prime2 = int(math.isqrt(L2))
        primes2 = get_primes(max_prime2)
        
        current_powerfuls2 = [1]
        for p in primes2:
            p2 = p * p
            if p2 > L2:
                break
            new_adds = []
            for x in current_powerfuls2:
                if x * p2 > L2:
                    break
                val = x * p2
                while val <= L2:
                    new_adds.append(val)
                    val *= p
            if new_adds:
                current_powerfuls2.extend(new_adds)
                
        current_powerfuls2.sort()
        K2 = len(current_powerfuls2)
        
        # Estimate time for K2
        # If K2 > 8000, it might be too slow (32M pairs ~ 6-8s)
        # Let's check K2.
        if K2 < 8000: # Heuristic
             sums2 = Counter()
             pow_list2 = current_powerfuls2
             for i in range(K2):
                 a = pow_list2[i]
                 for j in range(i, K2):
                     b = pow_list2[j]
                     sums2[a + b] += 1
                 if i % 1000 == 0 and time_left() < 5:
                     break
             
             if sums2:
                 best_N2, best_count2 = sums2.most_common(1)[0]
                 pow_set2 = set(pow_list2)
                 pairs2 = []
                 for a in pow_list2:
                     if a > best_N2 // 2:
                         break
                     b = best_N2 - a
                     if b in pow_set2 and a <= b:
                         pairs2.append([a, b])
                 
                 if len(pairs2) > len(best_pairs):
                     best_pairs = pairs2
                     best_N = best_N2

    # Fallback
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
