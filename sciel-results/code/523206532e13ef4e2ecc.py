"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute smallest prime factor for fast factorization up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def prime_factors_fast(n):
        """Return set of prime factors using SPF table."""
        if n <= 1:
            return frozenset()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return frozenset(factors)

    # Precompute prime factors for all values 2..20000
    pf_cache = [frozenset() for _ in range(MAX_SUM + 1)]
    for v in range(2, MAX_SUM + 1):
        pf_cache[v] = prime_factors_fast(v)

    def count_primes_fast(nums):
        """Count distinct primes dividing all pairwise sums using cache."""
        primes = frozenset()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                s = nums[i] + nums[j]
                primes = primes | pf_cache[s]
        return len(primes)

    def count_primes_with_set(nums):
        """Count and return the set of distinct primes."""
        primes = set()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                s = nums[i] + nums[j]
                primes.update(pf_cache[s])
        return len(primes), primes

    best_count = float('inf')
    best_nums = None

    def try_update(nums):
        nonlocal best_count, best_nums
        c = count_primes_fast(nums)
        if c < best_count:
            best_count = c
            best_nums = nums[:]
        return c

    # Strategy 1: Target small prime sets and find 12-number sets where all pairwise sums are P-smooth
    # This is the most promising approach - if we can find 12 numbers whose pairwise sums
    # only use primes from a small set P, we get |P| distinct primes.
    
    target_prime_sets = [
        frozenset([2, 3]),
        frozenset([2, 3, 5]),
        frozenset([2, 3, 5, 7]),
        frozenset([2, 3, 5, 7, 11]),
        frozenset([2, 5]),
        frozenset([3, 5]),
        frozenset([2, 3, 7]),
        frozenset([2, 3, 5, 7, 11, 13]),
        frozenset([2, 3, 5, 7, 11, 13, 17]),
    ]
    
    for target_p in target_prime_sets:
        if time.time() > deadline - 10:
            break
        
        # Precompute: for each sum value, is it P-smooth?
        is_smooth = [False] * (MAX_SUM + 1)
        for v in range(2, MAX_SUM + 1):
            if pf_cache[v] <= target_p:
                is_smooth[v] = True
        
        # Count how many valid sums exist
        valid_count = sum(is_smooth[2:])
        if valid_count < 100:
            continue
        
        # Greedy search: build a set of 12 numbers where all pairwise sums are P-smooth
        # Try multiple starting points and greedy constructions
        found_for_this_p = False
        
        for start_val in range(1, 50):
            if time.time() > deadline - 10:
                break
            
            # Try greedy construction from this starting point
            nums_try = [start_val]
            
            # Candidates to try, sorted by how many existing nums they're compatible with
            candidates = list(range(start_val + 1, min(start_val + 2000, 10001)))
            random.shuffle(candidates)
            
            for cand in candidates:
                if len(nums_try) >= 12:
                    break
                # Check if cand + all existing nums are P-smooth
                ok = True
                for existing in nums_try:
                    s = cand + existing
                    if s > MAX_SUM or not is_smooth[s]:
                        ok = False
                        break
                if ok and cand not in nums_try:
                    nums_try.append(cand)
            
            if len(nums_try) == 12:
                c = count_primes_fast(nums_try)
                if c < best_count:
                    best_count = c
                    best_nums = nums_try[:]
                    found_for_this_p = True
                    # Don't break - try to find even better sets
                    # But limit further attempts for this prime set
                    if c <= len(target_p):
                        break
        
        # Also try starting from larger values
        if not found_for_this_p and time.time() < deadline - 10:
            for start_val in range(100, 1000, 100):
                if time.time() > deadline - 10:
                    break
                nums_try = [start_val]
                candidates = list(range(start_val + 1, min(start_val + 2000, 10001)))
                random.shuffle(candidates)
                
                for cand in candidates:
                    if len(nums_try) >= 12:
                        break
                    ok = True
                    for existing in nums_try:
                        s = cand + existing
                        if s > MAX_SUM or not is_smooth[s]:
                            ok = False
                            break
                    if ok and cand not in nums_try:
                        nums_try.append(cand)
                
                if len(nums_try) == 12:
                    c = count_primes_fast(nums_try)
                    if c < best_count:
                        best_count = c
                        best_nums = nums_try[:]

    # Strategy 2: Multiples of smooth base
    # If all nums are multiples of B, all sums are multiples of B.
    smooth_bases = [2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 64, 80, 96, 100, 120, 128, 144, 160, 192, 200, 240, 256, 300, 320, 384, 400, 480, 500, 512, 600, 640, 768, 800, 900, 960, 1000, 1024, 1200, 1280, 1500, 1536, 1600, 1920, 2000, 2048, 2400, 2500, 2560, 3000, 3072, 3200, 3600, 3840, 4000, 4096, 4800, 5000, 5120, 6000, 6144, 6400, 7200, 7680, 8000, 8192, 9600, 10000]
    
    for base in smooth_bases:
        if time.time() > deadline - 5:
            break
        max_q = 10000 // base
        if max_q < 12:
            continue
        # Try consecutive quotients
        for start_q in range(1, min(max_q - 11, 30) + 1):
            if time.time() > deadline - 5:
                break
            qs = list(range(start_q, start_q + 12))
            nums = [q * base for q in qs]
            try_update(nums)
        # Try a few random subsets
        for _ in range(15):
            if time.time() > deadline - 5:
                break
            qs = random.sample(range(1, max_q + 1), 12)
            nums = [q * base for q in qs]
            try_update(nums)

    # Strategy 3: Arithmetic progressions with smooth difference
    for d in [2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 64, 80, 96, 100, 120, 128, 144, 160, 192, 200, 240, 256, 300, 320, 384, 400, 480, 500, 512, 600, 640, 768, 800, 900, 960, 1000]:
        if time.time() > deadline - 5:
            break
        max_a = 10000 - 11 * d
        if max_a < 1:
            continue
        # Try a few values of a
        a_candidates = set(range(1, min(max_a + 1, 30) + 1))
        for _ in range(20):
            if max_a >= 1:
                a_candidates.add(random.randint(1, max_a))
        
        for a in a_candidates:
            if time.time() > deadline - 5:
                break
            if a < 1 or a + 11*d > 10000:
                continue
            nums = [a + i*d for i in range(12)]
            try_update(nums)

    # Strategy 4: Local improvement with incremental evaluation
    def count_primes_incremental(nums, i, new_val):
        """Compute new prime count after changing nums[i] to new_val.
        Only recompute sums involving index i."""
        primes = set()
        n = len(nums)
        # All pairs NOT involving i
        for a in range(n):
            if a == i:
                continue
            for b in range(a+1, n):
                if b == i:
                    continue
                s = nums[a] + nums[b]
                primes.update(pf_cache[s])
        # Pairs involving i with new_val
        for j in range(n):
            if j == i:
                continue
            s = new_val + nums[j]
            primes.update(pf_cache[s])
        return len(primes)

    if best_nums is not None and time.time() < deadline - 10:
        nums = best_nums[:]
        c = best_count
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old_val = nums[i]
                # Try wider range of deltas
                for delta in list(range(-50, 51)) + list(range(100, 201, 5)) + list(range(-200, -100, 5)):
                    new_val = old_val + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in nums:
                        continue
                    nums[i] = new_val
                    c2 = count_primes_incremental(nums, i, new_val)
                    if c2 < c:
                        c = c2
                        if c < best_count:
                            best_count = c
                            best_nums = nums[:]
                        improved = True
                    else:
                        nums[i] = old_val

    # Strategy 5: Random restarts with local improvement
    if time.time() < deadline - 15:
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            # Generate random structured candidate
            mode = random.randint(0, 3)
            if mode == 0:
                base = random.choice([2,3,4,5,6,8,10,12,15,16,20,24,30,32,40,48,60,64,80,96,100,120,128,144,160,192,200,240,256,300,320,384,400,480,500,512,600,640,768,800,900,960,1000,1024,1200,1280,1500,1536,1600,1920,2000,2048,2400,2500,2560,3000,3072,3200,3600,3840,4000,4096,4800,5000,5120,6000,6144,6400,7200,7680,8000,8192,9600,10000])
                max_q = 10000 // base
                if max_q < 12:
                    continue
                qs = random.sample(range(1, max_q + 1), 12)
                nums = [q * base for q in qs]
            elif mode == 1:
                d = random.choice([2,3,4,5,6,8,10,12,15,16,20,24,30,32,40,48,60,64,80,96,100,120,128,144,160,192,200,240,256,300,320,384,400,480,500,512,600,640,768,800,900,960,1000])
                max_a = 10000 - 11 * d
                if max_a < 1:
                    continue
                a = random.randint(1, max_a)
                nums = [a + i*d for i in range(12)]
            elif mode == 2:
                M = random.choice([4,6,8,10,12,14,16,18,20,24,28,30,32,36,40,42,48,50,54,56,60,64,70,72,80,84,90,96,100,108,112,120,128,140,144,150,160,168,180,192,200,210,216,224,240,250,252,256,270,280,288,300,320,336,350,360,375,384,400,420,432,448,450,480,500,504,512,540,560,576,600,630,640,672,700,720,750,768,800,840,864,900,960,1000])
                r = M // 2
                if r < 1:
                    continue
                last = r + 11 * M
                if last > 10000:
                    continue
                start_k = random.randint(0, max(0, (10000 - r) // M - 11))
                nums = [r + (start_k + k)*M for k in range(12)]
            else:
                nums = random.sample(range(1, 10001), 12)
            
            c = count_primes_fast(nums)
            if c <= best_count + 2:
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for i in range(12):
                        if time.time() > deadline - 5:
                            break
                        old_val = nums[i]
                        for delta in range(-20, 21):
                            if delta == 0:
                                continue
                            new_val = old_val + delta
                            if new_val < 1 or new_val > 10000:
                                continue
                            if new_val in nums:
                                continue
                            nums[i] = new_val
                            c2 = count_primes_incremental(nums, i, new_val)
                            if c2 < c:
                                c = c2
                                if c < best_count:
                                    best_count = c
                                    best_nums = nums[:]
                                improved = True
                            else:
                                nums[i] = old_val

    # Strategy 6: Simulated annealing polish
    if best_nums is not None and time.time() < deadline - 5:
        nums = best_nums[:]
        c = best_count
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        
        for step in range(2000):
            if time.time() > deadline - 2:
                break
            if T < T_min:
                break
            
            i = random.randint(0, 11)
            old_val = nums[i]
            # Random perturbation
            if random.random() < 0.3:
                new_val = random.randint(1, 10000)
            else:
                delta = random.randint(-30, 30)
                new_val = old_val + delta
            
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in nums:
                continue
            
            nums[i] = new_val
            c2 = count_primes_incremental(nums, i, new_val)
            
            if c2 < c:
                c = c2
                if c < best_count:
                    best_count = c
                    best_nums = nums[:]
            else:
                # Accept with probability exp(-(c2-c)/T)
                if random.random() < math.exp(-(c2 - c) / T):
                    c = c2
                else:
                    nums[i] = old_val
            
            T *= cooling

    if best_nums is None:
        return list(range(1, 13))
    
    return best_nums
# EVOLVE-BLOCK-END
