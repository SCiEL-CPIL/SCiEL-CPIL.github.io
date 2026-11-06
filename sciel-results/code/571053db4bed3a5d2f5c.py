"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations, product

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

    def incremental_count_change(nums, idx, old_val, new_val):
        """
        Compute change in distinct prime count when replacing nums[idx] from old_val to new_val.
        Returns (delta_count, new_primes_set) where delta_count = new_count - old_count.
        """
        n = len(nums)
        # Primes that appear ONLY in pairs involving idx (with old_val)
        # These may disappear if new_val doesn't bring them back
        # Primes that appear in pairs NOT involving idx must remain
        
        # First, compute primes from pairs NOT involving idx (these are fixed)
        fixed_primes = set()
        for i in range(n):
            if i == idx:
                continue
            for j in range(i+1, n):
                if j == idx:
                    continue
                s = nums[i] + nums[j]
                fixed_primes.update(pf_cache[s])
        
        # Primes from pairs involving idx with old_val
        old_idx_primes = set()
        for j in range(n):
            if j == idx:
                continue
            s = old_val + nums[j]
            old_idx_primes.update(pf_cache[s])
        
        # Primes from pairs involving idx with new_val
        new_idx_primes = set()
        for j in range(n):
            if j == idx:
                continue
            s = new_val + nums[j]
            new_idx_primes.update(pf_cache[s])
        
        # Total old primes
        old_total = fixed_primes | old_idx_primes
        # Total new primes
        new_total = fixed_primes | new_idx_primes
        
        return len(new_total) - len(old_total), new_total

    best_count = float('inf')
    best_nums = None

    def try_update(nums):
        nonlocal best_count, best_nums
        c = count_primes_fast(nums)
        if c < best_count:
            best_count = c
            best_nums = nums[:]
        return c

    # Strategy 1: All multiples of a smooth base B
    # If all nums are multiples of B, all sums are multiples of B.
    # Prime factors of B are always present. Extra primes come from quotient sums.
    # Try bases that are products of small primes.
    
    # Generate smooth bases (5-smooth, 7-smooth, etc.)
    smooth_bases = set()
    # 2^a * 3^b * 5^c * 7^d * 11^e
    for a in range(0, 14):
        v2 = 1 << a
        if v2 > 10000: break
        for b in range(0, 10):
            v = v2 * (3 ** b)
            if v > 10000: break
            for c in range(0, 7):
                v2_ = v * (5 ** c)
                if v2_ > 10000: break
                for d in range(0, 5):
                    v3 = v2_ * (7 ** d)
                    if v3 > 10000: break
                    smooth_bases.add(v3)
    
    smooth_bases = sorted(smooth_bases)

    for base in smooth_bases:
        if time.time() > deadline:
            break
        max_q = 10000 // base
        if max_q < 12:
            continue
        # Try consecutive quotients
        for start_q in range(1, min(max_q - 11, 20) + 1):
            if time.time() > deadline:
                break
            qs = list(range(start_q, start_q + 12))
            nums = [q * base for q in qs]
            try_update(nums)
        # Try a few random subsets
        for _ in range(10):
            if time.time() > deadline:
                break
            qs = random.sample(range(1, max_q + 1), 12)
            nums = [q * base for q in qs]
            try_update(nums)

    # Strategy 2: Arithmetic progressions with smooth difference
    # nums = a + i*d for i=0..11, all <= 10000
    # Sums: 2a + (i+j)*d for 0<=i<j<=11
    # If d is smooth and a is chosen well, sums may be smooth
    for d in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 28, 30, 32, 35, 36, 40, 42, 45, 48, 50, 54, 56, 60, 63, 64, 70, 72, 75, 80, 84, 90, 96, 100, 105, 108, 112, 120, 126, 128, 135, 140, 144, 150, 160, 168, 180, 192, 200, 210, 216, 224, 240, 252, 256, 270, 280, 288, 300, 315, 320, 336, 360, 375, 384, 400, 420, 432, 448, 450, 480, 500, 504, 512, 540, 560, 576, 600, 630, 640, 672, 700, 720, 750, 768, 800, 840, 864, 900, 960, 1000, 1008, 1024, 1050, 1080, 1120, 1152, 1200, 1260, 1280, 1344, 1400, 1440, 1500, 1536, 1600, 1680, 1728, 1800, 1920, 2000, 2016, 2048, 2100, 2160, 2240, 2304, 2400, 2520, 2560, 2688, 2800, 2880, 3000, 3072, 3200, 3360, 3456, 3600, 3840, 4000, 4032, 4096, 4200, 4320, 4480, 4608, 4800, 5040, 5120, 5376, 5600, 5760, 6000, 6144, 6400, 6720, 6912, 7200, 7680, 8000, 8064, 8192, 8400, 8640, 8960, 9216, 9600, 10000]:
        if time.time() > deadline:
            break
        max_a = 10000 - 11 * d
        if max_a < 1:
            continue
        # Try a few values of a
        a_candidates = set()
        for a in range(1, min(max_a + 1, 50) + 1):
            a_candidates.add(a)
        # Also try a values that make 2a+d, 2a+2d, ... smooth
        for _ in range(20):
            if max_a >= 1:
                a_candidates.add(random.randint(1, max_a))
        
        for a in a_candidates:
            if time.time() > deadline:
                break
            if a < 1 or a + 11*d > 10000:
                continue
            nums = [a + i*d for i in range(12)]
            try_update(nums)

    # Strategy 3: All numbers congruent to r mod M for smooth M
    # If all nums ≡ r (mod M), then sums ≡ 2r (mod M)
    # If M divides 2r, then M divides all sums
    # Choose M = 2^a * 3^b * 5^c * 7^d, r = M/2 (so 2r = M)
    # Then all sums are divisible by M
    for M in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48, 50, 54, 56, 60, 64, 70, 72, 80, 84, 90, 96, 100, 108, 112, 120, 128, 140, 144, 150, 160, 168, 180, 192, 200, 210, 216, 224, 240, 250, 252, 256, 270, 280, 288, 300, 320, 336, 350, 360, 375, 384, 400, 420, 432, 448, 450, 480, 500, 504, 512, 540, 560, 576, 600, 630, 640, 672, 700, 720, 750, 768, 800, 840, 864, 900, 960, 1000, 1008, 1024, 1050, 1080, 1120, 1152, 1200, 1260, 1280, 1344, 1400, 1440, 1500, 1536, 1600, 1680, 1728, 1800, 1920, 2000, 2016, 2048, 2100, 2160, 2240, 2304, 2400, 2520, 2560, 2688, 2800, 2880, 3000, 3072, 3200, 3360, 3456, 3600, 3840, 4000, 4032, 4096, 4200, 4320, 4480, 4608, 4800, 5040, 5120, 5376, 5600, 5760, 6000, 6144, 6400, 6720, 6912, 7200, 7680, 8000, 8064, 8192, 8400, 8640, 8960, 9216, 9600, 10000]:
        if time.time() > deadline:
            break
        r = M // 2  # so 2r = M
        if r < 1:
            continue
        # nums = r + k*M for k=0,1,...,11
        last = r + 11 * M
        if last > 10000:
            continue
        nums = [r + k*M for k in range(12)]
        try_update(nums)
        
        # Also try r + offset*M patterns with different starting k
        for start_k in range(1, min(5, (10000 - r) // M - 11 + 1)):
            if time.time() > deadline:
                break
            nums = [r + (start_k + k)*M for k in range(12)]
            if nums[11] > 10000:
                break
            try_update(nums)

    # Strategy 4: Local improvement with incremental counting
    if best_nums is not None and time.time() < deadline - 10:
        nums = best_nums[:]
        c = best_count
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                # Try larger deltas first for more impactful changes
                for delta in range(-50, 51):
                    if delta == 0:
                        continue
                    new_val = nums[i] + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in nums:
                        continue
                    old = nums[i]
                    nums[i] = new_val
                    delta_count, _ = incremental_count_change(nums, i, old, new_val)
                    c2 = c + delta_count
                    if c2 < c:
                        c = c2
                        if c < best_count:
                            best_count = c
                            best_nums = nums[:]
                        improved = True
                    else:
                        nums[i] = old
                # Also try swapping with another element
                for j in range(i+1, 12):
                    if time.time() > deadline:
                        break
                    old_i, old_j = nums[i], nums[j]
                    nums[i], nums[j] = old_j, old_i
                    c2 = count_primes_fast(nums)
                    if c2 < c:
                        c = c2
                        if c < best_count:
                            best_count = c
                            best_nums = nums[:]
                        improved = True
                    else:
                        nums[i], nums[j] = old_i, old_j

    # Strategy 5: Random restarts with local improvement using incremental counting
    if time.time() < deadline - 15:
        for attempt in range(300):
            if time.time() > deadline - 5:
                break
            # Generate random structured candidate
            mode = random.randint(0, 3)
            if mode == 0:
                base = random.choice([2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,28,30,32,35,36,40,42,45,48,50,54,56,60,63,64,70,72,75,80,84,90,96,100,105,108,112,120,126,128,135,140,144,150,160,168,180,192,200,210,216,224,240,252,256,270,280,288,300,315,320,336,360,375,384,400,420,432,448,450,480,500,504,512,540,560,576,600,630,640,672,700,720,750,768,800,840,864,900,960,1000,1008,1024,1050,1080,1120,1152,1200,1260,1280,1344,1400,1440,1500,1536,1600,1680,1728,1800,1920,2000,2016,2048,2100,2160,2240,2304,2400,2520,2560,2688,2800,2880,3000,3072,3200,3360,3456,3600,3840,4000,4032,4096,4200,4320,4480,4608,4800,5040,5120,5376,5600,5760,6000,6144,6400,6720,6912,7200,7680,8000,8064,8192,8400,8640,8960,9216,9600,10000])
                max_q = 10000 // base
                if max_q < 12:
                    continue
                qs = random.sample(range(1, max_q + 1), 12)
                nums = [q * base for q in qs]
            elif mode == 1:
                d = random.choice([2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,28,30,32,35,36,40,42,45,48,50,54,56,60,63,64,70,72,75,80,84,90,96,100,105,108,112,120,126,128,135,140,144,150,160,168,180,192,200,210,216,224,240,252,256,270,280,288,300,315,320,336,360,375,384,400,420,432,448,450,480,500,504,512,540,560,576,600,630,640,672,700,720,750,768,800,840,864,900,960,1000,1008,1024,1050,1080,1120,1152,1200,1260,1280,1344,1400,1440,1500,1536,1600,1680,1728,1800,1920,2000,2016,2048,2100,2160,2240,2304,2400,2520,2560,2688,2800,2880,3000,3072,3200,3360,3456,3600,3840,4000,4032,4096,4200,4320,4480,4608,4800,5040,5120,5376,5600,5760,6000,6144,6400,6720,6912,7200,7680,8000,8064,8192,8400,8640,8960,9216,9600,10000])
                max_a = 10000 - 11 * d
                if max_a < 1:
                    continue
                a = random.randint(1, max_a)
                nums = [a + i*d for i in range(12)]
            elif mode == 2:
                nums = random.sample(range(1, 10001), 12)
            else:
                # New mode: all numbers are multiples of a small smooth number
                # and chosen to make sums have few prime factors
                base = random.choice([6, 12, 24, 30, 42, 60, 84, 120, 180, 240, 360, 420, 600, 840, 1200, 1680, 2400, 4200])
                max_q = 10000 // base
                if max_q < 12:
                    continue
                # Try to pick quotients that are also smooth
                smooth_qs = []
                for q in range(1, max_q + 1):
                    # Check if q is 5-smooth (only factors 2,3,5)
                    tmp = q
                    while tmp % 2 == 0:
                        tmp //= 2
                    while tmp % 3 == 0:
                        tmp //= 3
                    while tmp % 5 == 0:
                        tmp //= 5
                    if tmp == 1 and len(smooth_qs) < 200:
                        smooth_qs.append(q)
                if len(smooth_qs) >= 12:
                    qs = random.sample(smooth_qs, 12)
                else:
                    qs = random.sample(range(1, max_q + 1), 12)
                nums = [q * base for q in qs]
            
            c = count_primes_fast(nums)
            if c <= best_count + 3:
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for i in range(12):
                        if time.time() > deadline - 5:
                            break
                        for delta in range(-20, 21):
                            if delta == 0:
                                continue
                            new_val = nums[i] + delta
                            if new_val < 1 or new_val > 10000:
                                continue
                            if new_val in nums:
                                continue
                            old = nums[i]
                            nums[i] = new_val
                            delta_count, _ = incremental_count_change(nums, i, old, new_val)
                            c2 = c + delta_count
                            if c2 < c:
                                c = c2
                                if c < best_count:
                                    best_count = c
                                    best_nums = nums[:]
                                improved = True
                            else:
                                nums[i] = old

    if best_nums is None:
        return list(range(1, 13))
    
    # Final polish: try to reduce further with targeted swaps
    if time.time() < deadline - 2:
        nums = best_nums[:]
        c = best_count
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            improved = False
            for i in range(12):
                for j in range(i+1, 12):
                    if time.time() > deadline - 1:
                        break
                    old_i, old_j = nums[i], nums[j]
                    # Try swapping
                    nums[i], nums[j] = old_j, old_i
                    c2 = count_primes_fast(nums)
                    if c2 < c:
                        c = c2
                        if c < best_count:
                            best_count = c
                            best_nums = nums[:]
                        improved = True
                    else:
                        nums[i], nums[j] = old_i, old_j
                    # Try adding/subtracting small amounts to one element
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_val = nums[i] + delta
                        if new_val < 1 or new_val > 10000 or new_val in nums:
                            continue
                        old = nums[i]
                        nums[i] = new_val
                        c3 = count_primes_fast(nums)
                        if c3 < c:
                            c = c3
                            if c < best_count:
                                best_count = c
                                best_nums = nums[:]
                            improved = True
                        else:
                            nums[i] = old
            if not improved:
                break
    
    return best_nums
# EVOLVE-BLOCK-END
