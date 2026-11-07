import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer
    
    # Strategy:
    # 1. Generate a set of "seed" k values (small, simple forms).
    # 2. Compute v_i = k_i * sigma(k_i) for each seed.
    # 3. Let M = LCM(v_1, ..., v_n).
    # 4. Verify M <= 10^18.
    # 5. Perform a full scan of k from 1 to isqrt(M) to find all k such that k*sigma(k) == M.
    # 6. Return the best M found.
    
    # To make this efficient, we need a fast sigma function for the scan.
    # We will use a sieve for sigma up to some limit, but M can be large.
    # If M is around 10^16, sqrt(M) is 10^8. A sieve of 10^8 is ~800MB (int32) or 1.6GB (int64).
    # This might be too much memory or time in Python.
    # However, we can limit the search space for M.
    
    # Let's try to keep M <= 10^14 so sqrt(M) <= 10^7.
    # A sieve of 10^7 is ~40MB (int32). Very fast.
    
    # What seeds k_i should we use?
    # We want v_i to be small so their LCM is small.
    # But we also want the resulting M to have many preimages.
    # Often, highly composite numbers or numbers with many factors have many preimages.
    
    # Let's pick a set of small k's that produce distinct v's.
    # k=1 -> v=1
    # k=2 -> v=2*3=6
    # k=3 -> v=3*4=12
    # k=4 -> v=4*7=28
    # k=5 -> v=5*6=30
    # k=6 -> v=6*12=72
    # k=7 -> v=7*8=56
    # k=8 -> v=8*15=120
    # k=9 -> v=9*13=117
    # k=10 -> v=10*18=180
    
    # LCM of {1, 6, 12, 28, 30, 72, 56, 120, 117, 180}
    # This M will be a multiple of all these.
    # The number of solutions to k*sigma(k)=M might be larger than for a random M.
    
    # Let's try a few different sets of seeds and pick the one with the most solutions.
    
    seed_sets = [
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60], # Multiples of 6
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], # Multiples of 10
        [2, 4, 8, 16, 32, 64, 128, 256], # Powers of 2
        [3, 6, 9, 12, 15, 18, 21, 24], # Multiples of 3
    ]
    
    best_M = None
    best_ks = []
    
    # Helper to compute sigma for a single n (trial division) for seed generation
    def sigma_single(n):
        if n == 1:
            return 1
        res = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                term = 1
                while temp % p == 0:
                    temp //= p
                    term *= p
                res *= (term * (p + 1) // (p - 1))
            p += 1
        if temp > 1:
            res *= (temp + 1)
        return res
    
    # Helper to compute LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    def lcm_list(lst):
        if not lst:
            return 1
        res = lst[0]
        for x in lst[1:]:
            res = lcm(res, x)
            if res > 10**18:
                return res
        return res

    # Pre-compute sigma for seeds
    seed_vals = {}
    for s_set in seed_sets:
        for k in s_set:
            if k not in seed_vals:
                seed_vals[k] = k * sigma_single(k)
                
    for s_set in seed_sets:
        if time.time() > deadline:
            break
            
        vs = [seed_vals[k] for k in s_set]
        M = lcm_list(vs)
        
        if M > 10**18:
            continue
            
        # Check if M is small enough for a sieve scan
        # sqrt(M) limit for sieve. Let's say we can handle sqrt(M) up to 5*10^6 in ~10s?
        # Sieve of 5*10^6 takes ~0.5s in Python.
        # Let's be conservative: sqrt(M) <= 2*10^6 (M <= 4*10^12)
        # If M is larger, we might skip or use a slower method.
        # But small M is better for finding *more* collisions? 
        # Actually, larger M allows larger k, but the density of solutions might drop.
        # Let's stick to M <= 10^14 (sqrt <= 10^7).
        
        if M > 10**14:
            continue
            
        sqrt_M = math.isqrt(M)
        
        # Sieve sigma up to sqrt_M
        # We need sigma(k) for all k <= sqrt_M
        sigma_arr = [1] * (sqrt_M + 1)
        for i in range(2, sqrt_M + 1):
            for j in range(i, sqrt_M + 1, i):
                sigma_arr[j] += i
                
        # Find all k
        ks = []
        for k in range(1, sqrt_M + 1):
            if k * sigma_arr[k] == M:
                ks.append(k)
                
        if len(ks) > len(best_ks):
            best_ks = ks
            best_M = M
            
    # If we didn't find anything good, fallback to a known good M
    # The seed program found 333312 with 3 solutions.
    # Let's ensure we at least match that.
    if best_M is None or len(best_ks) < 3:
        # Try one more specific construction: M = LCM of v_i for i in 1..50
        # But this M might be huge.
        # Let's just return the best found or the seed.
        if best_M is not None:
            return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
        else:
            return {'M': 333312, 'k': [336, 372, 434]}
            
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}

# EVOLVE-BLOCK-END
