import time
import random
import math

MAX_VAL = 10000
N = 12
DEADLINE = 85  # safety margin from 90s wall-clock

def construct():
    t0 = time.time()
    
    # Build smallest prime factor table up to 2*MAX_VAL
    limit = 2 * MAX_VAL
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def prime_factors(n):
        """Return set of prime factors of n using SPF table."""
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def evaluate(arr):
        """Count distinct prime factors across all pairwise sums."""
        all_primes = set()
        for i in range(N):
            for j in range(i + 1, N):
                s = arr[i] + arr[j]
                all_primes |= prime_factors(s)
        return len(all_primes)
    
    def evaluate_delta(arr, idx, old_val, new_val):
        """Compute change in distinct prime count when arr[idx] changes from old_val to new_val.
        Returns (new_count, new_primes_set) or None if same count (we'll just recompute)."""
        # For efficiency, just do full recompute for simplicity; with N=12 it's 66 sums
        arr[idx] = new_val
        cnt = evaluate(arr)
        arr[idx] = old_val
        return cnt
    
    # Targeted initializations
    def gen_even():
        random.shuffle([x for x in range(2, 2 * N + 1)])
        return [2 * (i + 1) for i in range(N)]
    
    def gen_multiples_of_6():
        return [6 * (i + 1) for i in range(N)]
    
    def gen_multiples_of_3():
        return [3 * (i + 1) for i in range(N)]
    
    def gen_random():
        return random.sample(range(1, MAX_VAL + 1), N)
    
    def gen_smooth_biased():
        # Prefer numbers whose sums with each other tend to be smooth
        # Use numbers that are multiples of small primes
        candidates = [x for x in range(1, MAX_VAL + 1) if x % 6 == 0 or x % 10 == 0 or x % 15 == 0]
        random.shuffle(candidates)
        return candidates[:N]
    
    initializers = [gen_even, gen_multiples_of_6, gen_multiples_of_3, gen_random, gen_smooth_biased,
                    lambda: [x * 2 for x in random.sample(range(1, 1000), N)],
                    lambda: [x * 3 for x in random.sample(range(1, 700), N)]]
    
    best_arr = None
    best_score = float('inf')
    
    # Try multiple initializations
    for init_fn in initializers:
        if time.time() - t0 > DEADLINE:
            break
        arr = init_fn()
        # Ensure distinct and <= MAX_VAL
        if len(set(arr)) != N or any(x < 1 or x > MAX_VAL for x in arr):
            continue
        score = evaluate(arr)
        if score < best_score:
            best_score = score
            best_arr = arr[:]
    
    if best_arr is None:
        best_arr = list(range(1, 13))
        best_score = evaluate(best_arr)
    
    # Local search: swap one element with another value
    # Candidate pool: mix of random and structured candidates
    def local_search(arr, max_time):
        nonlocal best_arr, best_score
        current = arr[:]
        current_score = best_score
        # Precompute all pairwise sums' prime factors for current
        no_improve = 0
        max_no_improve = 5000
        pool_size = 500
        # Build candidate pool
        pool = random.sample(range(1, MAX_VAL + 1), pool_size)
        # Also add structured candidates
        for m in [2, 3, 5, 6, 10, 15, 30]:
            for k in range(1, MAX_VAL // m + 1):
                pool.append(m * k)
        random.shuffle(pool)
        
        while time.time() - t0 < max_time and no_improve < max_no_improve:
            idx = random.randint(0, N - 1)
            old_val = current[idx]
            # Try several candidates
            improved = False
            for _ in range(20):
                cand = random.choice(pool)
                if cand == old_val or cand in current:
                    continue
                current[idx] = cand
                new_score = evaluate(current)
                if new_score < current_score:
                    current_score = new_score
                    no_improve = 0
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_arr = current[:]
                    break
                else:
                    current[idx] = old_val
                    no_improve += 1
            if not improved:
                # Occasionally mutate pool
                for _ in range(10):
                    pool[random.randint(0, len(pool) - 1)] = random.randint(1, MAX_VAL)
        
        return current, current_score
    
    # Run local search from best initialization
    if time.time() - t0 < DEADLINE:
        current, cur_score = local_search(best_arr, DEADLINE - 5)
    
    # Final check
    if time.time() - t0 > DEADLINE - 1:
        return best_arr
    
    return best_arr
