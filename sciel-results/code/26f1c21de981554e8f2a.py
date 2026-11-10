import itertools
import time

def _factor_primes(n):
    primes = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            primes.add(d)
            n //= d
        d += 1
    if n > 1:
        primes.add(n)
    return primes

def _is_smooth(n, primes):
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    start_time = time.time()
    best_set = None
    best_count = float('inf')
    
    def evaluate(nums):
        nonlocal best_set, best_count
        all_primes = set()
        for i in range(12):
            for j in range(i + 1, 12):
                all_primes |= _factor_primes(nums[i] + nums[j])
        count = len(all_primes)
        if count < best_count:
            best_count = count
            best_set = list(nums)
    
    # Try smooth quotient approach with multiple prime sets
    for prime_set in [[2,3,5], [2,3,5,7], [2,3,5,7,11]]:
        base = 1
        for p in prime_set:
            base *= p
        if base > 800:  # need at least 12 multiples <= 10000
            continue
        
        max_q = 10000 // base
        if max_q < 12:
            continue
        
        # Generate smooth quotients
        smooth_q = [q for q in range(1, min(max_q, 500) + 1) 
                    if _is_smooth(q, prime_set)]
        
        # Try random subsets with local improvement
        import random
        for attempt in range(1000):
            if time.time() - start_time > 40:
                break
            # Random start
            selected = random.sample(smooth_q, min(12, len(smooth_q)))
            # Local search: try to fix conflicts
            for _ in range(100):
                if time.time() - start_time > 40:
                    break
                # Check all pairwise
                conflicts = []
                for i in range(12):
                    for j in range(i + 1, 12):
                        if not _is_smooth(selected[i] + selected[j], prime_set):
                            conflicts.append((i, j))
                if not conflicts:
                    nums = [base * q for q in selected]
                    if max(nums) <= 10000:
                        evaluate(nums)
                    break
                # Pick a conflict, replace one element
                i, j = random.choice(conflicts)
                pos = random.choice([i, j])
                # Try replacing selected[pos] with a random smooth q
                new_q = random.choice(smooth_q)
                if new_q not in selected:
                    selected[pos] = new_q
        
        if time.time() - start_time > 40:
            break
    
    if time.time() - start_time > 50:
        if best_set:
            return best_set
        return list(range(1, 13))
    
    # Also try some structured sets
    structured_sets = [
        [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144],
        [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10000],
    ]
    for s in structured_sets:
        if max(s) <= 10000 and len(set(s)) == 12:
            evaluate(s)
    
    if best_set:
        return best_set
    return list(range(1, 13))
