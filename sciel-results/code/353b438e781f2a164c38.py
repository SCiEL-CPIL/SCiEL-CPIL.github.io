import time
import random
import math
from collections import Counter

def _init_primes():
    """Precompute small primes up to 200 for fast trial division."""
    is_prime = [True] * 200
    is_prime[0] = is_prime[1] = False
    for i in range(2, 200):
        if is_prime[i]:
            for j in range(i * i, 200, i):
                is_prime[j] = False
    return [i for i in range(2, 200) if is_prime[i]]

_SMALL_PRIMES = _init_primes()
_prime_cache = {}

def get_primes(n):
    """Return the set of prime factors of n."""
    if n == 1:
        return set()
    primes = set()
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            primes.add(p)
            n //= p
    if n > 1:
        primes.add(n)
    return primes

def get_primes_cached(n):
    """Return the set of prime factors of n with caching."""
    if n in _prime_cache:
        return _prime_cache[n]
    result = get_primes(n)
    _prime_cache[n] = result
    return result

def evaluate(set_a):
    """Calculate the union of prime factors of all pairwise sums."""
    all_primes = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            s = set_a[i] + set_a[j]
            all_primes.update(get_primes_cached(s))
    return all_primes

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    def time_left():
        return deadline - time.time()

    best_set = None
    best_primes = None
    best_count = float('inf')
    
    candidates = []
    
    # Strategy 1: Residue classes modulo 2*p for small primes p
    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        mod = 2 * p
        res = p
        nums = []
        k = 0
        while len(nums) < 12:
            val = mod * k + res
            if val < 1:
                k += 1
                continue
            if val > 10000:
                break
            if val not in nums:
                nums.append(val)
            k += 1
        if len(nums) == 12:
            candidates.append(nums)
            
    # Strategy 2: Small multiples of highly composite numbers
    for base in [6, 12, 30, 60, 120, 210]:
        nums = []
        k = 1
        while len(nums) < 12:
            val = base * k
            if val > 10000:
                break
            nums.append(val)
            k += 1
        if len(nums) == 12:
            candidates.append(nums)

    # Strategy 3: Powers of 2
    candidates.append([2**i for i in range(1, 13)])
    
    # Strategy 4: Small integers
    candidates.append(list(range(1, 13)))
    candidates.append(list(range(2, 14)))
    
    # Strategy 5: Random
    for _ in range(5):
        candidates.append([random.randint(1, 10000) for _ in range(12)])

    # Evaluate all candidates
    for c in candidates:
        if time_left() < 5:
            break
        c = [max(1, min(10000, x)) for x in c]
        if len(set(c)) < 12:
            existing = set(c)
            while len(c) < 12:
                v = random.randint(1, 10000)
                if v not in existing:
                    c.append(v)
                    existing.add(v)
        c = c[:12]
        
        primes = evaluate(c)
        if len(primes) < best_count:
            best_count = len(primes)
            best_primes = primes
            best_set = c[:]
    
    if best_set is None:
        return list(range(1, 13))
    
    # Local Search
    current_set = best_set[:]
    current_primes = set(best_primes)
    current_count = best_count
    
    small_deltas = list(range(-20, 21))
    med_deltas = list(range(-100, 101, 5))
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time_left() > 5:
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        r = random.random()
        if r < 0.4:
            new_val = old_val + random.choice(small_deltas)
        elif r < 0.7:
            new_val = old_val + random.choice(med_deltas)
        elif r < 0.9:
            new_val = random.randint(1, 10000)
        else:
            other_idx = random.randint(0, 11)
            if other_idx == idx:
                other_idx = (idx + 1) % 12
            base = current_set[other_idx]
            step = random.choice([2, 3, 5, 6, 10, 30])
            new_val = base + random.randint(-10, 10) * step
            
        if new_val < 1 or new_val > 10000:
            continue
        if new_val == old_val:
            continue
            
        if new_val in current_set and current_set.index(new_val) != idx:
            continue
        
        new_set = current_set[:]
        new_set[idx] = new_val
        
        new_primes = evaluate(new_set)
        new_count = len(new_primes)
        
        if new_count < current_count:
            current_set = new_set
            current_primes = new_primes
            current_count = new_count
            no_improve_count = 0
            if current_count < best_count:
                best_count = current_count
                best_primes = current_primes
                best_set = current_set[:]
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                no_improve_count = 0
                restart_set = [random.randint(1, 10000) for _ in range(12)]
                while len(set(restart_set)) < 12:
                    restart_set.append(random.randint(1, 10000))
                restart_set = restart_set[:12]
                restart_primes = evaluate(restart_set)
                if len(restart_primes) < current_count:
                    current_set = restart_set
                    current_primes = restart_primes
                    current_count = len(restart_primes)
                    if current_count < best_count:
                        best_count = current_count
                        best_primes = current_primes
                        best_set = current_set[:]

    return best_set
