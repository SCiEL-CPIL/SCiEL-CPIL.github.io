import time
import random
from itertools import combinations

def _prime_factors(n):
    """Return the set of distinct prime factors of n."""
    factors = set()
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            factors.add(f)
            while n % f == 0:
                n //= f
        f += 2
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(nums):
    """Count distinct prime factors across all pairwise sums."""
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            all_primes |= _prime_factors(s)
    return len(all_primes)

def _local_search(nums, deadline):
    """Perform local search to minimize distinct prime factors."""
    best = list(nums)
    best_score = _count_distinct_primes(best)
    
    n = len(best)
    max_val = 10000
    
    # Try to improve by replacing elements
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n):
            # Try replacing best[i] with a nearby value
            old_val = best[i]
            # Generate candidate replacements
            candidates = set()
            # Nearby values
            for d in range(1, 50):
                for v in [old_val + d, old_val - d]:
                    if 1 <= v <= max_val and v not in best:
                        candidates.add(v)
            # Also try some random values
            for _ in range(20):
                v = random.randint(1, max_val)
                if v not in best:
                    candidates.add(v)
            
            for v in candidates:
                trial = list(best)
                trial[i] = v
                score = _count_distinct_primes(trial)
                if score < best_score:
                    best = trial
                    best_score = score
                    improved = True
                    break
                elif score == best_score:
                    # Accept equal-score moves to escape local minima (simulated annealing-like)
                    if random.random() < 0.1:
                        best = trial
                        improved = True
                        break
    
    return best, best_score

def _construct_with_prime_target(target_primes, deadline):
    """Try to find 12 distinct integers whose pairwise sums use only target_primes."""
    # Target: all pairwise sums should be divisible by only primes in target_primes
    # Strategy: pick numbers such that a_i + a_j is composed only of primes in target_primes
    
    n = 12
    max_val = 10000
    
    # Generate candidate numbers: numbers whose pairwise sums with other candidates
    # only involve primes in target_primes
    # This is complex, so use a heuristic: start with multiples of small primes
    
    # Try: all numbers are multiples of the product of target_primes
    prod = 1
    for p in target_primes:
        prod *= p
    
    # Numbers that are multiples of prod will have sums that are multiples of prod
    # But we want the sums to have NO prime factors outside target_primes
    # If a_i = k_i * prod, then a_i + a_j = (k_i + k_j) * prod
    # We need (k_i + k_j) * prod to have no prime factors outside target_primes
    # Since prod already contains all target_primes, we need (k_i + k_j) to have no prime factors outside target_primes
    
    # So we need k_i + k_j to be a "target-prime-smooth" number
    # Let's try k_i = 1, 2, 3, ... and check
    
    best = None
    best_score = float('inf')
    
    # Try small k values
    k_values = list(range(1, 100))
    random.shuffle(k_values)
    
    for start in range(min(20, len(k_values))):
        if time.time() > deadline - 3:
            break
        # Greedily build the set
        chosen_k = []
        for k in k_values[start:start+50]:
            if len(chosen_k) >= n:
                break
            nums = [k * prod]
            for existing_k in chosen_k:
                s = (k + existing_k) * prod
                pf = _prime_factors(s)
                if pf - set(target_primes):
                    nums = None
                    break
            if nums is not None:
                chosen_k.append(k)
        
        if len(chosen_k) >= n:
            nums = [k * prod for k in chosen_k[:n]]
            if all(1 <= x <= max_val for x in nums) and len(set(nums)) == n:
                score = _count_distinct_primes(nums)
                if score < best_score:
                    best_score = score
                    best = nums
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 12
    max_val = 10000
    
    best_result = list(range(1, 13))
    best_score = _count_distinct_primes(best_result)
    
    # Strategy 1: Try target prime sets
    # Common small prime sets
    prime_sets = [
        [2],
        [2, 3],
        [2, 5],
        [2, 7],
        [3, 5],
        [2, 3, 5],
        [2, 3, 7],
        [2, 5, 7],
        [3, 5, 7],
        [2, 3, 5, 7],
    ]
    
    for primes in prime_sets:
        if time.time() > deadline - 10:
            break
        result, score = _construct_with_prime_target(primes, deadline)
        if result is not None:
            if score < best_score:
                best_score = score
                best_result = result
    
    # Strategy 2: Randomized local search from various starting points
    random.seed(42)
    
    # Starting point: all even numbers
    start_even = list(range(2, 2*n+1, 2))
    if len(start_even) == n:
        result, score = _local_search(start_even, deadline)
        if score < best_score:
            best_score = score
            best_result = result
    
    # Starting point: multiples of 6
    start_mult6 = [i*6 for i in range(1, n+1)]
    if all(x <= max_val for x in start_mult6):
        result, score = _local_search(start_mult6, deadline)
        if score < best_score:
            best_score = score
            best_result = result
    
    # Random restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # Generate a random starting set
        nums = random.sample(range(1, max_val+1), n)
        result, score = _local_search(nums, deadline)
        if score < best_score:
            best_score = score
            best_result = result
        if restarts >= 30:
            break
    
    # Final polish: try to improve the best result
    if time.time() < deadline - 2:
        result, score = _local_search(best_result, deadline)
        if score < best_score:
            best_score = score
            best_result = result
    
    return best_result
