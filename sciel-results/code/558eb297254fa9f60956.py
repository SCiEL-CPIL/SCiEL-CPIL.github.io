import time
import random
from itertools import combinations
from collections import defaultdict

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# Primes up to 20000 (max sum is 10000+10000=20000)
PRIMES = get_primes(20000)
PRIME_SET = set(PRIMES)
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}

def factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            while n % d == 0:
                n //= d
            factors.add(d)
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

# Precompute prime factors for all sums 2..20000
SUM_PRIME_FACTORS = [set() for _ in range(20001)]
for s in range(2, 20001):
    SUM_PRIME_FACTORS[s] = factorize(s)

def count_distinct_primes(nums):
    """Count distinct primes dividing all pairwise sums."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes.update(SUM_PRIME_FACTORS[nums[i] + nums[j]])
    return len(all_primes)

def get_prime_set_for_pair(a, b):
    """Get the set of primes dividing a+b."""
    return SUM_PRIME_FACTORS[a + b]

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Generate candidate pool: focus on small numbers and smooth numbers
    def generate_smooth_numbers(max_val, primes):
        nums = {1}
        for p in primes:
            current = list(nums)
            for x in current:
                y = x
                while y * p <= max_val:
                    y *= p
                    nums.add(y)
        return sorted(nums)
    
    # Use 2,3,5,7-smooth numbers as they produce sums with few prime factors
    smooth_nums = generate_smooth_numbers(10000, [2, 3, 5, 7])
    # Also include small integers 1-200 as they have diverse but manageable sums
    small_ints = list(range(1, 201))
    
    candidate_pool = list(set(smooth_nums + small_ints))
    candidate_pool.sort()
    
    # Strategy: Greedy construction + simulated annealing
    # 1. Build initial set greedily
    # 2. Refine with simulated annealing (swap elements)
    
    best_result = None
    best_score = float('inf')
    
    def greedy_build(pool, target_size=12):
        """Build a set greedily, adding elements that minimize prime factor increase."""
        current = []
        current_primes = set()
        
        # Start with a random pair that has few prime factors
        if len(pool) < 2:
            return pool[:]
        
        # Find the pair with fewest distinct prime factors
        best_pair = None
        best_pair_primes = float('inf')
        # Sample pairs to avoid O(n^2)
        sample_size = min(len(pool), 500)
        indices = random.sample(range(len(pool)), sample_size)
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                a, b = pool[indices[i]], pool[indices[j]]
                pf = SUM_PRIME_FACTORS[a + b]
                if len(pf) < best_pair_primes:
                    best_pair_primes = len(pf)
                    best_pair = (a, b)
        
        if best_pair:
            current = [best_pair[0], best_pair[1]]
            current_primes = set(SUM_PRIME_FACTORS[best_pair[0] + best_pair[1]])
        
        # Greedily add elements
        while len(current) < target_size:
            best_candidate = None
            best_new_primes = float('inf')
            
            for candidate in pool:
                if candidate in current:
                    continue
                
                # Compute new primes if we add this candidate
                new_primes = set(current_primes)
                for elem in current:
                    new_primes.update(SUM_PRIME_FACTORS[candidate + elem])
                
                if len(new_primes) < best_new_primes:
                    best_new_primes = len(new_primes)
                    best_candidate = candidate
                    if best_new_primes <= len(current_primes):
                        break
            
            if best_candidate is None:
                break
            
            current.append(best_candidate)
            for elem in current[:-1]:
                current_primes.update(SUM_PRIME_FACTORS[best_candidate + elem])
        
        return current
    
    def simulated_annealing(initial_set, pool, iterations=5000, start_temp=10.0, end_temp=0.1):
        """Simulated annealing: swap elements to minimize distinct prime count."""
        current = initial_set[:]
        current_score = count_distinct_primes(current)
        best = current[:]
        best_score_local = current_score
        
        current_temp = start_temp
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(iterations, 1))
        
        for it in range(iterations):
            if time.time() > deadline - 2:
                break
            
            # Pick a random element to replace
            idx = random.randint(0, len(current) - 1)
            old_val = current[idx]
            
            # Pick a new candidate from pool
            new_val = random.choice(pool)
            if new_val == old_val or new_val in current:
                continue
            
            # Compute new score
            trial = current[:]
            trial[idx] = new_val
            new_score = count_distinct_primes(trial)
            
            delta = new_score - current_score
            
            if delta <= 0 or random.random() < (2.718281828 ** (-delta / current_temp)):
                current = trial
                current_score = new_score
                
                if new_score < best_score_local:
                    best_score_local = new_score
                    best = trial[:]
            else:
                # Reject: keep current
                pass
            
            current_temp *= cooling_rate
        
        return best, best_score_local
    
    # Multiple restarts with different initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Pure greedy
            initial = greedy_build(candidate_pool, 12)
        elif restart == 1:
            # Start with smallest smooth numbers
            initial = smooth_nums[:12]
        elif restart == 2:
            # Start with smallest integers
            initial = small_ints[:12]
        elif restart == 3:
            # Random sample from smooth numbers
            initial = random.sample(smooth_nums[:200], 12)
        elif restart == 4:
            # Random sample from full pool
            initial = random.sample(candidate_pool[:500], 12)
        elif restart == 5:
            # Greedy with different random seed
            random.seed(restart * 1000)
            initial = greedy_build(candidate_pool, 12)
        elif restart == 6:
            # Start with numbers that are multiples of small primes
            multiples = [x for x in range(1, 1000) if x % 2 == 0 or x % 3 == 0 or x % 5 == 0]
            initial = random.sample(multiples[:300], 12)
        else:
            # Mix of smooth and small
            mixed = smooth_nums[:50] + small_ints[:50]
            initial = random.sample(mixed, 12)
        
        if len(initial) < 12:
            continue
        
        # Refine with simulated annealing
        refined, score = simulated_annealing(initial, candidate_pool[:1000], iterations=3000)
        
        if score < best_score:
            best_score = score
            best_result = refined
        
        # Early stop if we found a very good solution
        if best_score <= 3:
            break
    
    # Final local search on the best result
    if best_result is not None:
        # Do a final intensive local search
        def intensive_local_search(initial_set, pool, max_iters=5000):
            current = initial_set[:]
            current_score = count_distinct_primes(current)
            best = current[:]
            best_score_local = current_score
            
            for it in range(max_iters):
                if time.time() > deadline - 1:
                    break
                
                improved = False
                # Try swapping each element with each pool element (limited)
                for idx in range(len(current)):
                    if improved:
                        break
                    old_val = current[idx]
                    # Sample a few pool elements
                    for new_val in random.sample(pool, min(50, len(pool))):
                        if new_val == old_val or new_val in current:
                            continue
                        
                        trial = current[:]
                        trial[idx] = new_val
                        new_score = count_distinct_primes(trial)
                        
                        if new_score < current_score:
                            current = trial
                            current_score = new_score
                            improved = True
                            
                            if new_score < best_score_local:
                                best_score_local = new_score
                                best = trial[:]
                            break
                
                if not improved:
                    # Try a random perturbation to escape local optimum
                    idx = random.randint(0, len(current) - 1)
                    old_val = current[idx]
                    new_val = random.choice(pool)
                    if new_val != old_val and new_val not in current:
                        trial = current[:]
                        trial[idx] = new_val
                        current = trial
                        current_score = count_distinct_primes(current)
            
            return best, best_score_local
        
        final_result, final_score = intensive_local_search(best_result, candidate_pool[:1000])
        if final_score < best_score:
            best_score = final_score
            best_result = final_result
    
    if best_result is None:
        # Last resort
        best_result = smooth_nums[:12]
    
    return best_result
