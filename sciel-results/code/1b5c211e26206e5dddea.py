import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin

    # 1. Generate 3-smooth numbers (Hamming numbers with primes 2, 3) up to 1666
    # x_i = 6 * a_i <= 10000 => a_i <= 1666
    max_a = 1666
    smooth_numbers = set()
    p2 = 1
    while p2 <= max_a:
        p3 = 1
        while p2 * p3 <= max_a:
            smooth_numbers.add(p2 * p3)
            p3 *= 3
        p2 *= 2
    
    candidates = sorted(list(smooth_numbers))
    
    # If we don't have enough, expand to 5-smooth? 
    # But 3-smooth sums are more likely to be 3-smooth.
    # Let's check count.
    if len(candidates) < 12:
        # Fallback: add 5-smooth
        p2 = 1
        while p2 <= max_a:
            p3 = 1
            while p2 * p3 <= max_a:
                p5 = 1
                while p2 * p3 * p5 <= max_a:
                    smooth_numbers.add(p2 * p3 * p5)
                    p5 *= 5
                p3 *= 3
            p2 *= 2
        candidates = sorted(list(smooth_numbers))

    # We want to select 12 distinct a_i from candidates
    # such that the number of distinct prime factors of all (a_i + a_j) is minimized.
    # Note: The final primes will be {2, 3} U primes(a_i+a_j).
    # So we just minimize primes(a_i+a_j).
    
    n_needed = 12
    best_set = None
    best_score = float('inf')
    
    # Precompute factorization for numbers up to 2 * max_a (max sum)
    max_sum_a = 2 * max_a
    spf = list(range(max_sum_a + 1))
    for i in range(2, int(math.isqrt(max_sum_a)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_sum_a + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_primes(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def evaluate(a_set):
        primes = set()
        for i in range(n_needed):
            for j in range(i + 1, n_needed):
                s = a_set[i] + a_set[j]
                primes.update(get_primes(s))
        return len(primes)

    # Initialize with first 12 candidates
    current_set = candidates[:n_needed]
    current_score = evaluate(current_set)
    best_score = current_score
    best_set = current_set.copy()
    
    # Local Search
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        indices = list(range(n_needed))
        random.shuffle(indices)
        for i in indices:
            # Try swapping current_set[i] with a random candidate
            for _ in range(200): # More swaps per position since pool is small
                if (time.time() - start_time) > time_limit:
                    break
                c = random.choice(candidates)
                if c in current_set:
                    continue
                
                old_val = current_set[i]
                current_set[i] = c
                new_score = evaluate(current_set)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current_set.copy()
                else:
                    current_set[i] = old_val
                    
        # If we found a perfect solution (0 extra primes), we can stop
        # 0 extra primes means all sums are 3-smooth.
        # This is very hard, but let's see.
        if best_score == 0:
            break

    # Construct final result
    # x_i = 6 * a_i
    result = [6 * a for a in best_set]
    
    # Verify distinctness and limit
    if len(set(result)) != 12 or max(result) > 10000:
        return list(range(1, 13))
        
    return result
