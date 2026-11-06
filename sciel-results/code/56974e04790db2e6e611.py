import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave margin for overhead
    
    # Precompute Smallest Prime Factor (SPF) up to 20000
    max_sum = 20000
    spf = list(range(max_sum + 1))
    for i in range(2, int(math.isqrt(max_sum)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_sum + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def get_prime_set(numbers):
        """
        numbers: list of 12 distinct integers
        returns: set of distinct prime factors of all pairwise sums
        """
        primes = set()
        n = len(numbers)
        for i in range(n):
            for j in range(i + 1, n):
                s = numbers[i] + numbers[j]
                primes.update(factorize(s))
        return primes

    def evaluate(numbers):
        return len(get_prime_set(numbers))

    # Generate candidate pool: 5-smooth numbers (Hamming numbers) <= 10000
    # 2^a * 3^b * 5^c <= 10000
    pool = set()
    max_val = 10000
    for a in range(0, 15):  # 2^13 = 8192
        pow2 = 2 ** a
        if pow2 > max_val:
            break
        for b in range(0, 10):  # 3^8 = 6561
            pow3 = 3 ** b
            if pow2 * pow3 > max_val:
                break
            for c in range(0, 8):  # 5^6 = 15625
                val = pow2 * pow3 * (5 ** c)
                if val > max_val:
                    break
                pool.add(val)
    
    # If pool is too small, add 7-smooth numbers
    if len(pool) < 100:
        for a in range(0, 15):
            pow2 = 2 ** a
            if pow2 > max_val:
                break
            for b in range(0, 10):
                pow3 = 3 ** b
                if pow2 * pow3 > max_val:
                    break
                for c in range(0, 8):
                    pow5 = 5 ** c
                    if pow2 * pow3 * pow5 > max_val:
                        break
                    for d in range(0, 6): # 7^5 = 16807
                        val = pow2 * pow3 * pow5 * (7 ** d)
                        if val > max_val:
                            break
                        pool.add(val)
    
    candidates = list(pool)
    if len(candidates) < 12:
        # Fallback to all integers if pool is somehow too small (unlikely)
        candidates = list(range(1, max_val + 1))
    
    n_needed = 12
    
    # Initialize with random selection
    best_set = None
    best_score = float('inf')
    
    # Try a few random initializations
    for _ in range(20):
        current_set = random.sample(candidates, n_needed)
        current_score = evaluate(current_set)
        if current_score < best_score:
            best_score = current_score
            best_set = current_set.copy()
            
        # Local Search
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Random order of indices to avoid bias
            indices = list(range(n_needed))
            random.shuffle(indices)
            for i in indices:
                # Try swapping current_set[i] with a random candidate
                # To be efficient, we can just pick random candidates
                for _ in range(50): # Try 50 random swaps per position
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
                        # Update global best
                        if current_score < best_score:
                            best_score = current_score
                            best_set = current_set.copy()
                    else:
                        current_set[i] = old_val
                        
            # If we improved, loop again. If not, break.
            # Actually, the 'improved' flag is set if ANY swap improved.
            # We should restart the inner loop if improved.
            
        if best_score <= 2: # Theoretical lower bound is likely 2 or 3
            break

    # Ensure we return a valid list of 12 distinct integers
    if best_set is None:
        return list(range(1, 13))
        
    return best_set
