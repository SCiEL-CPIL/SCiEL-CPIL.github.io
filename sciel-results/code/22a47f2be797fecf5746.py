import math
from itertools import combinations
import time

def construct():
    """
    Find 12 distinct positive integers <= 10000 that minimize the number of
    distinct prime factors across all pairwise sums of distinct elements.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s kill
    
    n = 12
    max_val = 10000
    
    # Generate primes up to 2*max_val (max possible sum is 19999)
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(2 * max_val)
    prime_set = set(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factor bitmask for each possible sum (2 to 2*max_val)
    # We'll use a dictionary mapping sum -> set of prime factors (as a frozenset or tuple)
    def factorize(x):
        """Return set of prime factors of x"""
        factors = set()
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        return factors
    
    # Precompute factor sets for all possible sums
    sum_factors = {}
    for s in range(2, 2 * max_val + 1):
        sum_factors[s] = factorize(s)
    
    # Objective function: count distinct primes across all pairwise sums
    def objective(arr):
        all_primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                s = arr[i] + arr[j]
                all_primes.update(sum_factors[s])
        return len(all_primes)
    
    # Strategy: Try to find numbers whose pairwise sums use as few distinct primes as possible.
    # Key insight: If we can make all sums divisible by only a small set of primes, we win.
    # Try candidate prime sets of size 2, 3, 4, ...
    # For a given set of primes P, we want all pairwise sums to be P-smooth (only have prime factors in P).
    # Actually, we want the UNION of prime factors of all sums to be as small as possible.
    
    # Let's try a constructive approach:
    # Pick a small set of primes P = {p1, p2, ...}
    # Find numbers such that for any two distinct a, b in our set, a+b has only prime factors in P.
    
    # For |P| = 2: sums must be of form 2^a * 3^b (if P={2,3}) or similar
    # For |P| = 3: sums can have 3 primes
    
    # Let's try P = {2, 3} first
    # Numbers a, b such that a+b is 2^x * 3^y
    # Try to build a set of 12 numbers where all pairwise sums are 2^x * 3^y
    
    # Alternative: use a local search / simulated annealing approach
    # Start with a random set and try to improve it
    
    def random_initial_set():
        import random
        return random.sample(range(1, max_val + 1), n)
    
    # Better: structured initial sets
    # Try sets where numbers are all multiples of a common number
    # Or numbers clustered together
    
    best_set = None
    best_score = float('inf')
    
    # Try some structured initial sets
    structured_sets = []
    
    # Set 1: powers of 2 scaled
    for base in [1, 2, 3, 5]:
        s = [base * (2**i) for i in range(n)]
        if max(s) <= max_val and len(set(s)) == n:
            structured_sets.append(s)
    
    # Set 2: consecutive numbers
    for start in range(1, 20):
        s = list(range(start, start + n))
        if max(s) <= max_val:
            structured_sets.append(s)
    
    # Set 3: multiples
    for m in [2, 3, 4, 5, 10]:
        s = [m * (i + 1) for i in range(n)]
        if max(s) <= max_val:
            structured_sets.append(s)
    
    # Evaluate structured sets
    for s in structured_sets:
        score = objective(s)
        if score < best_score:
            best_score = score
            best_set = s[:]
    
    # Local search to improve
    def local_search(initial_set, time_budget):
        import random
        current = initial_set[:]
        current_score = objective(current)
        best_local = current[:]
        best_local_score = current_score
        temp = 100.0
        cooling = 0.995
        max_iters = 100000
        
        for it in range(max_iters):
            if time.time() > time_budget:
                break
            
            # Pick a random index and a new random value
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            new_val = random.randint(1, max_val)
            
            # Ensure distinctness
            if new_val in current and current.index(new_val) != idx:
                continue
            
            current[idx] = new_val
            new_score = objective(current)
            
            delta = new_score - current_score
            if delta < 0 or (temp > 0.01 and random.random() < math.exp(-delta / temp)):
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local = current[:]
            else:
                current[idx] = old_val
            
            temp *= cooling
        
        return best_local, best_local_score
    
    # Run local search from multiple starts
    import random
    random.seed(42)
    
    # First, run from the best structured set
    local_set, local_score = local_search(best_set, deadline - 2)
    if local_score < best_score:
        best_score = local_score
        best_set = local_set[:]
    
    # Run from several random starts
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        rand_set = random_initial_set()
        local_set, local_score = local_search(rand_set, deadline - 2)
        if local_score < best_score:
            best_score = local_score
            best_set = local_set[:]
    
    # Final polish: try swapping elements
    if time.time() < deadline - 2:
        for _ in range(5000):
            if time.time() > deadline - 1:
                break
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            if i == j:
                continue
            best_set[i], best_set[j] = best_set[j], best_set[i]
            score = objective(best_set)
            if score < best_score:
                best_score = score
            else:
                best_set[i], best_set[j] = best_set[j], best_set[i]
    
    return best_set

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Distinct: {len(set(result)) == 12}")
    print(f"Max: {max(result)}")
    
    # Count prime factors
    def factorize(x):
        factors = set()
        temp = x
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
            p += 1
        if temp > 1:
            factors.add(temp)
        return factors
    
    all_primes = set()
    for i in range(12):
        for j in range(i + 1, 12):
            s = result[i] + result[j]
            all_primes.update(factorize(s))
    print(f"Distinct primes: {len(all_primes)}")
    print(f"Primes: {sorted(all_primes)}")
