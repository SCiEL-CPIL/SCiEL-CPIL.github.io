import time
from math import gcd
import random

def compute_tau_perp(n):
    """Compute the number of adjacent coprime pairs in sorted divisors of n."""
    if n < 2:
        return 0
    
    # Generate all divisors
    divisors = [1]
    temp = n
    i = 2
    while i * i <= temp:
        while temp % i == 0:
            current_len = len(divisors)
            for j in range(current_len):
                divisors.append(divisors[j] * i)
            temp //= i
        i += 1
    if temp > 1:
        current_len = len(divisors)
        for j in range(current_len):
            divisors.append(divisors[j] * temp)
    
    divisors.sort()
    
    # Count adjacent coprime pairs
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Generate a pool of primes to consider
    # We'll use primes up to ~200
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    prime_pool = []
    x = 2
    while len(prime_pool) < 60:
        if is_prime(x):
            prime_pool.append(x)
        x += 1
    
    # Strategy: Try different combinations of primes
    # Start with squarefree products of various prime sets
    best_n = 2
    best_score = compute_tau_perp(2)
    
    # First, try squarefree products of first k primes for k = 10 to 15
    for k in range(10, 16):
        n = 1
        for p in prime_pool[:k]:
            n *= p
        if n <= 10**18:
            score = compute_tau_perp(n)
            if score > best_score:
                best_score = score
                best_n = n
    
    # Now do a more sophisticated search:
    # Represent n by its prime factorization
    # Try local moves: change which primes are included, change exponents
    
    # Start from the best squarefree product
    # Determine which primes are in best_n
    def factorize(n):
        factors = {}
        temp = n
        i = 2
        while i * i <= temp:
            while temp % i == 0:
                factors[i] = factors.get(i, 0) + 1
                temp //= i
            i += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors
    
    current_factors = factorize(best_n)
    current_n = best_n
    current_score = best_score
    
    # Local search with random restarts
    random.seed(42)
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a new prime
        for p in prime_pool:
            if time.time() > deadline:
                break
            if p in current_factors:
                continue
            new_n = current_n * p
            if new_n > 10**18:
                continue
            score = compute_tau_perp(new_n)
            if score > current_score:
                current_score = score
                current_n = new_n
                current_factors[p] = 1
                improved = True
                break
        
        # Try increasing an exponent
        if not improved:
            for p in list(current_factors.keys()):
                if time.time() > deadline:
                    break
                new_e = current_factors[p] + 1
                new_n = current_n * p
                if new_n > 10**18:
                    continue
                score = compute_tau_perp(new_n)
                if score > current_score:
                    current_score = score
                    current_n = new_n
                    current_factors[p] = new_e
                    improved = True
                    break
        
        # Try removing a prime
        if not improved:
            for p in list(current_factors.keys()):
                if time.time() > deadline:
                    break
                if current_factors[p] == 1:
                    new_n = current_n // p
                    if new_n < 2:
                        continue
                    score = compute_tau_perp(new_n)
                    if score > current_score:
                        current_score = score
                        current_n = new_n
                        del current_factors[p]
                        improved = True
                        break
                else:
                    new_e = current_factors[p] - 1
                    new_n = current_n // p
                    score = compute_tau_perp(new_n)
                    if score > current_score:
                        current_score = score
                        current_n = new_n
                        current_factors[p] = new_e
                        improved = True
                        break
        
        if not improved:
            # Random restart: try a random combination
            if random.random() < 0.3:
                # Pick a random subset of primes
                k = random.randint(8, 14)
                indices = random.sample(range(len(prime_pool)), k)
                n = 1
                for idx in sorted(indices):
                    n *= prime_pool[idx]
                    if n > 10**18:
                        n = -1
                        break
                if n > 0 and n <= 10**18 and n >= 2:
                    score = compute_tau_perp(n)
                    if score > current_score:
                        current_score = score
                        current_n = n
                        current_factors = factorize(n)
                        improved = True
        
        if not improved and random.random() < 0.1:
            # Occasional larger random perturbation
            k = random.randint(6, 16)
            indices = random.sample(range(len(prime_pool)), k)
            n = 1
            for idx in sorted(indices):
                n *= prime_pool[idx]
                if n > 10**18:
                    n = -1
                    break
            if n > 0 and n <= 10**18 and n >= 2:
                score = compute_tau_perp(n)
                if score > current_score:
                    current_score = score
                    current_n = n
                    current_factors = factorize(n)
    
    if current_n > best_n or current_score > best_score:
        best_n = current_n
        best_score = current_score
    
    return best_n
