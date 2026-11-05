import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    
    # Precompute number of divisors for sorting heuristics
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    div_counts = {m: num_divisors(m) for m in moduli}
    
    def is_compatible(r, m, selected):
        for r_k, m_k in selected:
            g = math.gcd(m, m_k)
            if r % g == r_k % g:
                return False
        return True

    def find_residue(m, selected):
        # Try residues in a specific order? 
        # Usually 0 is a good start, but might conflict.
        # We just need ANY valid residue.
        for r in range(m):
            if is_compatible(r, m, selected):
                return r
        return None

    def solve_with_order(order):
        selected = []
        for m in order:
            r = find_residue(m, selected)
            if r is not None:
                selected.append((r, m))
        return selected

    # Generate different orders
    orders = []
    
    # 1. Ascending number of divisors
    orders.append(sorted(moduli, key=lambda m: div_counts[m]))
    
    # 2. Descending number of divisors
    orders.append(sorted(moduli, key=lambda m: -div_counts[m]))
    
    # 3. Ascending value
    orders.append(sorted(moduli))
    
    # 4. Descending value
    orders.append(sorted(moduli, reverse=True))
    
    # 5. Primes first, then rest by divisors
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    orders.append(primes + sorted(composites, key=lambda m: div_counts[m]))
    
    # 6. Random shuffles (limited by time)
    random.seed(42)
    for _ in range(10):
        shuffled = moduli.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    best_solution = []
    best_len = 0
    
    for order in orders:
        if time.time() > deadline:
            break
        sol = solve_with_order(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Convert to list of lists as required
    # [[residue, modulus], ...]
    return [[r, m] for r, m in best_solution]
