import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcds
    gcd_matrix = {}
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            gcd_matrix[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
            
    def is_compatible(selected, m, r):
        for m_p, r_p in selected:
            g = gcd_matrix.get((m, m_p), math.gcd(m, m_p))
            if (r - r_p) % g == 0:
                return False
        return True

    def find_valid_residue(selected, m):
        if not selected:
            return 0
        # Collect forbidden residues for each gcd
        # For each existing (m_p, r_p), g = gcd(m, m_p)
        # r cannot be congruent to r_p mod g
        # We want to find r in [0, m) such that for all (m_p, r_p), r % g != r_p % g
        
        # Optimization: Instead of checking all r from 0 to m-1,
        # we can just iterate. m is at most 128, so it's fast.
        for r in range(m):
            if is_compatible(selected, m, r):
                return r
        return None

    best_solution = []
    
    # Identify primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    
    # Strategy: 
    # 1. Try to include all primes (they are always mutually disjoint).
    # 2. Add composites greedily.
    
    # Let's run randomized greedy with different orders
    orders = []
    
    # 1. Primes first, then composites random
    for _ in range(500):
        comp = composites.copy()
        random.shuffle(comp)
        orders.append(primes + comp)
        
    # 2. Composites first?
    for _ in range(500):
        comp = composites.copy()
        random.shuffle(comp)
        orders.append(comp + primes)
        
    # 3. Full random
    for _ in range(500):
        full = moduli.copy()
        random.shuffle(full)
        orders.append(full)
        
    # 4. Sorted by number of divisors (fewer first - easier?)
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
        
    div_count = {m: count_divisors(m) for m in moduli}
    orders.append(sorted(moduli, key=lambda x: div_count[x]))
    orders.append(sorted(moduli, key=lambda x: -div_count[x]))
    
    # 5. Randomized sort by divisors
    for _ in range(200):
        shuffled_moduli = moduli.copy()
        random.shuffle(shuffled_moduli)
        # Sort by divisors but with some noise
        orders.append(sorted(shuffled_moduli, key=lambda x: div_count[x] + random.uniform(-0.5, 0.5)))
        
    for order in orders:
        if time.time() > deadline:
            break
            
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(selected, m)
            if r is not None:
                selected.append((m, r))
                
        if len(selected) > len(best_solution):
            best_solution = selected
            
    # Local Search / Improvement
    # Try to add back rejected moduli?
    # Or try to swap residues to allow more moduli?
    
    # Let's try a simple local search: 
    # Take the best solution, and try to add any modulus not in it.
    # If we can add one, do it.
    # If not, try changing a residue of an existing modulus to free up space?
    
    # This is complex. Let's stick to more random restarts.
    
    result = [[r, m] for m, r in best_solution]
    
    if not result:
        return [[0, 2]]
        
    return result
