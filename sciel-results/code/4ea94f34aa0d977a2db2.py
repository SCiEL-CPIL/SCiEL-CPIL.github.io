import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: 90s limit, aim to finish by 85s
    deadline = start_time + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcds for O(1) lookup
    gcd_matrix = {}
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            gcd_matrix[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
            
    def is_compatible(selected, m, r):
        """
        Check if (m, r) is disjoint from all (m', r') in selected.
        Disjoint condition: (r - r') % gcd(m, m') != 0
        """
        for m_p, r_p in selected:
            # Use precomputed gcd if available, else compute
            g = gcd_matrix.get((m, m_p), math.gcd(m, m_p))
            if (r - r_p) % g == 0:
                return False
        return True

    def find_valid_residue(selected, m):
        """
        Find any r in [0, m-1] compatible with selected.
        """
        if not selected:
            return 0
        for r in range(m):
            if is_compatible(selected, m, r):
                return r
        return None

    best_solution = []
    
    # Pre-compute properties for heuristics
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
        
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    div_count = {m: count_divisors(m) for m in moduli}
    
    # Generate orderings
    orderings = []
    
    # 1. Deterministic Heuristics
    # Primes first (sorted by value), then composites sorted by value
    orderings.append(primes + composites)
    # Composites first, then primes
    orderings.append(composites + primes)
    # Sorted by divisor count (fewer divisors first - primes are 2)
    orderings.append(sorted(moduli, key=lambda x: div_count[x]))
    # Sorted by divisor count (more divisors first)
    orderings.append(sorted(moduli, key=lambda x: -div_count[x]))
    # Sorted by value
    orderings.append(sorted(moduli))
    # Sorted by value descending
    orderings.append(sorted(moduli, reverse=True))
    
    # 2. Randomized Restart Strategies
    # We will generate random orders on the fly to save memory and time spent generating
    # We run a loop that picks a strategy based on a counter or random choice
    
    # Strategy A: Primes First + Random Composites (Strongest from Inspiration 1)
    # Strategy B: Composites First + Random Primes
    # Strategy C: Full Random Shuffle
    # Strategy D: Random Shuffle of Composites, Primes appended at end (variant of A)
    
    # Estimate how many restarts we can do.
    # Greedy step: O(N * K) where N=127, K~50 avg. ~6000 ops.
    # 1000 restarts -> 6M ops. Very fast in Python (<1s).
    # We can do many more. Let's aim for 2000-5000 restarts if time permits.
    
    # We will interleave the deterministic ones first (already in list)
    
    # Let's run the deterministic ones first
    for order in orderings:
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
            
    # Now run randomized restarts
    # We will prioritize "Primes First" strategy as it showed best results
    # We'll do 60% Primes First, 20% Composites First, 20% Full Random
    
    restart_count = 0
    # Estimate remaining time
    # Let's just loop until deadline, checking time every 100 restarts or so
    # But we need to ensure we don't stall.
    
    while time.time() < deadline - 2.0: # Keep 2s buffer for final processing
        restart_count += 1
        
        # Choose strategy
        r_choice = random.random()
        if r_choice < 0.6:
            # Primes first, random composites
            comp = composites.copy()
            random.shuffle(comp)
            order = primes + comp
        elif r_choice < 0.8:
            # Composites first, random primes
            comp = composites.copy()
            random.shuffle(comp)
            pri = primes.copy()
            random.shuffle(pri)
            order = comp + pri
        else:
            # Full random
            order = moduli.copy()
            random.shuffle(order)
            
        selected = []
        for m in order:
            if time.time() > deadline - 2.0:
                break
            r = find_valid_residue(selected, m)
            if r is not None:
                selected.append((m, r))
                
        if len(selected) > len(best_solution):
            best_solution = selected
            # If we find a very good solution, we might want to try to improve it?
            # For now, greedy restarts are the main driver.
            
    # Convert to required format: list of [residue, modulus]
    result = [[r, m] for m, r in best_solution]
    
    # If no solution found (should not happen), return a valid single one
    if not result:
        return [[0, 2]]
        
    return result
