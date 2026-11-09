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
        # Try all residues
        # Optimization: If selected is empty, return 0
        if not selected:
            return 0
        for r in range(m):
            if is_compatible(selected, m, r):
                return r
        return None

    best_solution = []
    
    # Heuristic: Primes are easiest.
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
    
    # Base solution: All primes
    # We need to assign residues to primes. Since they are coprime, any residues work.
    # Let's assign 0 to all primes?
    # If r_p = 0 for all primes, then for a composite c, 
    # constraint with prime p: r_c != 0 mod gcd(c,p).
    # If p | c, gcd=p. r_c != 0 mod p.
    # If p !| c, gcd=1. No constraint.
    
    # Let's run randomized greedy with different orders
    orders = []
    
    # 1. Primes first, then composites random
    for _ in range(200):
        comp = composites.copy()
        random.shuffle(comp)
        orders.append(primes + comp)
        
    # 2. Composites first?
    for _ in range(200):
        comp = composites.copy()
        random.shuffle(comp)
        orders.append(comp + primes)
        
    # 3. Full random
    for _ in range(200):
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
    # Try to swap elements to increase size?
    # Or just run more random restarts if time permits
    
    # The above greedy might not be optimal.
    # A better approach: 
    # Start with a large set, remove conflicts.
    
    # Let's try a "remove conflicts" approach
    # Start with all moduli.
    # Assign random residues.
    # Check all pairs. If conflict, remove one.
    # This is slow.
    
    # The greedy approach is standard for this kind of packing problem.
    # The key is the order.
    
    # Let's add a "backtracking" lite version? No, too complex.
    
    # Let's ensure we try to keep primes.
    # In the "Primes first" order, all primes are added.
    # Then composites are added if they fit.
    
    result = [[r, m] for m, r in best_solution]
    
    if not result:
        return [[0, 2]]
        
    return result
