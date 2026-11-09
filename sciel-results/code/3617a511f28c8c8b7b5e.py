import math
import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcd for all pairs
    gcd_cache = {}
    for i, m1 in enumerate(moduli):
        for m2 in moduli[i+1:]:
            gcd_cache[(m1, m2)] = math.gcd(m1, m2)
    
    def get_gcd(m1, m2):
        if m1 > m2:
            m1, m2 = m2, m1
        return gcd_cache.get((m1, m2), math.gcd(m1, m2))
    
    # Identify primes and composites
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    
    # Factor each composite
    def factorize(n):
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    factors = {m: factorize(m) for m in composites}
    
    # Strategy: Greedy with smart ordering + local search
    # Assign r=0 to all primes initially
    # For composites, find valid residues
    
    def find_valid_residue(selected_map, m):
        """Find a residue r for modulus m compatible with all in selected_map.
        selected_map: dict {modulus: residue}
        """
        # Get prime factors of m for quick check
        prime_divs = set(factors.get(m, []))
        
        for r in range(m):
            # Quick check: for prime divisors of m, r should not be 0 mod p 
            # (since primes have r=0)
            # Actually this is only a necessary condition, not sufficient
            valid = True
            for m2, r2 in selected_map.items():
                if m2 == m:
                    continue
                g = get_gcd(m, m2)
                if g > 1 and (r - r2) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def build_greedy(order):
        """Build a solution by processing moduli in given order."""
        selected = {}  # modulus -> residue
        for m in order:
            if m in selected:
                continue
            r = find_valid_residue(selected, m)
            if r is not None:
                selected[m] = r
        return selected
    
    def solution_size(selected):
        return len(selected)
    
    # Try multiple orderings
    best_selected = {}
    
    # Ordering 1: Primes first (r=0), then composites by number of prime factors
    comp_by_factor_count = sorted(composites, key=lambda m: len(factors[m]))
    order1 = primes + comp_by_factor_count
    sel1 = build_greedy(order1)
    if solution_size(sel1) > solution_size(best_selected):
        best_selected = sel1
    
    # Ordering 2: Primes first, then composites by value
    order2 = primes + sorted(composites)
    sel2 = build_greedy(order2)
    if solution_size(sel2) > solution_size(best_selected):
        best_selected = sel2
    
    # Ordering 3: Primes first, then composites by number of distinct prime factors
    def num_distinct_prime_factors(m):
        return len(set(factors.get(m, [])))
    
    comp_by_distinct = sorted(composites, key=num_distinct_prime_factors)
    order3 = primes + comp_by_distinct
    sel3 = build_greedy(order3)
    if solution_size(sel3) > solution_size(best_selected):
        best_selected = sel3
    
    # Local search: try to add unselected composites by reassigning residues
    def local_search(selected, max_iters=5000):
        """Try to improve by reassigning residues to add more moduli."""
        selected = dict(selected)
        unselected = [m for m in moduli if m not in selected]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if not unselected:
                break
            
            # Pick a random unselected modulus
            idx = random.randint(0, len(unselected) - 1)
            m_new = unselected[idx]
            
            # Try to find a residue for m_new
            r_new = find_valid_residue(selected, m_new)
            if r_new is not None:
                selected[m_new] = r_new
                unselected.pop(idx)
                continue
            
            # Try reassigning residues of selected moduli to make room
            # Pick a random selected modulus to reassign
            sel_mods = list(selected.keys())
            if not sel_mods:
                break
            m_swap = sel_mods[random.randint(0, len(sel_mods) - 1)]
            old_r = selected[m_swap]
            
            # Try different residues for m_swap
            found = False
            for r_try in range(m_swap):
                if r_try == old_r:
                    continue
                # Check if m_swap can take r_try
                compatible = True
                for m3, r3 in selected.items():
                    if m3 == m_swap:
                        continue
                    g = get_gcd(m_swap, m3)
                    if g > 1 and (r_try - r3) % g == 0:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Now check if m_new can be added with some residue
                temp_selected = dict(selected)
                temp_selected[m_swap] = r_try
                r_new2 = find_valid_residue(temp_selected, m_new)
                if r_new2 is not None:
                    selected[m_swap] = r_try
                    selected[m_new] = r_new2
                    unselected.pop(idx)
                    found = True
                    break
            
            if not found and random.random() < 0.1:
                # Occasionally try a more aggressive reassignment
                pass
        
        return selected
    
    # Run local search from best greedy solution
    best_selected = local_search(best_selected)
    
    # Try multiple random restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random ordering with primes biased to front
        random_order = moduli.copy()
        random.shuffle(random_order)
        # Move primes to front with some randomness
        primes_in_order = [m for m in random_order if is_prime(m)]
        comps_in_order = [m for m in random_order if not is_prime(m)]
        random.shuffle(primes_in_order)
        random.shuffle(comps_in_order)
        mixed = primes_in_order + comps_in_order
        
        sel = build_greedy(mixed)
        sel = local_search(sel, max_iters=2000)
        
        if solution_size(sel) > solution_size(best_selected):
            best_selected = sel
    
    # Convert to required format: list of [residue, modulus]
    result = [[r, m] for m, r in best_selected.items()]
    
    if not result:
        return [[0, 2]]
    
    return result

