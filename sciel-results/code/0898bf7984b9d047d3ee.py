"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = _gcd(m1, m2)
    if g == 1:
        return False  # always intersect
    return (r2 - r1) % g != 0

def _can_add(r_new, m_new, existing):
    for (r, m) in existing:
        if not _disjoint(r_new, m_new, r, m):
            return False
    return True

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # safety margin
    
    # All candidate moduli from 2 to 128
    moduli = list(range(2, 129))
    
    # Sort moduli by a heuristic: prefer moduli with more shared factors with others
    # to maximize disjointness options. Actually, let's try multiple orderings.
    
    best_result = None
    best_count = 0
    
    # Strategy: try different orderings of moduli and use greedy with residue search
    # For each modulus in order, try all residues 0..m-1 and pick one that works.
    
    # Precompute gcds between all pairs of moduli for speed
    n_mod = len(moduli)
    gcd_table = [[0]*n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i+1, n_mod):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Try multiple heuristic orderings
    orderings = []
    
    # Ordering 1: by number of prime factors (more factors = more shared gcd potential)
    def num_prime_factors(m):
        count = 0
        d = 2
        mm = m
        while d * d <= mm:
            if mm % d == 0:
                count += 1
                while mm % d == 0:
                    mm //= d
            d += 1
        if mm > 1:
            count += 1
        return count
    
    # Ordering by descending number of prime factors, then by modulus
    orderings.append(sorted(moduli, key=lambda m: (-num_prime_factors(m), m)))
    # Ordering by descending modulus
    orderings.append(sorted(moduli, reverse=True))
    # Ordering by ascending modulus
    orderings.append(sorted(moduli))
    # Ordering by descending number of divisors
    def num_divisors(m):
        count = 0
        for d in range(1, int(math.isqrt(m)) + 1):
            if m % d == 0:
                count += 1
                if d != m // d:
                    count += 1
        return count
    orderings.append(sorted(moduli, key=lambda m: (-num_divisors(m), m)))
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        # Greedy: for each modulus in order, find a residue that works
        selected = []  # list of (residue, modulus)
        selected_moduli = set()
        
        for m in ordering:
            if time.time() > deadline:
                break
            
            # Try all residues for this modulus
            found = False
            for r in range(m):
                if _can_add(r, m, selected):
                    selected.append((r, m))
                    selected_moduli.add(m)
                    found = True
                    break
            
            if not found:
                # Try with different residue orderings (even, odd, etc.)
                # Actually let's just skip this modulus
                pass
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = [[r, m] for (r, m) in selected]
    
    # Also try a more careful backtracking on a subset
    if time.time() < deadline - 5:
        # Use the best ordering and try to improve with local search / backtracking
        # Start with the best greedy result and try to add more moduli
        
        # Try all remaining moduli that weren't selected
        if best_result:
            selected_set = set(m for _, m in best_result)
            remaining = [m for m in moduli if m not in selected_set]
            
            # Try to add each remaining modulus
            improved = list(best_result)
            for m in remaining:
                if time.time() > deadline:
                    break
                for r in range(m):
                    if _can_add(r, m, improved):
                        improved.append((r, m))
                        break
            
            if len(improved) > best_count:
                best_count = len(improved)
                best_result = [[r, m] for (r, m) in improved]
    
    if best_result is None:
        # Fallback: return the original solution
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    return best_result
# EVOLVE-BLOCK-END
