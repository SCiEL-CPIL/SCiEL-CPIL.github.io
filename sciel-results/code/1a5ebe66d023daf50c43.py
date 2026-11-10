"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
    MAX_MOD = 128
    
    def is_pairwise_disjoint(classes):
        """Check exact gcd criterion for all pairs."""
        n = len(classes)
        for i in range(n):
            r1, m1 = classes[i]
            for j in range(i + 1, n):
                r2, m2 = classes[j]
                g = math.gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True
    
    def find_residue_for(m, selected):
        """Find a valid residue for modulus m given already selected classes."""
        for r in range(m):
            valid = True
            for r_prime, m_prime in selected:
                g = math.gcd(m, m_prime)
                if (r - r_prime) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    # Strategy 1: Greedy by modulus order (original approach)
    def greedy_by_order(moduli_order):
        selected = []
        for m in moduli_order:
            if time.time() > deadline:
                break
            r = find_residue_for(m, selected)
            if r is not None:
                selected.append([r, m])
        return selected
    
    # Try multiple orderings
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Ordering 1: ascending
    best = greedy_by_order(all_moduli)
    best_count = len(best)
    
    if time.time() > deadline:
        return best if best else [[1, 2]]
    
    # Ordering 2: descending
    cand = greedy_by_order(list(reversed(all_moduli)))
    if len(cand) > best_count:
        best = cand
        best_count = len(cand)
    
    if time.time() > deadline:
        return best if best else [[1, 2]]
    
    # Ordering 3: powers of 2 first, then others
    def mod_key(m):
        # Sort by: prime factors, then value
        def num_prime_factors(n):
            count = 0
            d = 2
            while d * d <= n:
                while n % d == 0:
                    count += 1
                    n //= d
                d += 1
            if n > 1:
                count += 1
            return count
        return (num_prime_factors(m), m)
    
    sorted_mods = sorted(all_moduli, key=mod_key)
    cand = greedy_by_order(sorted_mods)
    if len(cand) > best_count:
        best = cand
        best_count = len(cand)
    
    if time.time() > deadline:
        return best if best else [[1, 2]]
    
    # Strategy 2: Local search - try to improve by reassigning residues
    # For the current best set, try swapping residues to free up space for more moduli
    if time.time() < deadline - 5.0:
        current = [list(c) for c in best]
        used_moduli = set(m for _, m in current)
        unused_moduli = [m for m in all_moduli if m not in used_moduli]
        
        # Try adding unused moduli one by one, possibly reassigning residues
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for m_new in list(unused_moduli):
                if time.time() > deadline - 2.0:
                    break
                # Try all possible residues for m_new
                r_new = find_residue_for(m_new, current)
                if r_new is not None:
                    current.append([r_new, m_new])
                    unused_moduli.remove(m_new)
                    improved = True
                    break
            
            # Also try reassigning residues of existing classes to free up space
            if not improved:
                for i, (r_i, m_i) in enumerate(current):
                    if time.time() > deadline - 2.0:
                        break
                    for r_try in range(m_i):
                        if r_try == r_i:
                            continue
                        old_r = current[i][0]
                        current[i][0] = r_try
                        # Check if still valid
                        if is_pairwise_disjoint(current):
                            # Try to add an unused modulus
                            for m_new in list(unused_moduli):
                                r_new = find_residue_for(m_new, current)
                                if r_new is not None:
                                    current.append([r_new, m_new])
                                    unused_moduli.remove(m_new)
                                    improved = True
                                    break
                            if not improved:
                                current[i][0] = old_r
                        else:
                            current[i][0] = old_r
                    if improved:
                        break
        
        if len(current) > best_count:
            best = current
            best_count = len(current)
    
    # Final verification
    if not is_pairwise_disjoint(best):
        # Fallback to a known valid small set
        best = [[1, 2], [1, 4], [1, 8], [1, 16], [1, 32], [1, 64], [1, 128]]
    
    if not best:
        best = [[1, 2]]
    
    return best
# EVOLVE-BLOCK-END
