import random
import time
import math

def construct():
    start_time = time.time()
    timeout = 85  # Leave 5s margin for safety
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    best_count = 0
    best_solution = []
    
    # Precompute gcds for speed
    gcd_cache = {}
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            gcd_cache[(i, j)] = math.gcd(i, j)
            
    def get_gcd(m1, m2):
        if m1 == m2: return m1
        if m1 > m2: m1, m2 = m2, m1
        return gcd_cache[(m1, m2)]

    def is_valid_assignment(residues, m_new, r_new):
        """Check if adding (m_new, r_new) to current residues is valid."""
        for m_prime, r_prime in residues.items():
            g = get_gcd(m_new, m_prime)
            if r_new % g == r_prime % g:
                return False
        return True

    def find_valid_residue(residues, m_new):
        """Find a residue for m_new that is valid against existing residues."""
        for r in range(m_new):
            if is_valid_assignment(residues, m_new, r):
                return r
        return -1

    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    div_counts = {m: count_divisors(m) for m in moduli}
    
    # Identify primes
    primes = [m for m in moduli if all(m % i != 0 for i in range(2, int(math.isqrt(m)) + 1))]
    composites = [m for m in moduli if m not in primes]
    
    def try_add_with_repair(residues, m_new, max_repair_steps=20):
        """
        Try to add m_new by finding a valid residue or repairing conflicts.
        Returns True if successful.
        """
        # First try direct addition
        r = find_valid_residue(residues, m_new)
        if r != -1:
            residues[m_new] = r
            return True
        
        # Try repair: pick a random residue and fix conflicts
        for attempt in range(max_repair_steps):
            if time.time() - start_time > timeout:
                return False
            
            r_new = random.randint(0, m_new - 1)
            
            # Find all conflicts
            conflicting = []
            for m_prime, r_prime in residues.items():
                g = get_gcd(m_new, m_prime)
                if r_new % g == r_prime % g:
                    conflicting.append(m_prime)
            
            if not conflicting:
                residues[m_new] = r_new
                return True
            
            # Try to resolve one conflict at a time
            random.shuffle(conflicting)
            resolved_any = False
            
            for m_conf in conflicting:
                g_conf = get_gcd(m_new, m_conf)
                forbidden_val = r_new % g_conf
                
                # Try to change m_conf's residue
                for r_alt in range(m_conf):
                    if r_alt % g_conf == forbidden_val:
                        continue
                    
                    # Check if r_alt is valid against all other moduli
                    valid_alt = True
                    for m_other, r_other in residues.items():
                        if m_other == m_conf:
                            continue
                        g_other = get_gcd(m_conf, m_other)
                        if r_alt % g_other == r_other % g_other:
                            valid_alt = False
                            break
                    
                    if valid_alt:
                        residues[m_conf] = r_alt
                        resolved_any = True
                        break
                
                if resolved_any:
                    break
            
            if not resolved_any:
                continue
            
            # After repair, check if m_new can now be added
            if is_valid_assignment(residues, m_new, r_new):
                residues[m_new] = r_new
                return True
        
        return False

    # Strategy: Multiple approaches
    # 1. Primes first (they have gcd=1 with most numbers)
    # 2. Then composites sorted by ascending divisor count (easier first)
    # 3. Random order
    
    for restart in range(1000):
        if time.time() - start_time > timeout:
            break
        
        residues = {}
        
        # Choose ordering strategy based on restart
        if restart % 4 == 0:
            # Primes first, then composites by ascending divisor count
            ordered = primes[:] + sorted(composites, key=lambda m: div_counts[m])
        elif restart % 4 == 1:
            # Primes first, then composites by descending divisor count
            ordered = primes[:] + sorted(composites, key=lambda m: div_counts[m], reverse=True)
        elif restart % 4 == 2:
            # All sorted by ascending divisor count
            ordered = sorted(moduli, key=lambda m: div_counts[m])
        else:
            # Random
            ordered = moduli[:]
            random.shuffle(ordered)
        
        # Build greedily
        for m in ordered:
            if time.time() - start_time > timeout:
                break
            
            r = find_valid_residue(residues, m)
            if r != -1:
                residues[m] = r
        
        current_count = len(residues)
        
        if current_count > best_count:
            best_count = current_count
            best_solution = [[residues[m2], m2] for m2 in residues]
        
        # Local search: try to add remaining moduli with repair
        remaining = [m for m in moduli if m not in residues]
        random.shuffle(remaining)
        
        for m_new in remaining:
            if time.time() - start_time > timeout:
                break
            
            if try_add_with_repair(residues, m_new):
                current_count = len(residues)
                if current_count > best_count:
                    best_count = current_count
                    best_solution = [[residues[m2], m2] for m2 in residues]
        
        # Additional local search: try removing one modulus and adding two others
        # This is a 1-2 exchange move
        if time.time() - start_time < timeout - 10:
            for _ in range(50):  # Limit exchanges per restart
                if time.time() - start_time > timeout:
                    break
                
                if len(residues) < 2:
                    break
                
                # Pick a random modulus to potentially remove
                m_remove = random.choice(list(residues.keys()))
                r_remove = residues[m_remove]
                del residues[m_remove]
                
                # Try to add two other moduli
                added = 0
                remaining = [m for m in moduli if m not in residues]
                random.shuffle(remaining)
                
                for m_new in remaining:
                    if added >= 2:
                        break
                    if try_add_with_repair(residues, m_new):
                        added += 1
                
                if added > 1:
                    current_count = len(residues)
                    if current_count > best_count:
                        best_count = current_count
                        best_solution = [[residues[m2], m2] for m2 in residues]
                else:
                    # Restore the removed modulus
                    residues[m_remove] = r_remove

    if not best_solution:
        # Fallback: just use primes
        best_solution = [[0, m] for m in primes]
        
    return best_solution
