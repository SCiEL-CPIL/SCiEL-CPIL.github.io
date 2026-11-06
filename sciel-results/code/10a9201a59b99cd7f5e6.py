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

    def find_valid_residue(m, selected_moduli, residues):
        """
        Finds a valid residue for m given the current selection.
        Returns None if no valid residue exists.
        """
        domain = list(range(m))
        for m_prime in selected_moduli:
            g = get_gcd(m, m_prime)
            if g == 1:
                continue
            forbidden = residues[m_prime] % g
            # Filter domain
            new_domain = [r for r in domain if r % g != forbidden]
            if not new_domain:
                return None
            domain = new_domain
        
        if not domain:
            return None
            
        # Choose a random value from the remaining domain
        return random.choice(domain)

    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    # Precompute divisor counts
    div_counts = {m: count_divisors(m) for m in moduli}
    
    def local_search(selected, residues, available):
        """
        Attempts to improve the solution using local search.
        selected: list of moduli
        residues: dict of modulus -> residue
        available: list of unselected moduli
        Returns new selected, residues, available if improved.
        """
        improved = True
        while improved:
            if time.time() - start_time > timeout:
                break
            improved = False
            
            # 1. Try to add any available modulus
            # Iterate over a copy because we modify available
            for m in available[:]: 
                r = find_valid_residue(m, selected, residues)
                if r is not None:
                    selected.append(m)
                    residues[m] = r
                    available.remove(m)
                    improved = True
                    break # Restart the while loop to check for more additions
            
            if not improved:
                # 2. Try Swap + Recover
                # For each m_in in available, try swapping with m_out in selected
                # to see if we can add a third modulus.
                
                # To keep it fast, limit attempts
                attempts = 0
                max_attempts = 100 # Increased attempts to allow more exploration
                
                while attempts < max_attempts and not improved:
                    attempts += 1
                    if time.time() - start_time > timeout:
                        break
                    
                    if not available or not selected:
                        break
                        
                    m_in = random.choice(available)
                    m_out = random.choice(selected)
                    
                    # Simulate swap
                    selected_backup = selected[:]
                    residues_backup = residues.copy()
                    
                    # Remove m_out
                    selected.remove(m_out)
                    del residues[m_out]
                    
                    # Try to add m_in
                    r_in = find_valid_residue(m_in, selected, residues)
                    if r_in is not None:
                        # m_in added successfully
                        selected.append(m_in)
                        residues[m_in] = r_in
                        available.remove(m_in)
                        
                        # Now try to recover m_out or add another from available
                        # We have size k-1+1 = k. We want k+1.
                        # So we need to add one more.
                        # Candidates: m_out (which is now free) or any other from available.
                        # If we add m_out back, size is k. No gain.
                        # We need to add a *different* one to gain +1?
                        # Actually, if we add m_out back, we are back to start.
                        # So we must add a *third* modulus.
                        
                        recovery_found = False
                        # Try to add any modulus from available (which now excludes m_in)
                        # Note: m_out is NOT in available.
                        for m_rec in available[:]:
                             r_rec = find_valid_residue(m_rec, selected, residues)
                             if r_rec is not None:
                                 selected.append(m_rec)
                                 residues[m_rec] = r_rec
                                 available.remove(m_rec)
                                 # We added m_in and m_rec. Removed m_out.
                                 # Net change: +1
                                 improved = True
                                 recovery_found = True
                                 break
                                 
                        if not recovery_found:
                            # Revert swap
                            selected = selected_backup
                            residues = residues_backup
                            # m_in is back in available?
                            # In the success path, we did available.remove(m_in).
                            # So we need to put it back.
                            available.append(m_in)
                            
                    else:
                        # m_in couldn't be added even after removing m_out
                        # Revert swap
                        selected = selected_backup
                        residues = residues_backup
                        # m_in is still in available (we didn't remove it in the failure path)

        return selected, residues, available

    for restart in range(2000):
        if time.time() - start_time > timeout:
            break
            
        selected = []
        residues = {}
        available = moduli[:]
        
        # Choose ordering strategy based on restart
        strategy = restart % 3
        if strategy == 0:
            random.shuffle(available)
        elif strategy == 1:
            # Sort by number of divisors (ascending)
            available = sorted(moduli, key=lambda m: div_counts[m])
        else:
            # Sort by value (ascending)
            available = moduli[:]
        
        # Greedy Phase
        for m in available[:]: # Iterate over copy as we remove from available
            if time.time() - start_time > timeout:
                break
            
            r = find_valid_residue(m, selected, residues)
            if r is not None:
                selected.append(m)
                residues[m] = r
                available.remove(m)
                
        # Local Search Phase
        if time.time() - start_time < timeout:
            selected, residues, available = local_search(selected, residues, available)
            
        if len(selected) > best_count:
            best_count = len(selected)
            best_solution = [[residues[m], m] for m in selected]
            
    if not best_solution:
        return [[(1<<k)-1,1<<(k+1)] for k in range(7)]
        
    return best_solution
