import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    best_set = None
    best_size = 0
    
    # Precompute gcds
    gcd_table = {}
    for i in range(2, max_mod + 1):
        for j in range(i + 1, max_mod + 1):
            gcd_table[(i, j)] = gcd(i, j)
            
    def check_residues(mods):
        """
        Greedily assign residues.
        Order mods by modulus.
        For each mod, pick a residue that doesn't conflict with previous ones.
        """
        if not mods:
            return []
        
        # Sort by modulus to make constraints simpler (smaller moduli constrain larger ones more strongly in terms of density? Not necessarily, but helps backtracking)
        sorted_mods = sorted(mods)
        residues = [None] * len(sorted_mods)
        
        for i, m in enumerate(sorted_mods):
            valid = True
            for r in range(m):
                conflict = False
                for j in range(i):
                    m_j = sorted_mods[j]
                    r_j = residues[j]
                    g = gcd_table.get((m_j, m), gcd(m_j, m))
                    if r % g == r_j % g:
                        conflict = True
                        break
                if not conflict:
                    residues[i] = r
                    break
            if valid is True and residues[i] is None:
                return None
        return [(residues[i], sorted_mods[i]) for i in range(len(sorted_mods))]

    # Try to find a large set
    # Strategy: Start with a promising base and grow randomly, or random restarts.
    
    # Known good set: Powers of 2
    powers_of_2 = [2**k for k in range(1, 8) if 2**k <= 128]
    res = check_residues(powers_of_2)
    if res:
        best_set = res
        best_size = len(res)
        
    # Randomized search for larger sets
    # We want to find a set of moduli M such that check_residues(M) is not None.
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        # Randomly select a subset of moduli, biased towards smaller moduli or specific structures
        # Or start with powers of 2 and try to swap/replace
        
        # Heuristic: Try to build a set from scratch using greedy addition with random order
        random.shuffle(moduli)
        current_set = []
        
        for m in moduli:
            if time.time() > deadline:
                break
                
            # Try to add m to current_set
            trial_set = current_set + [m]
            
            # Quick check: if gcd is 1 with any existing, skip?
            # Actually, check_residues will handle it, but pre-filtering helps.
            possible = True
            for existing in current_set:
                if gcd(m, existing) == 1:
                    possible = False
                    break
            
            if not possible:
                continue
                
            res = check_residues(trial_set)
            if res:
                current_set = trial_set
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = check_residues(current_set)
            
            # If we found a very good one, we can stop early if we hit a theoretical upper bound?
            # Upper bound is 127. But practically, much lower.
            if best_size > 15:
                break

    # One more specific attempt: Mix of small primes
    # Try all multiples of 2, 3, 5 etc in small numbers
    small_mods = [m for m in range(2, 20)]
    random.shuffle(small_mods)
    current_set = []
    for m in small_mods:
        trial_set = current_set + [m]
        possible = True
        for existing in current_set:
            if gcd(m, existing) == 1:
                possible = False
                break
        if not possible:
            continue
        res = check_residues(trial_set)
        if res:
            current_set = trial_set
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = check_residues(current_set)

    if not best_set:
        # Fallback to powers of 2
        best_set = [((1<<k)-1, 1<<(k+1)) for k in range(7)]

    return best_set
