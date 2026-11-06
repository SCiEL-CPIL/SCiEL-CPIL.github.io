import random
from fractions import Fraction

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Strategy 1: Use primes >= 13 (Seed set)
    # Strategy 2: Try swapping
    
    # We will use a local search to optimize residues
    # to maximize the covered prefix L.
    
    # Initial set: primes >= 13
    P = [p for p in primes if p >= 13]
    
    # Check budget
    sum_recip = sum(Fraction(1, p) for p in P)
    if sum_recip > Fraction(1, 1):
        # Remove largest primes until budget is met
        # (Seed already satisfies this, but good for robustness)
        pass
        
    # Initialize residues
    # Seed initialization: r_p = (index+1) % p
    # But we can start with random or greedy.
    # Let's use the seed init as a good starting point.
    residues = {}
    for i, p in enumerate(P):
        residues[p] = (i + 1) % p
        
    def get_covered_prefix(res_map, P_list, max_L=10000):
        covered = [False] * (max_L + 1)
        for p in P_list:
            r = res_map[p]
            # Mark multiples
            # n = r, r+p, r+2p...
            # Note: r can be 0. If r=0, covers p, 2p...
            # If r>0, covers r, r+p...
            start = r if r > 0 else p
            for n in range(start, max_L + 1, p):
                covered[n] = True
        
        L = 0
        for n in range(1, max_L + 1):
            if covered[n]:
                L = n
            else:
                break
        return L

    current_L = get_covered_prefix(residues, P, 10000)
    
    # Local Search
    # We want to increase L.
    # Find smallest uncovered n.
    # Try to cover it by changing a residue.
    
    improved = True
    while improved:
        improved = False
        # Recalculate coverage to find smallest uncovered
        covered = [False] * (10001)
        for p in P:
            r = residues[p]
            start = r if r > 0 else p
            for n in range(start, 10001, p):
                covered[n] = True
        
        # Find smallest uncovered
        n_uncovered = -1
        for n in range(1, 10001):
            if not covered[n]:
                n_uncovered = n
                break
        
        if n_uncovered == -1:
            break # Covered 1..10000
            
        # Try to cover n_uncovered
        # For each prime p, see if we can set r_p = n_uncovered % p
        # such that 1..n_uncovered-1 remain covered.
        
        best_change = None
        for p in P:
            new_r = n_uncovered % p
            if residues[p] == new_r:
                continue # Already set, but n_uncovered is uncovered? Impossible.
            
            # Check if changing r_p to new_r keeps 1..n_uncovered-1 covered
            # We only need to check if any number in 1..n_uncovered-1 
            # was covered ONLY by p with old_r, and is NOT covered by others.
            # This is expensive to check exactly.
            # Approximation: Check if the new residue covers n_uncovered, 
            # and assume it doesn't break previous coverage if we are lucky?
            # No, we must be safe.
            
            # Efficient check:
            # 1. Compute coverage of 1..n_uncovered-1 with current residues (excluding p? No, including).
            # 2. Compute coverage of 1..n_uncovered-1 with new residue for p.
            # 3. If old coverage == new coverage for 1..n_uncovered-1, it's safe.
            
            # Since n_uncovered is small (likely < 2000), we can just recompute.
            
            # Temporarily change
            old_r = residues[p]
            residues[p] = new_r
            
            # Check 1..n_uncovered-1
            ok = True
            # Rebuild coverage for 1..n_uncovered
            temp_covered = [False] * (n_uncovered + 1)
            for pp in P:
                rr = residues[pp]
                start = rr if rr > 0 else pp
                for nn in range(start, n_uncovered + 1, pp):
                    temp_covered[nn] = True
            
            for nn in range(1, n_uncovered):
                if not temp_covered[nn]:
                    ok = False
                    break
            
            if ok:
                # Accept change
                improved = True
                # Update L? We'll recalculate in next loop.
                break # Try next uncovered number
            else:
                # Revert
                residues[p] = old_r
        
        if not improved:
            # Try random restarts or different prime sets?
            # For now, break.
            break
            
    # Final calculation of L
    L = get_covered_prefix(residues, P, 10000)
    
    # Return classes
    return [[p, residues[p]] for p in P]
