import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Candidate moduli: even numbers in [20, 200]
    moduli = list(range(20, 201, 2))
    
    def is_valid(residues):
        n = len(residues)
        for i in range(n):
            for j in range(i + 1, n):
                g = gcd(moduli[i], moduli[j])
                if residues[i] % g == residues[j] % g:
                    return False
        return True

    def solve(moduli_subset):
        n = len(moduli_subset)
        if n == 0:
            return []
        
        # Initialize residues
        residues = [0] * n
        
        # Greedy initialization: try to set residues to distinct values mod small gcds
        # Just random is fine for local search
        for i in range(n):
            residues[i] = random.randint(0, moduli_subset[i] - 1)
            
        # Local Search
        max_iterations = 10000
        for _ in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Find a conflict
            conflict_found = False
            for i in range(n):
                for j in range(i + 1, n):
                    g = gcd(moduli_subset[i], moduli_subset[j])
                    if residues[i] % g == residues[j] % g:
                        # Conflict found
                        # Try to fix residue i
                        old_r = residues[i]
                        # Try random new residue
                        for _ in range(10):
                            new_r = random.randint(0, moduli_subset[i] - 1)
                            if new_r == old_r:
                                continue
                            # Check if this fixes conflicts with all others?
                            # Or just check if it doesn't create NEW conflicts?
                            # Simpler: just swap and check if total conflicts decreased?
                            # For speed, just try to find a residue for i that is valid with respect to a few neighbors?
                            # Let's just try random values and check full validity? Too slow.
                            # Check validity with respect to all j' != i?
                            # That's O(n) per trial.
                            valid = True
                            for k in range(n):
                                if k == i:
                                    continue
                                g_k = gcd(moduli_subset[i], moduli_subset[k])
                                if new_r % g_k == residues[k] % g_k:
                                    valid = False
                                    break
                            if valid:
                                residues[i] = new_r
                                conflict_found = True
                                break
                        if conflict_found:
                            break
                if conflict_found:
                    break
            
            if not conflict_found:
                return residues
        
        return None

    # Try full set of even moduli
    res = solve(moduli)
    if res:
        return [[moduli[i], res[i]] for i in range(len(moduli))]
        
    # If failed, try multiples of 4
    moduli_4 = list(range(20, 201, 4))
    res = solve(moduli_4)
    if res:
        return [[moduli_4[i], res[i]] for i in range(len(moduli_4))]
        
    # If failed, try multiples of 5
    moduli_5 = list(range(20, 201, 5))
    res = solve(moduli_5)
    if res:
        return [[moduli_5[i], res[i]] for i in range(len(moduli_5))]
        
    # Fallback: Seed solution
    return [[20 * j, j - 1] for j in range(1, 11)]
