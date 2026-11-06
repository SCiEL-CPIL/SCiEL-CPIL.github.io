import random
import math
import time

def construct():
    # Target: maximize number of classes
    # Moduli must be distinct in [2, 128]
    # Any pair must have gcd > 1 to be potentially disjoint
    # Largest clique of gcd>1 is multiples of 2: {2, 4, ..., 128} -> 63 moduli
    
    moduli = [m for m in range(2, 129) if m % 2 == 0]
    n = len(moduli)
    
    # We want to find residues r[0..n-1] such that
    # for all i != j, r[i] != r[j] mod gcd(moduli[i], moduli[j])
    
    # Strategy: Randomized greedy with backtracking or just random restarts
    # Given the loose constraints (gcd >= 2), random assignment has high success rate
    
    def try_solve():
        # Order moduli by descending gcd with others? Or just random?
        # Let's try random order
        indices = list(range(n))
        random.shuffle(indices)
        
        assigned = {}
        
        for idx in indices:
            m = moduli[idx]
            # Try random residues
            # Residue must be in [0, m-1]
            # Constraint: r != assigned[j] mod gcd(m, moduli[j]) for all j in assigned
            # Let's just pick a random residue and check
            found = False
            for _ in range(1000): # Try 1000 random residues for this modulus
                r = random.randint(0, m - 1)
                valid = True
                for j, rj in assigned.items():
                    g = math.gcd(m, moduli[j])
                    if r % g == rj % g:
                        valid = False
                        break
                if valid:
                    assigned[idx] = r
                    found = True
                    break
            if not found:
                return None
        return [assigned[i] for i in range(n)]

    # Try multiple times
    start_time = time.time()
    best_result = None
    
    # The seed program returned 7 classes. We aim for 63.
    # A single random run is very likely to succeed for 63.
    # Let's do a few attempts to be safe.
    
    for _ in range(10):
        if time.time() - start_time > 50:
            break
        res = try_solve()
        if res:
            # Verify
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    g = math.gcd(moduli[i], moduli[j])
                    if res[i] % g == res[j] % g:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                # Format as [residue, modulus]
                result = [[res[i], moduli[i]] for i in range(n)]
                # Ensure canonical residues
                for pair in result:
                    pair[0] = pair[0] % pair[1]
                return result

    # Fallback: return the seed result if we fail (should not happen)
    return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
