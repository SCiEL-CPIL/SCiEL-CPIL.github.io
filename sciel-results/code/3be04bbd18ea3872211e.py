"""Rewritten program: fast deterministic construction of disjoint residue classes."""
# EVOLVE-BLOCK-START
import math
import time

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin
    
    MAX_MOD = 128
    
    # Strategy: Try to construct a large set using multiples of small primes.
    # For a prime P, if we use moduli P, 2P, ..., kP, and residues r_i = i,
    # the condition r_i != r_j mod (P * gcd(i, j)) is satisfied if P * gcd(i, j) > max(i, j).
    # This holds for P=11, k=11 (max i=11, min mod=11).
    # For P=7, k=18, min mod=7. i=1, j=8 -> mod 7. 1 == 8 mod 7. Fails.
    # So P=11 is safe with r_i = i.
    
    # We can do better by trying to solve the CSP for P=2, 3, 5, 7.
    
    def try_construct_for_prime(P, max_mod):
        """
        Try to construct a set of classes using moduli that are multiples of P.
        Returns list of [residue, modulus] or None if failed.
        """
        moduli = [i * P for i in range(1, max_mod // P + 1) if i * P <= max_mod]
        if not moduli:
            return None
        
        # Assign residues greedily or with backtracking
        # For simplicity and speed, we try a few simple patterns first.
        # Pattern 1: r_i = i
        candidates = [[i, i * P] for i in range(1, len(moduli) + 1)]
        
        # Verify pattern 1
        valid = True
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                r1, m1 = candidates[i]
                r2, m2 = candidates[j]
                if not _is_disjoint(r1, m1, r2, m2):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return candidates
        
        # Pattern 2: r_i = 0 for all (usually fails)
        # Pattern 3: Random search with timeout
        # Given time constraints, we rely on P=11 or smaller P with careful assignment.
        # For P=2, 3, 5, 7, we can use a simple backtracking solver.
        
        def solve_csp(moduli_list):
            n = len(moduli_list)
            residues = [None] * n
            # Order moduli by increasing value (tighter constraints first? or larger gcd?)
            # Just use given order
            def backtrack(idx):
                if idx == n:
                    return True
                m = moduli_list[idx]
                for r in range(m):
                    ok = True
                    for j in range(idx):
                        if not _is_disjoint(r, m, residues[j], moduli_list[j]):
                            ok = False
                            break
                    if ok:
                        residues[idx] = r
                        if backtrack(idx + 1):
                            return True
                        residues[idx] = None
                return False
            
            if backtrack(0):
                return [[residues[i], moduli_list[i]] for i in range(n)]
            return None

        # Try backtracking for P=2, 3, 5, 7
        if P in [2, 3, 5, 7]:
            # Limit backtracking depth/time
            # For P=2, 64 moduli. Backtracking might be slow.
            # But we only have 50s budget.
            # Let's try P=3 (42 moduli) first as it's likely solvable.
            res = solve_csp(moduli)
            if res:
                return res
        
        return None

    # Try primes in increasing order of P (larger sets)
    # P=2 (64 moduli), P=3 (42), P=5 (25), P=7 (18), P=11 (11)
    
    best_result = None
    
    for P in [2, 3, 5, 7, 11]:
        if time.time() > deadline:
            break
            
        result = try_construct_for_prime(P, MAX_MOD)
        if result:
            best_result = result
            # We found a valid set. Can we do better?
            # Since P=2 is the largest set, if it works, we are done.
            # If P=2 fails, P=3 is next best.
            # So we break as soon as we find a valid set for the smallest P (largest set).
            # Wait, P=2 is smallest prime, largest set.
            # So if P=2 works, return it.
            # If P=2 fails, try P=3.
            break

    if best_result is None:
        # Fallback: P=11 with r_i=i is guaranteed valid
        best_result = [[i, i * 11] for i in range(1, 12)]
        
    return best_result
# EVOLVE-BLOCK-END
