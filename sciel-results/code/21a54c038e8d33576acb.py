"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
    best_score = 23
    best_poly = [1] * 12

    def square_nonzero_count(poly):
        """
        Computes the number of non-zero coefficients in the square of a polynomial.
        poly is a list of coefficients in ascending degree.
        """
        n = len(poly)
        if n == 0:
            return 0
        # Determine max degree
        max_deg = n - 1
        # Square length is 2 * max_deg + 1
        sq = [0] * (2 * max_deg + 1)
        
        # Optimize by only iterating over non-zero terms
        nz_indices = [i for i, c in enumerate(poly) if c != 0]
        
        for i in nz_indices:
            ci = poly[i]
            for j in nz_indices:
                sq[i + j] += ci * poly[j]
                
        return sum(1 for c in sq if c != 0)

    def try_solve_signs(exponents, steps=50000):
        """
        Given a set of exponents, tries to find signs (+1/-1) such that
        the square has a minimal number of non-zero terms.
        Returns (score, poly_list)
        """
        k = len(exponents)
        # Initialize with all +1
        signs = [1] * k
        
        # Helper to compute score from signs
        def get_score(s):
            # Construct polynomial
            p = [0] * (max(exponents) + 1)
            for e, sgn in zip(exponents, s):
                p[e] = sgn
            return square_nonzero_count(p)
        
        curr_score = get_score(signs)
        if curr_score <= 12: # Theoretical minimum (perfect square)
            return curr_score, [0] * (max(exponents) + 1)
            
        # Randomized local search
        for _ in range(steps):
            if time.time() > deadline:
                break
            idx = random.randint(0, k - 1)
            signs[idx] *= -1
            new_score = get_score(signs)
            
            if new_score <= curr_score:
                curr_score = new_score
                if curr_score <= 12:
                    break
            else:
                # Simulated annealing style: accept worse with small prob?
                # For now, just revert if worse to keep it simple and fast
                signs[idx] *= -1
        
        # Construct final poly
        p = [0] * (max(exponents) + 1)
        for e, sgn in zip(exponents, signs):
            p[e] = sgn
        return curr_score, p

    # Strategy 1: Test the known optimal Golay set for 12 terms
    # This set is known to yield a square with very few non-zero terms (likely 12 or 13)
    # Exponents: 0, 1, 2, 4, 7, 8, 11, 12, 14, 16, 19, 20
    # This corresponds to the binary Golay code structure or similar.
    golay_exponents = [0, 1, 2, 4, 7, 8, 11, 12, 14, 16, 19, 20]
    score, poly = try_solve_signs(golay_exponents, steps=200000)
    if score < best_score:
        best_score = score
        best_poly = poly

    # Strategy 2: Search for other "self-orthogonal" or "low-correlation" sets
    # We look for sets where the difference multiset has low multiplicity.
    # If all differences (a-b) for a!=b are distinct, the square has 12*2-1 = 23 terms? No.
    # If the set is a "Sidon set", the sums i+j are distinct except for i=j and i=j swaps.
    # Actually, if the set is a Sidon set (B2-sequence), the sums i+j are unique for pairs {i,j} with i<=j.
    # For a Sidon set of size 12, the number of distinct sums is 12 (diagonal) + C(12,2) = 12 + 66 = 78? No.
    # Wait. The square of sum x^a is sum x^(a+b).
    # Coeff of x^k is sum_{a+b=k} s_a s_b.
    # If the set is a Sidon set, then for k = a+a, only (a,a) contributes -> s_a^2 = 1. (12 terms)
    # For k = a+b (a!=b), only (a,b) and (b,a) contribute -> 2 s_a s_b.
    # If we can choose signs such that 2 s_a s_b = 0? Impossible for integers.
    # So Sidon sets give 12 + 66 = 78 non-zero terms. That's worse than 23.
    
    # We want Cancellations.
    # Cancellations happen when multiple pairs (a,b) sum to the same k.
    # We want s_a s_b + s_c s_d + ... = 0.
    
    # The Golay set above has high symmetry. Let's try a few other structured sets.
    # Sets related to binary codes often work well.
    
    # Another candidate: Powers of 2 (no cancellations, 23 terms) - already covered by seed.
    
    # Let's try a few random sets that look "structured" or just random sets with small span.
    # But random sets are unlikely to beat Golay.
    
    # Let's try to optimize the Golay set signs more aggressively if time permits.
    if time.time() < deadline - 10:
        # Try different starting points for the Golay set
        for _ in range(10):
            random.shuffle(golay_exponents) # Doesn't change set, but maybe restart logic? No.
            # Just run more steps
            score2, poly2 = try_solve_signs(golay_exponents, steps=100000)
            if score2 < best_score:
                best_score = score2
                best_poly = poly2

    # Try a different family: Exponents from a linear recurrence or specific pattern?
    # Let's try a set that is a "perfect difference set" if one exists for 12?
    # Perfect difference sets exist for q^2+q+1. 12 is not of that form.
    
    # Let's try a heuristic search for exponents in [0, 30]
    # We want high collision count for sums, but signs must cancel them.
    
    # Given the time constraint, the Golay set is the strongest candidate.
    # The seed is 23. The Golay set is expected to be ~12-16.
    
    # Ensure final poly is valid
    if best_poly[-1] == 0:
         # Find last non-zero
         while best_poly and best_poly[-1] == 0:
             best_poly.pop()
         if not best_poly:
             best_poly = [1]
    
    # Ensure length <= 41
    if len(best_poly) > 41:
        # This shouldn't happen with our exponents, but just in case
        # Truncation would change the polynomial, so we shouldn't do it.
        # But the task says length <= 41. Our max exponent is 20, so length 21.
        pass
        
    # Ensure entries in [-64, 64]
    # Our entries are +/- 1, so this is fine.
    
    # Ensure exactly 12 non-zero terms
    nz_count = sum(1 for c in best_poly if c != 0)
    if nz_count != 12:
        # Fallback to seed if something went wrong
        return [1] * 12
        
    return best_poly
# EVOLVE-BLOCK-END
