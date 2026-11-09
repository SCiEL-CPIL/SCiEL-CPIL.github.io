"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    """
    Finds a 12-term integer polynomial whose square has the minimum number of non-zero terms.
    Uses a SAT-based backtracking search to determine if a square with K non-zero terms exists.
    """
    deadline = time.time() + 80.0  # Safety margin
    
    # Theoretical minimum for a square of a 12-term polynomial is 12 (if it's a perfect square of a 6-term poly).
    # However, the problem asks for a 12-term poly P, not P^2 = Q^2 where Q has 6 terms.
    # Wait, if P has 12 terms, P^2 can have as few as 12 terms if P is a perfect square of a 6-term polynomial.
    # Let's check if a 12-term polynomial can be a perfect square of a 6-term polynomial with integer coefficients.
    # If P = Q^2, and Q has 6 terms, then P has 12 terms? No, Q^2 has up to 11 terms if Q has 6 terms?
    # If Q = sum_{i=0}^{5} c_i x^{e_i}, Q^2 = sum c_i^2 x^{2e_i} + sum_{i<j} 2 c_i c_j x^{e_i+e_j}.
    # For P to have exactly 12 terms, all these sums must be distinct and non-zero.
    # This requires the set of exponents {2e_i} and {e_i+e_j} to be disjoint and all coefficients non-zero.
    # This is possible. For example, if exponents are powers of 2: 0, 1, 2, 4, 8, 16.
    # 2e: 0, 2, 4, 8, 16, 32
    # e_i+e_j: 1, 2, 4, 8, 16, 3, 5, 9, 6, 10, 12...
    # Collisions: 2 (from 2*1 and 0+2), 4 (from 2*2 and 0+4), etc.
    # So powers of 2 don't work for perfect square sparsity.
    
    # We will search for a polynomial P with 12 non-zero terms such that P^2 has K non-zero terms.
    # We start with K=12 and go up.
    
    def solve_sat(K_target, max_deg=40):
        """
        Tries to find a set of 12 exponents and signs such that P^2 has exactly K_target non-zero terms.
        Uses backtracking on the exponents.
        """
        # We fix the first exponent to 0 (shift invariance).
        # We need to choose 11 more exponents from [1, max_deg].
        # To make search tractable, we impose an ordering: 0 = e_0 < e_1 < ... < e_11 <= max_deg.
        
        # State: current list of exponents, current list of signs (initially all +1, we can fix signs later or include in search)
        # Actually, for a given set of exponents, the signs can be optimized separately or included.
        # Including signs in the SAT search over exponents is complex. 
        # Better: Search for exponents that *allow* a solution with K non-zero terms.
        # For a fixed set of exponents, we can check if there exist signs such that the number of non-zero terms in the square is <= K_target.
        # This is a sub-problem.
        
        # Let's simplify: We will search for exponents AND signs simultaneously using a constraint solver approach.
        # Variables: e_1, ..., e_11 (integers > 0, strictly increasing) and s_0, ..., s_11 (+1/-1).
        # Constraints: The number of indices k such that (sum_{i+j=k} s_i s_j) != 0 is <= K_target.
        
        # This is hard to propagate.
        # Alternative: Iterate over small spans and use random restarts with a strict "accept only if improvement" local search, but with a much better heuristic for the *exponents* as well, not just signs.
        
        # Let's try a different structural approach:
        # Look for polynomials of the form P(x) = A(x) + x^d B(x) where A and B are 6-term polynomials.
        # This doesn't guarantee sparsity.
        
        # Let's stick to the most promising lead from inspirations: 
        # "Parametric algebraic decomposition" or "structured candidates".
        # The inspiration 1 (score 1.4375) likely found a specific pattern.
        # A common structure for sparse squares is related to "Golay sequences" or "Costas arrays".
        
        # Let's try to find a set of 12 exponents that forms a "Sidon set" variant where sums collide in a way that allows cancellation.
        # If we can make P(x) = (1 + x + ... + x^5)^2 / something? No.
        
        # Let's try a specific known construction:
        # If P(x) = (1 + x)^6, it has 7 terms.
        # If P(x) = (1 + x^2)^6, it has 7 terms.
        
        # Let's use a randomized local search that optimizes BOTH exponents and signs.
        # Representation: List of 12 exponents (sorted) and 12 signs.
        # Move 1: Change a sign.
        # Move 2: Shift an exponent by +/- 1 (maintaining order).
        # Move 3: Swap exponents? (Redundant if sorted).
        
        best_score = 23
        best_exponents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        best_signs = [1] * 12
        
        def get_score(exponents, signs):
            # Calculate number of non-zero terms in square
            # Exponents are sorted
            sq_counts = {}
            for i in range(12):
                ei = exponents[i]
                si = signs[i]
                for j in range(12):
                    ej = exponents[j]
                    sj = signs[j]
                    k = ei + ej
                    sq_counts[k] = sq_counts.get(k, 0) + si * sj
            return sum(1 for v in sq_counts.values() if v != 0)
        
        curr_exponents = list(best_exponents)
        curr_signs = list(best_signs)
        curr_score = get_score(curr_exponents, curr_signs)
        
        # Initialize with a "spread out" set to avoid trivial collisions? 
        # Or start with compact? Compact allows more collisions.
        # Let's start with compact [0..11].
        
        import random
        random.seed(42)
        
        steps = 0
        while time.time() < deadline:
            steps += 1
            if steps % 10000 == 0 and time.time() > deadline - 5:
                break
                
            # Choose a move type
            if random.random() < 0.7:
                # Sign flip
                idx = random.randint(0, 11)
                old_sign = curr_signs[idx]
                curr_signs[idx] *= -1
                new_score = get_score(curr_exponents, curr_signs)
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_exponents = list(curr_exponents)
                        best_signs = list(curr_signs)
                        if best_score <= 12:
                            break
                else:
                    # Revert if worse (or accept with small prob for annealing)
                    if random.random() > 0.05:
                        curr_signs[idx] = old_sign
                    else:
                        curr_score = new_score
            else:
                # Exponent shift
                # Pick an exponent to move
                idx = random.randint(1, 11) # Don't move e_0=0
                direction = random.choice([-1, 1])
                old_exp = curr_exponents[idx]
                new_exp = old_exp + direction
                
                # Check validity: must maintain sorted order and non-negative
                if new_exp < 0:
                    continue
                if idx > 0 and new_exp <= curr_exponents[idx-1]:
                    continue
                if idx < 11 and new_exp >= curr_exponents[idx+1]:
                    continue
                if new_exp > 40: # Max degree constraint for P is 40
                    continue
                    
                curr_exponents[idx] = new_exp
                new_score = get_score(curr_exponents, curr_signs)
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_exponents = list(curr_exponents)
                        best_signs = list(curr_signs)
                        if best_score <= 12:
                            break
                else:
                    if random.random() > 0.05:
                        curr_exponents[idx] = old_exp
                    else:
                        curr_score = new_score
            
            # Occasionally restart from best
            if steps % 5000 == 0:
                curr_exponents = list(best_exponents)
                curr_signs = list(best_signs)
                curr_score = best_score
                # Perturb
                for _ in range(5):
                    idx = random.randint(0, 11)
                    curr_signs[idx] *= -1
                    idx2 = random.randint(1, 11)
                    curr_exponents[idx2] += random.choice([-1, 1])
                    # Fix sorting if needed (simple fix: sort and reassign signs? No, signs are tied to exponents)
                    # For simplicity in restart, just use best as base.
                curr_exponents = list(best_exponents)
                curr_signs = list(best_signs)
                curr_score = best_score
                # Small random perturbation
                if random.random() < 0.5:
                    idx = random.randint(0, 11)
                    curr_signs[idx] *= -1
                else:
                    idx = random.randint(1, 11)
                    d = random.choice([-1, 1])
                    if curr_exponents[idx] + d > curr_exponents[idx-1] and (idx == 11 or curr_exponents[idx] + d < curr_exponents[idx+1]):
                        curr_exponents[idx] += d

        # Construct final polynomial
        poly = [0] * 41
        for e, s in zip(best_exponents, best_signs):
            if 0 <= e < 41:
                poly[e] = s
        
        # Trim trailing zeros
        while poly and poly[-1] == 0:
            poly.pop()
            
        if not poly:
            return [1] * 12
            
        # Verify constraints
        if len(poly) > 41:
            return [1] * 12
        if poly[-1] == 0:
            return [1] * 12
        nz_count = sum(1 for c in poly if c != 0)
        if nz_count != 12:
            return [1] * 12
        for c in poly:
            if c < -64 or c > 64:
                return [1] * 12
                
        return poly

# EVOLVE-BLOCK-END
