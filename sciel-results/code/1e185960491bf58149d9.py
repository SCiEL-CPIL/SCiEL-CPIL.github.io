"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    """
    Return coefficient list in ascending degree, length<=41, nonzero final coefficient,
    entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square.
    """
    
    def square_coeffs(p):
        """Compute P^2 coefficients from P's coefficient list."""
        n = len(p)
        # Pad to even length for potential FFT? No, simple loop is fine for n<=41
        result = [0] * (2*n - 1)
        for i in range(n):
            if p[i] == 0:
                continue
            for j in range(i, n):
                if p[j] == 0:
                    continue
                val = p[i] * p[j]
                result[i+j] += val
                if i != j:
                    result[i+j] += val
        return result
    
    def count_nonzero(poly_coeffs):
        """Count nonzero terms in a polynomial given as coefficient list."""
        return sum(1 for c in poly_coeffs if c != 0)
    
    def is_valid(p):
        """Check if p satisfies all constraints."""
        if len(p) > 41:
            return False
        if len(p) == 0 or p[-1] == 0:
            return False
        if any(c < -64 or c > 64 for c in p):
            return False
        if sum(1 for c in p if c != 0) != 12:
            return False
        return True

    best_nz = float('inf')
    best_p = None
    
    # Generate candidate 6-term exponent sets
    candidate_sets = set()
    
    # Consecutive blocks
    for start in range(0, 11):
        candidate_sets.add(frozenset(range(start, start+6)))
        
    # Even spaced blocks
    for start in range(0, 11, 2):
        candidate_sets.add(frozenset(range(start, start+12, 2)))
        
    # Odd spaced blocks
    for start in range(1, 11, 2):
        candidate_sets.add(frozenset(range(start, start+12, 2)))
        
    # Some gapped blocks
    candidate_sets.add(frozenset([0, 1, 2, 5, 6, 7]))
    candidate_sets.add(frozenset([0, 2, 4, 6, 8, 10]))
    candidate_sets.add(frozenset([0, 1, 4, 5, 8, 9]))
    candidate_sets.add(frozenset([0, 3, 6, 9, 12, 15]))
    candidate_sets.add(frozenset([0, 2, 5, 7, 10, 12]))
    candidate_sets.add(frozenset([0, 1, 3, 6, 10, 15]))
    
    candidate_list = list(candidate_sets)
    
    # Search over pairs of sets
    for i, sa in enumerate(candidate_list):
        for sb in candidate_list[i:]: # Symmetry A*B = B*A
            # Compute Minkowski sum
            sums = set()
            for a in sa:
                for b in sb:
                    sums.add(a + b)
            
            if len(sums) < 12:
                continue
            if len(sums) > 25:
                continue
                
            # Sort sums to create index mapping
            sorted_sums = sorted(sums)
            sum_to_idx = {s: idx for idx, s in enumerate(sorted_sums)}
            p_len = sorted_sums[-1] + 1
            
            # Iterate over sign patterns for A and B
            # 2^6 * 2^6 = 4096 patterns
            for mask_a in range(64):
                coeffs_a = [1 if (mask_a >> k) & 1 else -1 for k in range(6)]
                # Map coeffs to exponents in sorted order? 
                # We need to align coeffs with the specific exponents in sa
                # Let's just iterate through sa and assign signs based on index
                a_list = sorted(sa)
                b_list = sorted(sb)
                
                # Construct A and B as dicts or lists
                # A has terms at a_list[k] with coeff coeffs_a[k]
                # B has terms at b_list[k] with coeff coeffs_b[k]
                
                for mask_b in range(64):
                    coeffs_b = [1 if (mask_b >> k) & 1 else -1 for k in range(6)]
                    
                    p = [0] * p_len
                    for k, a_exp in enumerate(a_list):
                        ca = coeffs_a[k]
                        for l, b_exp in enumerate(b_list):
                            cb = coeffs_b[l]
                            idx = a_exp + b_exp
                            p[idx] += ca * cb
                    
                    # Check if P has exactly 12 non-zero terms
                    nz_p = count_nonzero(p)
                    if nz_p != 12:
                        continue
                    
                    # Check validity
                    if not is_valid(p):
                        continue
                        
                    # Compute square
                    sq = square_coeffs(p)
                    nz_sq = count_nonzero(sq)
                    
                    if nz_sq < best_nz:
                        best_nz = nz_sq
                        best_p = p[:]
                        
    # Fallback if no product candidate found or to try direct search if product search fails
    if best_p is None:
        # Simple fallback
        return [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        
    return best_p
# EVOLVE-BLOCK-END
