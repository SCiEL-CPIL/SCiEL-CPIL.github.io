import itertools
import numpy as np

def _square_coefficients(coeffs):
    """Compute exact integer square of polynomial given by coefficient list (ascending degree)."""
    n = len(coeffs)
    # Trim trailing zeros for efficiency
    while n > 0 and coeffs[n-1] == 0:
        n -= 1
    if n == 0:
        return [0]
    result = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(n):
            if coeffs[j] == 0:
                continue
            result[i + j] += coeffs[i] * coeffs[j]
    # Trim trailing zeros
    while len(result) > 0 and result[-1] == 0:
        result.pop()
    if not result:
        result = [0]
    return result

def _count_nonzero(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _valid_candidate(coeffs):
    """Check all constraints for a candidate polynomial."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    for c in coeffs:
        if not isinstance(c, int) or c < -64 or c > 64:
            return False
    if _count_nonzero(coeffs) != 12:
        return False
    return True

def construct():
    best_square_count = float('inf')
    best_coeffs = None
    
    # Strategy: P(x) = Q(x) + x^k Q(x) where Q has 6 nonzero terms.
    # This guarantees 12 nonzero terms in P if supports are disjoint.
    # We search for Q with 6 terms (coeffs +/-1) and shift k.
    
    # 1. Iterate over all possible 6-term base polynomials Q
    # To keep it fast, we restrict Q to have support in {0, ..., M}
    # Let's try supports of size 6 chosen from {0, ..., 10}
    # Number of subsets: C(11, 6) = 462
    # For each subset, 2^5 = 32 sign patterns (fix first coeff to 1)
    # Total Q candidates: 462 * 32 = 14784. This is very fast.
    
    base_indices = list(range(11)) # 0 to 10
    
    for subset in itertools.combinations(base_indices, 6):
        # Try all sign patterns for these 6 terms
        # Fix the sign of the term at the highest exponent in subset to +1 to avoid redundancy
        # But for Q, we just need to generate the list of coeffs for Q
        # We will construct Q as a list of length max(subset)+1
        
        # To avoid generating 32 lists for each subset, we can iterate masks
        for mask in range(32):
            q_coeffs = [0] * (max(subset) + 1)
            # Assign signs
            # We need to map mask bits to the 6 positions
            # Let's just assign signs to the 6 terms
            signs = []
            for i in range(6):
                if (mask >> i) & 1:
                    signs.append(-1)
                else:
                    signs.append(1)
            
            # Fix the last term (highest degree) to +1 to reduce redundancy?
            # Actually, Q and -Q give same Q^2. So we can fix one sign.
            # Let's fix the sign of the term at index max(subset) to +1.
            # Find its position in the subset
            max_idx = max(subset)
            pos_in_subset = subset.index(max_idx)
            signs[pos_in_subset] = 1
            
            for idx, s in zip(subset, signs):
                q_coeffs[idx] = s
            
            # Check if Q has exactly 6 nonzero terms (it does by construction)
            # Compute Q^2
            sq_q = _square_coefficients(q_coeffs)
            count_q_sq = _count_nonzero(sq_q)
            
            # If Q^2 is already very sparse, it's a good candidate for P.
            # P = Q + x^k Q
            # We need to try different shifts k.
            # P will have degree max(subset) + k.
            # Constraint: degree(P) <= 40.
            # So max(subset) + k <= 40 => k <= 40 - max(subset).
            # Also, for P to have 12 terms, supports of Q and x^k Q must be disjoint.
            # Support of Q: subset. Support of x^k Q: {s+k for s in subset}.
            # Disjoint if (subset) & (subset+k) == empty set.
            
            max_s = max(subset)
            min_s = min(subset)
            
            # Max possible k
            max_k = 40 - max_s
            
            for k in range(1, max_k + 1):
                # Check disjointness
                # If k > max_s - min_s, they are definitely disjoint?
                # No, if k is small, they overlap.
                # Overlap exists if there are s1, s2 in subset such that s1 = s2 + k.
                # i.e., s1 - s2 = k.
                # We can check this quickly.
                overlap = False
                for s1 in subset:
                    if s1 - k in subset:
                        overlap = True
                        break
                
                if overlap:
                    continue
                
                # Construct P
                p_coeffs = [0] * (max_s + k + 1)
                for idx, s in zip(subset, signs):
                    p_coeffs[idx] = s
                    p_coeffs[idx + k] = s
                
                # Verify P has 12 nonzero terms
                if _count_nonzero(p_coeffs) != 12:
                    continue
                    
                # Compute P^2
                sq_p = _square_coefficients(p_coeffs)
                count_p_sq = _count_nonzero(sq_p)
                
                if count_p_sq < best_square_count:
                    best_square_count = count_p_sq
                    best_coeffs = p_coeffs[:]
    
    # Fallback if no valid P found (should not happen)
    if best_coeffs is None:
        # Return a valid 12-term polynomial, e.g., 1 + x + ... + x^11
        best_coeffs = [1] * 12
        best_square_count = _count_nonzero(_square_coefficients(best_coeffs))
        
    return best_coeffs
