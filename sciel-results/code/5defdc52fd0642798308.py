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
    # We'll search for a 12-term polynomial whose square has minimal nonzero terms.
    # Strategy: Try structured polynomials with signs that encourage cancellation.
    
    best_square_count = float('inf')
    best_coeffs = None
    
    # The current best is 16 nonzero terms in the square (score 1.0952 implies base is ~15.6, but let's verify)
    # Actually, let's first compute the square of the current program's output to establish a baseline.
    current = [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]
    sq = _square_coefficients(current)
    current_count = _count_nonzero(sq)
    best_square_count = current_count
    best_coeffs = current
    
    # Try all sign patterns for the 12-term polynomial with coefficients ±1
    # There are 2^12 = 4096 such patterns. For each, compute the square and count nonzeros.
    # We fix the leading coefficient to +1 to avoid redundancy (sign flip gives same square).
    # So we only need 2^11 = 2048 patterns.
    
    for mask in range(2048):
        coeffs = [1] * 12
        for i in range(11):  # indices 0-10, index 11 stays +1
            if (mask >> i) & 1:
                coeffs[i] = -1
        sq = _square_coefficients(coeffs)
        count = _count_nonzero(sq)
        if count < best_square_count:
            best_square_count = count
            best_coeffs = coeffs[:]
    
    # Also try some polynomials with coefficients ±2 to see if they help
    # But the constraint is entries in [-64,64], so ±2 is allowed.
    # However, this increases the search space significantly. Let's try a few structured ones.
    
    # Try polynomials of the form sum_{i=0}^{5} a_i x^i + sum_{i=6}^{11} b_i x^i
    # with a_i, b_i in {-1, 1}
    # This is already covered above.
    
    # Try polynomials where some coefficients are ±2
    # For example, try replacing one ±1 with ±2
    for pos in range(12):
        for val in [-2, 2]:
            coeffs = best_coeffs[:]
            coeffs[pos] = val
            if _valid_candidate(coeffs):
                sq = _square_coefficients(coeffs)
                count = _count_nonzero(sq)
                if count < best_square_count:
                    best_square_count = count
                    best_coeffs = coeffs[:]
    
    # Try a few more structured candidates
    # P(x) = 1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^10 - x^11
    # This is (1-x^12)/(1-x), square is (1-x^12)^2/(1-x)^2
    # Let's just compute it.
    alt = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    sq = _square_coefficients(alt)
    count = _count_nonzero(sq)
    if count < best_square_count:
        best_square_count = count
        best_coeffs = alt[:]
    
    # P(x) = 1 + x - x^2 - x^3 + x^4 + x^5 - x^6 - x^7 + x^8 + x^9 - x^10 - x^11
    # Block pattern: ++--++--++--
    alt2 = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
    sq = _square_coefficients(alt2)
    count = _count_nonzero(sq)
    if count < best_square_count:
        best_square_count = count
        best_coeffs = alt2[:]
    
    # P(x) = 1 - x - x^2 + x^3 + x^4 - x^5 - x^6 + x^7 + x^8 - x^9 - x^10 + x^11
    # Block pattern: +--+--++--++
    alt3 = [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1]
    sq = _square_coefficients(alt3)
    count = _count_nonzero(sq)
    if count < best_square_count:
        best_square_count = count
        best_coeffs = alt3[:]
    
    # P(x) = 1 + x + x^2 - x^3 - x^4 - x^5 + x^6 + x^7 + x^8 - x^9 - x^10 - x^11
    # Block pattern: +++---+++---
    alt4 = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
    sq = _square_coefficients(alt4)
    count = _count_nonzero(sq)
    if count < best_square_count:
        best_square_count = count
        best_coeffs = alt4[:]
    
    return best_coeffs
