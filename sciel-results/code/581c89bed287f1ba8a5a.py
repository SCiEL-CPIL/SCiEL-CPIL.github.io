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

def _poly_square_count(coeffs):
    """Return number of nonzero terms in the square of coeffs."""
    sq = _square_coefficients(coeffs)
    return _count_nonzero(sq)

def construct():
    best_square_count = float('inf')
    best_coeffs = None
    
    # Strategy: P = A + B where A and B are 6-term polynomials.
    # If A and B have disjoint support, then P^2 = A^2 + B^2 + 2AB.
    # We want to minimize the total nonzero count.
    # 
    # Key insight: If we choose A and B such that:
    #   1. A and B have disjoint exponent sets
    #   2. The supports of A^2, B^2, and AB are mostly disjoint
    # then we can minimize the total.
    #
    # Let's enumerate: A has 6 terms at positions a_0 < a_1 < ... < a_5
    #                  B has 6 terms at positions b_0 < b_1 < ... < b_5
    # with all 12 positions distinct.
    #
    # For efficiency, we'll try structured choices:
    #   A occupies low degrees, B occupies high degrees, or interleaved.
    
    # First, establish a baseline with simple sign patterns
    current = [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]
    sq = _square_coefficients(current)
    current_count = _count_nonzero(sq)
    best_square_count = current_count
    best_coeffs = current[:]
    
    # Try all 2^11 sign patterns for 12-term ±1 polynomials (fix leading to +1)
    for mask in range(2048):
        coeffs = [1] * 12
        for i in range(11):
            if (mask >> i) & 1:
                coeffs[i] = -1
        sq = _square_coefficients(coeffs)
        count = _count_nonzero(sq)
        if count < best_square_count:
            best_square_count = count
            best_coeffs = coeffs[:]
    
    # Now try the A+B decomposition approach.
    # A = sum of 6 terms, B = sum of 6 terms, disjoint supports.
    # We'll try to place A in low degrees and B in high degrees,
    # or use other structured placements.
    
    # Helper: given support positions and signs, build the full 12-term poly
    def _build_from_parts(a_pos, a_signs, b_pos, b_signs):
        """Build a 12-term polynomial from A and B parts."""
        max_deg = max(max(a_pos) if a_pos else 0, max(b_pos) if b_pos else 0)
        if max_deg >= 41:
            return None
        coeffs = [0] * (max_deg + 1)
        for p, s in zip(a_pos, a_signs):
            coeffs[p] += s
        for p, s in zip(b_pos, b_signs):
            coeffs[p] += s
        # Check exactly 12 nonzero terms
        if _count_nonzero(coeffs) != 12:
            return None
        # Check coefficient bounds
        for c in coeffs:
            if c < -64 or c > 64:
                return None
        # Trim trailing zeros but keep length <= 41
        while len(coeffs) > 0 and coeffs[-1] == 0:
            coeffs.pop()
        if len(coeffs) == 0 or coeffs[-1] == 0:
            return None
        return coeffs
    
    # Try various structured placements of A and B
    # Pattern 1: A in degrees 0-5, B in degrees 6-11
    # Try all sign patterns for A (2^6=64) and B (2^6=64), fix leading of B to +1
    a_pos_1 = [0, 1, 2, 3, 4, 5]
    b_pos_1 = [6, 7, 8, 9, 10, 11]
    
    for a_mask in range(64):
        a_signs = []
        for i in range(6):
            if (a_mask >> i) & 1:
                a_signs.append(-1)
            else:
                a_signs.append(1)
        for b_mask in range(32):  # fix b_signs[5] = 1
            b_signs = []
            for i in range(5):
                if (b_mask >> i) & 1:
                    b_signs.append(-1)
                else:
                    b_signs.append(1)
            b_signs.append(1)
        
        coeffs = _build_from_parts(a_pos_1, a_signs, b_pos_1, b_signs)
        if coeffs is None:
            continue
        sq = _square_coefficients(coeffs)
        count = _count_nonzero(sq)
        if count < best_square_count:
            best_square_count = count
            best_coeffs = coeffs[:]
    
    # Pattern 2: A in even degrees 0,2,4,6,8,10; B in odd degrees 1,3,5,7,9,11
    a_pos_2 = [0, 2, 4, 6, 8, 10]
    b_pos_2 = [1, 3, 5, 7, 9, 11]
    
    for a_mask in range(64):
        a_signs = []
        for i in range(6):
            if (a_mask >> i) & 1:
                a_signs.append(-1)
            else:
                a_signs.append(1)
        for b_mask in range(32):
            b_signs = []
            for i in range(5):
                if (b_mask >> i) & 1:
                    b_signs.append(-1)
                else:
                    b_signs.append(1)
            b_signs.append(1)
        
        coeffs = _build_from_parts(a_pos_2, a_signs, b_pos_2, b_signs)
        if coeffs is None:
            continue
        sq = _square_coefficients(coeffs)
        count = _count_nonzero(sq)
        if count < best_square_count:
            best_square_count = count
            best_coeffs = coeffs[:]
    
    # Pattern 3: Interleaved - A at 0,3,6,9,12,15; B at 1,4,7,10,13,16
    # This gives more separation between A^2, B^2, and AB supports
    a_pos_3 = [0, 3, 6, 9, 12, 15]
    b_pos_3 = [1, 4, 7, 10, 13, 16]
    
    for a_mask in range(64):
        a_signs = []
        for i in range(6):
            if (a_mask >> i) & 1:
                a_signs.append(-1)
            else:
                a_signs.append(1)
        for b_mask in range(32):
            b_signs = []
            for i in range(5):
                if (b_mask >> i) & 1:
                    b_signs.append(-1)
                else:
                    b_signs.append(1)
            b_signs.append(1)
        
        coeffs = _build_from_parts(a_pos_3, a_signs, b_pos_3, b_signs)
        if coeffs is None:
            continue
        sq = _square_coefficients(coeffs)
        count = _count_nonzero(sq)
        if count < best_square_count:
            best_square_count = count
            best_coeffs = coeffs[:]
    
    # Pattern 4: A at 0,2,4,6,8,10; B at 1,3,5,7,9,11 but with larger gaps
    # Actually let's try A at 0,4,8,12,16,20; B at 1,5,9,13,17,21
    a_pos_4 = [0, 4, 8, 12, 16, 20]
    b_pos_4 = [1, 5, 9, 13, 17, 21]
    
    for a_mask in range(64):
        a_signs = []
        for i in range(6):
            if (a_mask >> i) & 1:
                a_signs.append(-1)
            else:
                a_signs.append(1)
        for b_mask in range(32):
            b_signs = []
            for i in range(5):
                if (b_mask >> i) & 1:
                    b_signs.append(-1)
                else:
                    b_signs.append(1)
            b_signs.append(1)
        
        coeffs = _build_from_parts(a_pos_4, a_signs, b_pos_4, b_signs)
        if coeffs is None:
            continue
        sq = _square_coefficients(coeffs)
        count = _count_nonzero(sq)
        if count < best_square_count:
            best_square_count = count
            best_coeffs = coeffs[:]
    
    # Pattern 5: Try to make A^2 and B^2 have minimal nonzero terms independently.
    # A 6-term polynomial whose square has few nonzeros: try geometric-like patterns.
    # A = 1 + x + x^2 + x^3 + x^4 + x^5 has square with 11 nonzeros (all 1s).
    # A = 1 - x + x^2 - x^3 + x^4 - x^5: let's compute.
    
    # Let's try more structured sign patterns for the interleaved case.
    # For pattern 2 (even/odd), try alternating signs within A and B.
    a_pos_5 = [0, 2, 4, 6, 8, 10]
    b_pos_5 = [1, 3, 5, 7, 9, 11]
    
    # Try specific structured sign patterns
    structured_signs = [
        [1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1],
        [1, -1, -1, 1, 1, -1],
        [1, 1, 1, -1, -1, -1],
        [1, -1, 1, 1, -1, 1],
    ]
    
    for a_signs in structured_signs:
        for b_signs in structured_signs:
            # Fix leading sign of B to +1 for uniqueness
            if b_signs[5] != 1:
                b_signs = [s if i < 5 else 1 for i, s in enumerate(b_signs)]
            coeffs = _build_from_parts(a_pos_5, a_signs, b_pos_5, b_signs)
            if coeffs is None:
                continue
            sq = _square_coefficients(coeffs)
            count = _count_nonzero(sq)
            if count < best_square_count:
                best_square_count = count
                best_coeffs = coeffs[:]
    
    # Also try with coefficients ±2 in some positions to see if it helps
    # Start from best_coeffs and try modifying individual coefficients
    for pos in range(len(best_coeffs)):
        for val in [-2, 2, -3, 3]:
            if pos == len(best_coeffs) - 1 and val == 0:
                continue
            new_coeffs = best_coeffs[:]
            new_coeffs[pos] = val
            if _valid_candidate(new_coeffs):
                sq = _square_coefficients(new_coeffs)
                count = _count_nonzero(sq)
                if count < best_square_count:
                    best_square_count = count
                    best_coeffs = new_coeffs[:]
    
    # Try a few more hand-crafted patterns
    # P(x) = (1+x+x^2+x^3+x^4+x^5) + x^6(1-x+x^2-x^3+x^4-x^5)
    alt5 = [1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1]
    sq = _square_coefficients(alt5)
    count = _count_nonzero(sq)
    if count < best_square_count:
        best_square_count = count
        best_coeffs = alt5[:]
    
    # P(x) = (1+x+x^2+x^3+x^4+x^5) - x^6(1+x+x^2+x^3+x^4+x^5)
    alt6 = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    sq = _square_coefficients(alt6)
    count = _count_nonzero(sq)
    if count < best_square_count:
        best_square_count = count
        best_coeffs = alt6[:]
    
    # P(x) = (1-x+x^2-x^3+x^4-x^5) + x^6(1-x+x^2-x^3+x^4-x^5)
    alt7 = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    sq = _square_coefficients(alt7)
    count = _count_nonzero(sq)
    if count < best_square_count:
        best_square_count = count
        best_coeffs = alt7[:]
    
    # P(x) = (1-x+x^2-x^3+x^4-x^5) - x^6(1-x+x^2-x^3+x^4-x^5)
    alt8 = [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1]
    sq = _square_coefficients(alt8)
    count = _count_nonzero(sq)
    if count < best_square_count:
        best_square_count = count
        best_coeffs = alt8[:]
    
    return best_coeffs
