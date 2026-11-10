import itertools
from collections import defaultdict

def construct():
    """
    Find a 12-term integer polynomial (coeffs in [-64,64], degree <= 40)
    whose exact square has the minimum number of nonzero terms.
    
    Uses backtracking with constraint propagation on the square's coefficients.
    """
    
    # We'll search for polynomials with exactly 12 nonzero terms.
    # The square has degree up to 2*max_degree.
    # We want to minimize nonzero terms in the square.
    
    best_score = 21  # Current best from seed
    best_poly = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    
    def count_nonzeros(poly):
        return sum(1 for c in poly if c != 0)
    
    def compute_square(coeffs):
        """Compute the exact square of a polynomial given as coefficient list."""
        n = len(coeffs)
        square = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(i, n):
                if coeffs[j] == 0:
                    continue
                val = coeffs[i] * coeffs[j]
                if i == j:
                    square[i + j] += val
                else:
                    square[i + j] += 2 * val
        return square
    
    def check_constraints(coeffs):
        """Check if coeffs form a valid 12-term polynomial."""
        # Exactly 12 nonzero terms
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        # All entries in [-64, 64]
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        # Nonzero final coefficient
        if coeffs[-1] == 0:
            return False
        # Length <= 41
        if len(coeffs) > 41:
            return False
        return True
    
    # Strategy: Try small coefficient values and specific patterns
    # that are known to produce cancellations.
    
    # Pattern 1: Antisymmetric polynomials P(x) = A(x) - x^k A(x)
    # These naturally create cancellations in the square.
    
    def search_antisymmetric():
        """Search for antisymmetric patterns with good cancellation."""
        best = (21, [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1])
        
        # Try different split points and coefficient patterns
        for half_len in range(4, 9):
            for shift in range(half_len, min(21, 41 - half_len + 1)):
                # Generate all possible 6-term halves with small coefficients
                # We need exactly 6 nonzero terms in the half for 12 total
                if half_len != 6:
                    continue
                    
                # Try coefficient combinations
                coeff_range = range(-3, 4)
                for combo in itertools.product(coeff_range, repeat=6):
                    if 0 in combo:
                        continue
                    if any(abs(c) > 64 for c in combo):
                        continue
                    # Create the antisymmetric polynomial
                    poly = list(combo) + [-c for c in combo]
                    # Pad or truncate to ensure proper structure
                    # The second half starts at position shift
                    full_poly = [0] * (shift + 6)
                    for i, c in enumerate(combo):
                        full_poly[i] += c
                        full_poly[shift + i] += -c
                    
                    # Check constraints
                    if not check_constraints(full_poly):
                        continue
                    
                    # Compute square and count nonzeros
                    sq = compute_square(full_poly)
                    score = count_nonzeros(sq)
                    
                    if score < best[0]:
                        best = (score, full_poly)
                        if score <= 18:
                            return best
        
        return best
    
    # Pattern 2: Try polynomials with specific symmetric structures
    def search_symmetric_structures():
        """Search for polynomials with structured coefficients."""
        best = (21, [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1])
        
        # Try polynomials of form sum a_i x^{i} where exponents are chosen
        # to maximize cancellation in the square
        
        # Small coefficient search with pruning
        coeff_values = [-2, -1, 1, 2]
        
        # Try degree up to 15 (16 coefficients)
        for deg in range(10, 16):
            n = deg + 1
            # We need exactly 12 nonzero terms out of n
            if n < 12:
                continue
            if n > 41:
                break
            
            # Choose which 12 positions to use
            # To limit search, try consecutive blocks or specific patterns
            # Try: first 12 positions, or alternating, etc.
            
            # Simplified: try all subsets of size 12 from first 20 positions
            # This is still large, so use heuristic: try blocks
            for start in range(0, 20 - 12 + 1):
                positions = list(range(start, start + 12))
                
                # Try small coefficient assignments
                for coeffs in itertools.product(coeff_values, repeat=12):
                    if any(c == 0 for c in coeffs):
                        continue
                    
                    poly = [0] * n
                    for pos, c in zip(positions, coeffs):
                        poly[pos] = c
                    
                    if not check_constraints(poly):
                        continue
                    
                    sq = compute_square(poly)
                    score = count_nonzeros(sq)
                    
                    if score < best[0]:
                        best = (score, poly)
                        if score <= 18:
                            return best
        
        return best
    
    # Run searches
    score1, poly1 = search_antisymmetric()
    if score1 < best_score:
        best_score = score1
        best_poly = poly1
    
    if best_score > 19:  # Only run more expensive search if needed
        score2, poly2 = search_symmetric_structures()
        if score2 < best_score:
            best_score = score2
            best_poly = poly2
    
    # Final verification
    assert sum(1 for c in best_poly if c != 0) == 12
    assert all(-64 <= c <= 64 for c in best_poly)
    assert len(best_poly) <= 41
    assert best_poly[-1] != 0
    
    return best_poly
