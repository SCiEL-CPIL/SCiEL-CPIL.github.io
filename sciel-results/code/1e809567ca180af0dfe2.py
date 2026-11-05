"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin
    
    best_square_count = 24  # For 12 distinct terms with all 1s, square has 12*2-1 = 23 nonzero terms (if no overlap) or 24? 
    # Actually for [1]*12, square is [1,2,3,4,5,6,7,6,5,4,3,2,1] -> 13 nonzero terms.
    # So 13 is a good initial benchmark.
    best_square_count = 13
    best_poly = [1]*12
    
    # Try to find polynomials with fewer nonzero terms in their square.
    # Strategy: Search for structures like (1 + x^k)^n or similar binomial/trinomial expansions.
    # Or random search with specific coefficient patterns.
    
    # Known good structures:
    # (1 + x^2 + x^4 + ... + x^{2*11})^2 = sum_{j=0}^{22} (12-j) x^{2j} for j<=11, then decreasing.
    # Support is {0, 2, 4, ..., 22}. Count = 12.
    # This is better than 13!
    # Let's verify: P(x) = 1 + x^2 + x^4 + ... + x^22. 12 terms.
    # P(x)^2 = (1 + x^2 + ... + x^22)^2.
    # Coeff of x^{2k} is number of pairs (i,j) such that 2i+2j=2k => i+j=k, with 0<=i,j<=11.
    # For k=0..11: count is k+1.
    # For k=12..22: count is 23-k.
    # All coefficients are nonzero.
    # Support is {0, 2, 4, ..., 22}. Size = 12.
    # So we can achieve 12.
    
    # Can we do better than 12?
    # Minimum possible support size for square of 12-term polynomial?
    # If P has 12 terms, P^2 can have as few as 12 terms if P is of form c * (1+x^k) ... no.
    # If P = 1 + x^a + x^b + ...
    # Consider P(x) = 1 + x + x^2 + ... + x^11. Square has 13 terms.
    # Consider P(x) = 1 + x^2 + ... + x^22. Square has 12 terms.
    # Consider P(x) = 1 + x^3 + x^6 + ... + x^33. Square has 12 terms (support 0,6,...,66).
    
    # What if we use negative coefficients?
    # P(x) = 1 - x^k. 2 terms. Square: 1 - 2x^k + x^{2k}. 3 terms.
    # P(x) = 1 - x^k + x^{2k}. 3 terms. Square: (1-x^k)^2 + 2x^{2k}(1-x^k) ... 
    # (1 - x^k + x^{2k})^2 = 1 - 2x^k + 3x^{2k} - 2x^{3k} + x^{4k}. 5 terms.
    
    # Let's search for polynomials with 12 terms that have square support < 12.
    # Theoretically, the support of P^2 is at least the support of P? No.
    # But generally, it's hard to get below the number of terms in P if coefficients are positive.
    # With mixed signs, cancellations can occur.
    
    # Let's try random search with small coefficients {-1, 0, 1} and specific structures.
    
    # First, let's return the known 12-term square solution.
    candidate = [0]*41
    for i in range(12):
        candidate[2*i] = 1
    # Check square
    sq = np.convolve(candidate, candidate)
    nonzero_sq = np.count_nonzero(sq)
    if nonzero_sq < best_square_count:
        best_square_count = nonzero_sq
        best_poly = candidate
    
    # Try to improve further.
    # Search for P with 12 terms, coeff in {-1, 1}, that minimizes nonzero terms in P^2.
    
    # Heuristic: If P(x) = Q(x)^2 where Q has 3 or 4 terms?
    # If Q has n terms, Q^2 has at most 2n-1 terms.
    # If P = Q^2, then P^2 = Q^4.
    # If Q has 4 terms, Q^2 has ~7 terms, Q^4 has ~13 terms.
    # If Q has 3 terms, Q^2 has 5 terms, Q^4 has 9 terms.
    # But P must have exactly 12 terms.
    # So P cannot be a perfect square of a sparse polynomial unless the square expands to 12 terms.
    
    # Let's do a random search for better structures.
    # Focus on exponents that are multiples of a step k, to keep support sparse.
    
    for step in [2, 3, 4, 5, 6, 8, 10, 12]:
        if time.time() > deadline:
            break
            
        # Generate random sets of 12 exponents that are multiples of step
        # and within [0, 40]
        max_exp = 40 // step
        if max_exp < 11:
            continue
            
        # Try a few random subsets of exponents
        for _ in range(1000):
            if time.time() > deadline:
                break
            exponents = sorted(np.random.choice(range(max_exp+1), 12, replace=False) * step)
            coeffs = np.random.choice([-1, 1], 12)
            
            poly = np.zeros(41, dtype=int)
            for i, e in enumerate(exponents):
                poly[e] = coeffs[i]
                
            sq = np.convolve(poly, poly)
            nz = np.count_nonzero(sq)
            if nz < best_square_count:
                best_square_count = nz
                best_poly = poly.tolist()
                
    # Also try specific algebraic identities
    # P(x) = (1 + x^k + x^{2k} + x^{3k}) * (1 + x^{4k} + x^{5k} + x^{6k})?
    # This is getting complex. The random search above might find something.
    
    # One more specific check:
    # P(x) = 1 + x^2 + x^4 + x^6 + x^8 + x^10 + x^12 + x^14 + x^16 + x^18 + x^20 + x^22
    # Square support size is 12.
    
    # Can we get 11 or less?
    # If P has 12 terms, the sum of exponents in P^2 is 2*sum(exponents in P).
    # The support size is at least the number of distinct sums i+j.
    # For 12 distinct exponents, the number of distinct sums i+j is at least 2*12-1 = 23 if they are in arithmetic progression? No, that's for the number of sums.
    # Wait, for A = {a1, ..., a12}, the set A+A has size at least 2*12-1 = 23 if A is in AP?
    # No, if A is in AP, A+A is also in AP, size 2*12-1 = 23.
    # But if A is not in AP, A+A can be larger.
    # However, if coefficients cancel, the *nonzero* terms in the square can be fewer.
    # In the example P = 1 + x^2 + ... + x^22, the exponents are {0, 2, ..., 22}.
    # The sums are {0, 2, ..., 44}. There are 23 distinct sums.
    # But the coefficients are all positive, so no cancellation.
    # Wait, my previous calculation said 12 nonzero terms. Let me re-verify.
    # P(x) = sum_{i=0}^{11} x^{2i}.
    # P(x)^2 = sum_{i=0}^{11} sum_{j=0}^{11} x^{2(i+j)}.
    # The exponent is 2k where k = i+j.
    # k ranges from 0 to 22.
    # For each k, the coefficient is the number of pairs (i,j) such that i+j=k, 0<=i,j<=11.
    # For k=0: (0,0) -> 1
    # For k=1: (0,1),(1,0) -> 2
    # ...
    # For k=11: (0,11)..(11,0) -> 12
    # For k=12: (1,11)..(11,1) -> 11
    # ...
    # For k=22: (11,11) -> 1
    # All coefficients are nonzero.
    # The exponents in the square are 0, 2, 4, ..., 44.
    # That is 23 distinct exponents.
    # So the square has 23 nonzero terms, not 12!
    # My previous reasoning was flawed. I confused the number of terms in P with the support size of P^2.
    
    # So the benchmark [1]*12 has square [1,2,3,4,5,6,7,6,5,4,3,2,1] which has 13 nonzero terms.
    # The "step 2" polynomial has 23 nonzero terms.
    # So [1]*12 is better.
    
    # We need to find a polynomial with 12 terms whose square has < 13 nonzero terms.
    # This requires significant cancellation.
    # Example: P(x) = 1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^{10} - x^{11}
    # P(x) = (1-x^{12})/(1-x).
    # P(x)^2 = (1-x^{12})^2 / (1-x)^2 = (1 - 2x^{12} + x^{24}) / (1 - 2x + x^2).
    # This is a rational function, not a polynomial division that terminates nicely to a sparse polynomial?
    # Actually, (1-x^{12})^2 / (1-x)^2 is a polynomial.
    # Let's compute the first few terms.
    # 1/(1-x)^2 = 1 + 2x + 3x^2 + ...
    # (1 - 2x^{12} + x^{24}) * (1 + 2x + 3x^2 + ...)
    # = (1 + 2x + 3x^2 + ...) - 2x^{12}(1 + 2x + 3x^2 + ...) + x^{24}(1 + 2x + 3x^2 + ...)
    # Coeff of x^k for k < 12 is k+1.
    # Coeff of x^{12} is 13 - 2 = 11.
    # Coeff of x^{13} is 14 - 4 = 10.
    # ...
    # Coeff of x^{23} is 24 - 2*12 = 0?
    # Let's check: term from first part: 24. Term from second part: -2 * 12 (coeff of x^{11} in 1/(1-x)^2 is 12). 24 - 24 = 0.
    # So there are zeros in the middle!
    # Let's verify the number of nonzero terms.
    # The polynomial is symmetric around x^{12}? No, degree is 22.
    # Let's just compute it.
    
    candidate_alt = [0]*41
    for i in range(12):
        candidate_alt[i] = 1 if i % 2 == 0 else -1
        
    sq_alt = np.convolve(candidate_alt, candidate_alt)
    nz_alt = np.count_nonzero(sq_alt)
    if nz_alt < best_square_count:
        best_square_count = nz_alt
        best_poly = candidate_alt
        
    # Continue random search if time permits
    # Focus on alternating signs or other cancellation patterns
    
    return best_poly
# EVOLVE-BLOCK-END
