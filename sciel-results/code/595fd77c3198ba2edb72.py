import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # The goal is to find a polynomial P with exactly 12 nonzero terms,
    # coefficients in [-64, 64], degree <= 40,
    # such that the number of nonzero terms in P^2 is minimized.
    
    # Strategy:
    # 1. Fix a support structure that encourages collisions.
    #    A dense support like {0, 1, ..., 11} or a subset of a small range
    #    creates many pairwise sums that collide.
    #    Specifically, if the support is a contiguous block of 12 integers,
    #    the square has support size 23.
    #    If we use a non-contiguous support, the square support might be larger,
    #    but collisions might be "cleaner" if the support is structured (e.g., AP).
    #    
    # 2. For a fixed support, the coefficients (signs) determine the square's values.
    #    Let P(x) = sum_{i=0}^{11} s_i x^{e_i}, where s_i in {-1, 1}.
    #    (We restrict to {-1, 1} because larger coefficients increase the chance
    #     of non-zero results in the square, and we want cancellation.
    #     Also, the problem allows [-64, 64], but sparse squares are typically
    #     found with low-weight polynomials. If a solution with {-1, 1} exists
    #     with sparsity K, it's a strong candidate. If we allow larger coefficients,
    #     we might find sparser squares, but the search space is huge.
    #     Let's stick to {-1, 1} first, as it's the most common source of sparse squares
    #     like Golay sequences. If we don't find a good one, we can expand.)
    
    #    Actually, the task says "Minimize nonzero terms".
    #    Known result: For 12 terms, can we get a square with very few terms?
    #    A "perfect" square with 1 term is impossible for 12 terms.
    #    A "Golay pair" of length 12? Golay sequences exist for lengths that are
    #    powers of 2 or 3*2^k. 12 = 3*4. So Golay sequences of length 12 exist.
    #    A Golay sequence g has zero aperiodic autocorrelation.
    #    This means (sum g_i x^i)^2 + (sum h_i x^i)^2 = 2 * sum x^{2i} (if h is the other in the pair).
    #    Wait, the definition of Golay pair (g, h) is that the aperiodic autocorrelations
    #    of g and h sum to zero for all non-zero shifts.
    #    This implies that if we take P(x) = G(x) + x^k H(x) for some shift,
    #    the square might have cancellations.
    #    
    #    Actually, a simpler structure:
    #    Consider P(x) = A(x) + B(x) where A and B have disjoint supports.
    #    P^2 = A^2 + B^2 + 2AB.
    #    If we choose A and B such that A^2, B^2, and AB have overlapping supports
    #    with cancelling coefficients, we can reduce the nonzero count.
    
    #    Let's try to find a Golay pair of length 12.
    #    If (g, h) is a Golay pair, then the polynomial G(x) = sum g_i x^i and H(x) = sum h_i x^i
    #    satisfy that the aperiodic autocorrelation of G and H are zero.
    #    This is related to sparse squares if we consider P(x) = G(x) + x^6 H(x)?
    #    No, let's look at the square of a single polynomial.
    
    #    Alternative: Random search with local optimization on the sign vector.
    #    We fix the support to be a contiguous block {0, 1, ..., 11}.
    #    This maximizes the potential for collisions (all sums i+j are in 0..22).
    #    We want to minimize the number of nonzero entries in the convolution s * s.
    
    #    Let's use a local search (simulated annealing or hill climbing) on the 12 signs.
    #    There are only 2^12 = 4096 possible sign vectors.
    #    We can EXHAUSTIVELY search all 4096 vectors for a fixed support!
    #    This is extremely fast.
    #    
    #    So the bottleneck is the SUPPORT.
    #    We should search over different supports.
    #    Since 4096 sign checks per support is cheap, we can test many supports.
    
    #    What supports to test?
    #    1. Contiguous: [0, 1, ..., 11]
    #    2. Arithmetic Progressions: [0, d, 2d, ..., 11d] for d=1, 2, 3...
    #    3. Random subsets of {0, ..., 20}
    #    4. Structured subsets (e.g., even/odd splits)
    
    #    Let's implement the exhaustive sign search for a given support.
    
    def count_nonzero_square(support, signs):
        """
        support: list of 12 exponents
        signs: list of 12 signs (+1 or -1)
        returns: number of nonzero coefficients in the square
        """
        # Map exponent to coefficient
        # Square coefficients are at exponents support[i] + support[j]
        # We can use a dictionary or a fixed-size array if the range is small.
        # Max exponent is 2 * max(support).
        # Let's use a dict for simplicity, or an array if we know the range.
        
        # Optimization: Use an array of zeros.
        # Determine min and max exponent.
        min_e = 2 * min(support)
        max_e = 2 * max(support)
        # This might be large if support is sparse.
        # But for contiguous 0..11, max_e = 22.
        # For d=10, max_e = 220.
        # A dict is safer for sparse supports.
        
        coeffs = {}
        for i in range(12):
            for j in range(12):
                e = support[i] + support[j]
                val = signs[i] * signs[j]
                coeffs[e] = coeffs.get(e, 0) + val
        
        return sum(1 for v in coeffs.values() if v != 0)

    def best_signs_for_support(support):
        """
        Exhaustively find the sign vector that minimizes nonzero square terms.
        Returns (min_nz, best_signs)
        """
        best_nz = 100
        best_signs = None
        
        # Iterate over all 2^12 masks
        for mask in range(4096):
            # Construct signs
            # We can optimize this by using bit operations, but 4096 is small.
            signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
            
            nz = count_nonzero_square(support, signs)
            
            if nz < best_nz:
                best_nz = nz
                best_signs = signs
                if best_nz == 1: # Theoretical minimum is 1 (monomial square), impossible for 12 terms, but check anyway
                    break
                if best_nz == 2:
                    break
        
        return best_nz, best_signs

    best_global_nz = 100
    best_global_poly = None
    
    # Generate candidate supports
    supports_to_test = []
    
    # 1. Contiguous block starting at 0
    supports_to_test.append(list(range(12)))
    
    # 2. Arithmetic progressions
    for d in range(1, 6):
        for start in range(0, 5):
            supports_to_test.append([start + i*d for i in range(12)])
            
    # 3. Random supports from a dense range (0..20)
    # We want supports that are "dense" to allow collisions.
    # Let's generate 500 random supports from {0..20}
    random.seed(42)
    for _ in range(500):
        # Choose 12 distinct exponents from 0..20
        support = random.sample(range(21), 12)
        support.sort()
        supports_to_test.append(support)
        
    # 4. Structured supports: Two interleaved APs
    # e.g., evens and odds
    supports_to_test.append([i for i in range(0, 24, 2)][:12]) # 0, 2, ..., 22
    supports_to_test.append([i for i in range(1, 25, 2)][:12]) # 1, 3, ..., 23
    
    # 5. Golay-inspired: If we know a Golay pair, the support is just 0..11.
    # But maybe we can use a support that is a union of two smaller Golay supports?
    # For now, the random search covers a lot.
    
    # Remove duplicates
    unique_supports = []
    seen = set()
    for s in supports_to_test:
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            unique_supports.append(s)
    
    # Test each support
    for idx, support in enumerate(unique_supports):
        if time.time() > deadline:
            break
            
        nz, signs = best_signs_for_support(support)
        
        if nz < best_global_nz:
            best_global_nz = nz
            # Construct the polynomial coefficients
            min_e = min(support)
            max_e = max(support)
            length = max_e - min_e + 1
            poly = [0] * length
            for i in range(12):
                poly[support[i] - min_e] = signs[i]
            best_global_poly = poly
            
            # If we find a very sparse square, we can stop early
            # What is the theoretical minimum?
            # For 12 terms, a square with < 10 terms would be excellent.
            # A square with 1 term is impossible.
            # A square with 2 terms is impossible for real polynomials (sum of squares).
            # Wait, P^2 = (a x^k + b x^m)^2 = a^2 x^{2k} + 2ab x^{k+m} + b^2 x^{2m}.
            # This has 3 terms.
            # Can we get 3 terms?
            # If P has 2 terms, P^2 has 3 terms.
            # But P must have 12 terms.
            # If P has 12 terms, can P^2 have 3 terms?
            # P^2 = A x^{2k} + B x^{k+m} + C x^{2m}.
            # This implies P is a binomial. Contradiction.
            # So P^2 must have at least 3 terms?
            # Actually, if P is not a binomial, P^2 has at least 3 terms?
            # No, (x^2+1)^2 = x^4+2x^2+1. 3 terms.
            # (x^3+x^2+x+1)^2 = ... many terms.
            # It is known that the number of nonzero terms in the square of a polynomial
            # with n nonzero terms is at least n? No.
            # For n=2, square has 3 terms.
            # For n=3, (x^2+x+1)^2 = x^4+2x^3+3x^2+2x+1. 5 terms.
            # (x^2+1-x)^2 = (x^2+1)^2 - 2x(x^2+1) = x^4+2x^2+1 - 2x^3 - 2x = x^4-2x^3+2x^2-2x+1. 5 terms.
            # Can we get 3 terms with 3 input terms?
            # P = a x^i + b x^j + c x^k.
            # P^2 = a^2 x^{2i} + b^2 x^{2j} + c^2 x^{2k} + 2ab x^{i+j} + 2ac x^{i+k} + 2bc x^{j+k}.
            # To have 3 terms, 3 of these must cancel or be zero.
            # Coeffs are squares, so a^2, b^2, c^2 are non-zero.
            # So the 3 diagonal terms are always present.
            # Thus, if i,j,k are distinct, P^2 has at least 3 terms.
            # If we can make the cross terms cancel or overlap with diagonals?
            # Cross terms exponents: i+j, i+k, j+k.
            # Diagonal exponents: 2i, 2j, 2k.
            # If i+j = 2k, then the term 2ab x^{i+j} adds to c^2 x^{2k}.
            # If 2ab + c^2 = 0, that term vanishes.
            # Since a,b,c are integers, 2ab = -c^2.
            # c^2 is even, so c is even. Let c=2d. 2ab = -4d^2 => ab = -2d^2.
            # This is possible.
            # So we can potentially reduce the term count.
            
            # If best_nz is very low (e.g., < 15), we are in good shape.
            if best_global_nz <= 12:
                break

    # Fallback if no support was tested or all failed (should not happen)
    if best_global_poly is None:
        # Use the contiguous support 0..11 with all 1s
        best_global_poly = [1]*12
        # Calculate its square sparsity just to be safe, though we return the poly
        # The evaluator will calculate the score.
        
    return best_global_poly

# EVOLVE-BLOCK-END
