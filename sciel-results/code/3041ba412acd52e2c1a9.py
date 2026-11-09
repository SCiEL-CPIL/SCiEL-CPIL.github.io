"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np

def _count_nonzero(coeffs):
    """Count nonzero entries in a coefficient list."""
    return sum(1 for c in coeffs if c != 0)

def _poly_square_sparse(coeffs):
    """Compute the square of a polynomial given as a list of (exponent, coefficient) pairs.
    Returns the number of nonzero terms in the square."""
    # Build full coefficient array
    if not coeffs:
        return 0
    max_deg = max(e for e, c in coeffs if c != 0)
    arr = np.zeros(2 * max_deg + 1, dtype=int)
    for e, c in coeffs:
        if c != 0:
            arr[e] += c
    # Square via convolution
    sq = np.convolve(arr, arr, mode='full')
    # Count nonzero
    return int(np.count_nonzero(sq))

def _eval_candidate(exponents, signs):
    """Given exponents and signs, build polynomial and return sparsity of square."""
    coeffs = list(zip(exponents, signs))
    return _poly_square_sparse(coeffs)

def construct():
    best_sparsity = None
    best_result = None
    
    # Family 1: Alternating patterns with various gap structures
    # Try polynomials of form sum_{i=0}^{11} s_i * x^{e_i}
    # where e_i are strictly increasing, s_i in {-1, 1}
    
    # Strategy: Try structured exponent sets
    # 1. Arithmetic progressions
    # 2. Geometric-like
    # 3. Random but structured
    
    candidates = []
    
    # Arithmetic progressions: e_i = a + i*b for i=0..11
    for a in range(0, 5):
        for b in range(1, 8):
            exponents = [a + i * b for i in range(12)]
            # Try all sign patterns? Too many (2^12 = 4096)
            # Try alternating and some structured signs
            sign_patterns = [
                [1] * 12,
                [(-1)**i for i in range(12)],
                [1 if i % 4 < 2 else -1 for i in range(12)],
                [1 if i % 3 < 2 else -1 for i in range(12)],
            ]
            for signs in sign_patterns:
                candidates.append((exponents, signs))
    
    # Specific known-good structures
    # P(x) = 1 - x + x^2 - x^3 + ... alternating
    exponents_alt = list(range(12))
    candidates.append((exponents_alt, [(-1)**i for i in range(12)]))
    candidates.append((exponents_alt, [1 if i % 4 < 2 else -1 for i in range(12)]))
    
    # P(x) = 1 - x^2 + x^4 - x^6 + ... (every other power)
    exponents_even = [2*i for i in range(12)]
    candidates.append((exponents_even, [(-1)**i for i in range(12)]))
    candidates.append((exponents_even, [1] * 12))
    
    # P(x) = 1 + x - x^2 - x^3 + x^4 + x^5 - ... (blocks of 2)
    exponents_seq = list(range(12))
    signs_block = [1 if (i // 2) % 2 == 0 else -1 for i in range(12)]
    candidates.append((exponents_seq, signs_block))
    
    # P(x) = 1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^10 - x^11
    # with various shifts
    
    # Try exponents with gaps: 0,1,3,4,6,7,9,10,12,13,15,16
    exponents_gap = [0,1,3,4,6,7,9,10,12,13,15,16]
    for signs in [[(-1)**i for i in range(12)], [1] * 12]:
        candidates.append((exponents_gap, signs))
    
    # Try: 0,2,4,6,8,10,12,14,16,18,20,22 (step 2)
    exponents_step2 = [2*i for i in range(12)]
    for signs in [[(-1)**i for i in range(12)], [1] * 12]:
        candidates.append((exponents_step2, signs))
    
    # Try: 0,1,4,5,8,9,12,13,16,17,20,21 (blocks of 2 with gap 2)
    exponents_blocks = [3*(i//2) + (i%2) for i in range(12)]
    for signs in [[(-1)**(i//2) for i in range(12)], [1] * 12]:
        candidates.append((exponents_blocks, signs))
    
    # Evaluate all candidates
    for exponents, signs in candidates:
        sparsity = _eval_candidate(exponents, signs)
        if best_sparsity is None or sparsity < best_sparsity:
            best_sparsity = sparsity
            best_result = (exponents, signs)
    
    # If we found something good, use it. Otherwise, use a reasonable default.
    if best_result is not None:
        exponents, signs = best_result
        # Build the coefficient list
        max_deg = max(exponents)
        result = [0] * (max_deg + 1)
        for e, s in zip(exponents, signs):
            result[e] = s
        # Ensure nonzero final coefficient
        if result[-1] == 0:
            result[-1] = 1
        # Ensure exactly 12 nonzero terms
        nonzero_count = sum(1 for c in result if c != 0)
        if nonzero_count != 12:
            # Fallback: use a simple 12-term polynomial
            result = [0] * 12
            for i in range(12):
                result[i] = 1
        return result
    else:
        # Fallback
        return [1] * 12

# EVOLVE-BLOCK-END
