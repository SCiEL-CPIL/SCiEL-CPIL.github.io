"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    """
    Return coefficient list in ascending degree, length<=41, nonzero final coefficient,
    entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square.
    
    Approach: Model as combinatorial optimization over exponent multisets.
    Construct P = A * B where A, B are 6-term polynomials with carefully
    chosen exponent supports to force cross-term cancellations in P^2 = A^2 * B^2.
    """
    
    def poly_mul(p, q):
        """Multiply two polynomials given as coefficient lists."""
        n, m = len(p), len(q)
        result = [0] * (n + m - 1)
        for i in range(n):
            if p[i] == 0:
                continue
            for j in range(m):
                if q[j] == 0:
                    continue
                result[i+j] += p[i] * q[j]
        return result
    
    def square_coeffs(p):
        """Compute P^2 coefficients from P's coefficient list."""
        return poly_mul(p, p)
    
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
    
    def evaluate_p(p):
        """Evaluate a candidate polynomial and update best if better."""
        nonlocal best_nz, best_p
        if not is_valid(p):
            return
        sq = square_coeffs(p)
        nz = count_nonzero(sq)
        if nz < best_nz:
            best_nz = nz
            best_p = p[:]
    
    # Strategy: P = A * B where A and B are 6-term polynomials
    # Key insight: P^2 = A^2 * B^2
    # If A^2 has few terms and B^2 has few terms, and their exponent sets
    # have many collisions when summed, P^2 will be sparse.
    
    # Phase 1: Systematic search over structured A and B
    # Use small exponent ranges and coefficient magnitudes
    
    # Define candidate exponent sets for A (6 terms)
    # Focus on sets that make A^2 sparse
    a_exp_sets = []
    
    # Consecutive exponents
    for start in range(0, 20):
        a_exp_sets.append(list(range(start, start+6)))
    
    # Even exponents
    for start in range(0, 20, 2):
        a_exp_sets.append([start, start+2, start+4, start+6, start+8, start+10])
    
    # Arithmetic progressions with various steps
    for start in range(0, 15):
        for step in [1, 2, 3]:
            exps = [start + i*step for i in range(6)]
            if max(exps) < 41:
                a_exp_sets.append(exps)
    
    # Some structured sets
    a_exp_sets.extend([
        [0, 1, 2, 3, 4, 10],
        [0, 1, 2, 3, 10, 11],
        [0, 1, 2, 10, 11, 12],
        [0, 1, 10, 11, 12, 13],
        [0, 10, 11, 12, 13, 14],
        [0, 2, 4, 6, 8, 14],
        [0, 2, 4, 6, 14, 16],
        [0, 2, 4, 14, 16, 18],
        [0, 2, 14, 16, 18, 20],
        [0, 14, 16, 18, 20, 22],
    ])
    
    # Define candidate exponent sets for B (6 terms)
    # B should be "shifted" or "scaled" relative to A to create structure
    b_exp_sets = []
    
    for start in range(0, 20):
        b_exp_sets.append(list(range(start, start+6)))
    
    for start in range(0, 20, 2):
        b_exp_sets.append([start, start+2, start+4, start+6, start+8, start+10])
    
    for start in range(0, 15):
        for step in [1, 2, 3]:
            exps = [start + i*step for i in range(6)]
            if max(exps) < 41:
                b_exp_sets.append(exps)
    
    b_exp_sets.extend([
        [0, 1, 2, 3, 4, 10],
        [0, 1, 2, 3, 10, 11],
        [0, 1, 2, 10, 11, 12],
        [0, 1, 10, 11, 12, 13],
        [0, 10, 11, 12, 13, 14],
        [0, 2, 4, 6, 8, 14],
        [0, 2, 4, 6, 14, 16],
        [0, 2, 4, 14, 16, 18],
        [0, 2, 14, 16, 18, 20],
        [0, 14, 16, 18, 20, 22],
    ])
    
    # Coefficient patterns for 6-term polynomials
    # Use small magnitudes to keep coefficients bounded
    coeff_patterns = []
    
    # All +/-1
    for mask in range(64):  # 2^6
        coeff_patterns.append([1 if (mask >> i) & 1 else -1 for i in range(6)])
    
    # Some with magnitude 2
    for mask in range(64):
        base = [1 if (mask >> i) & 1 else -1 for i in range(6)]
        # Try doubling some coefficients
        for double_mask in [0, 1, 2, 4, 8, 16, 32, 63]:
            variant = base[:]
            for i in range(6):
                if (double_mask >> i) & 1:
                    variant[i] *= 2
            coeff_patterns.append(variant)
    
    # Deduplicate
    seen = set()
    unique_patterns = []
    for cp in coeff_patterns:
        key = tuple(cp)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(cp)
    coeff_patterns = unique_patterns
    
    # Phase 1: Search over A and B with structured exponents
    # Limit to manageable combinations
    a_exp_sets_sample = a_exp_sets[:50]
    b_exp_sets_sample = b_exp_sets[:50]
    coeff_patterns_sample = coeff_patterns[:30]
    
    for a_exps in a_exp_sets_sample:
        max_a = max(a_exps)
        for b_exps in b_exp_sets_sample:
            max_b = max(b_exps)
            if max_a + max_b >= 41:
                continue
            
            for a_coeffs in coeff_patterns_sample:
                for b_coeffs in coeff_patterns_sample:
                    # Compute P = A * B
                    p = [0] * (max_a + max_b + 1)
                    for ae, ac in zip(a_exps, a_coeffs):
                        for be, bc in zip(b_exps, b_coeffs):
                            p[ae + be] += ac * bc
                    
                    evaluate_p(p)
    
    # Phase 2: Broader search with random sampling
    random.seed(42)
    for _ in range(5000):
        # Random 6-term A
        a_exps = sorted(random.sample(range(0, 25), 6))
        a_coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        
        # Random 6-term B, shifted to avoid overlap
        shift = random.randint(5, 15)
        b_exps = sorted(random.sample(range(shift, shift + 20), 6))
        b_coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        
        if max(b_exps) >= 41:
            continue
        
        # Compute P = A * B
        max_exp = max(a_exps) + max(b_exps)
        if max_exp >= 41:
            continue
        p = [0] * (max_exp + 1)
        for ae, ac in zip(a_exps, a_coeffs):
            for be, bc in zip(b_exps, b_coeffs):
                p[ae + be] += ac * bc
        
        evaluate_p(p)
    
    # Phase 3: Direct search on 12-term polynomials with structured patterns
    # Try polynomials with symmetric or anti-symmetric coefficient patterns
    for exps in [
        list(range(12)),
        list(range(0, 12, 1)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13],
    ]:
        if len(exps) != 12 or max(exps) >= 41:
            continue
        # Try all +/-1 patterns
        for mask in range(4096):
            coeffs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
            p = [0] * (max(exps) + 1)
            for i, e in enumerate(exps):
                p[e] = coeffs[i]
            evaluate_p(p)
    
    # Phase 4: Try specific high-value constructions
    # P = (1+x+x^2+x^3+x^4+x^5)(x^k - x^(k+1) + x^(k+2) - x^(k+3) + x^(k+4) - x^(k+5))
    for k in range(5, 25):
        a_exps = [0, 1, 2, 3, 4, 5]
        a_coeffs = [1, 1, 1, 1, 1, 1]
        b_exps = [k, k+1, k+2, k+3, k+4, k+5]
        b_coeffs = [1, -1, 1, -1, 1, -1]
        
        if max(b_exps) >= 41:
            continue
        
        p = [0] * (max(a_exps) + max(b_exps) + 1)
        for ae, ac in zip(a_exps, a_coeffs):
            for be, bc in zip(b_exps, b_coeffs):
                p[ae + be] += ac * bc
        evaluate_p(p)
        
        # Also try with A having alternating signs
        a_coeffs2 = [1, -1, 1, -1, 1, -1]
        p2 = [0] * (max(a_exps) + max(b_exps) + 1)
        for ae, ac in zip(a_exps, a_coeffs2):
            for be, bc in zip(b_exps, b_coeffs):
                p2[ae + be] += ac * bc
        evaluate_p(p2)
    
    # If no valid candidate found, fall back to a simple valid one
    if best_p is None:
        # Simple fallback: 12 consecutive terms with alternating signs
        best_p = [1 if i % 2 == 0 else -1 for i in range(12)]
    
    return best_p
# EVOLVE-BLOCK-END
