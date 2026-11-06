import time
import random
from itertools import product

DEADLINE = 90.0
SAFETY_MARGIN = 5.0

def _square_nonzero_count_fast(exponents, signs):
    """
    Calculate the number of nonzero terms in the square of a polynomial 
    defined by exponents and signs.
    exponents: list of 12 ints
    signs: list of 12 ints (+1 or -1)
    """
    n = len(exponents)
    coeffs = {}
    for i in range(n):
        e_i = exponents[i]
        s_i = signs[i]
        for j in range(i, n):
            e_sum = e_i + exponents[j]
            s_prod = s_i * signs[j]
            if i == j:
                coeffs[e_sum] = coeffs.get(e_sum, 0) + 1
            else:
                coeffs[e_sum] = coeffs.get(e_sum, 0) + 2 * s_prod
    
    return sum(1 for v in coeffs.values() if v != 0)

def _best_signs_for_exponents(exponents):
    """
    For a fixed set of 12 exponents, find the sign assignment (+1/-1)
    that minimizes the nonzero count of the square.
    Brute forces 2^12 = 4096 possibilities.
    """
    best_count = 10**9
    best_signs = None
    n = len(exponents)
    
    # Iterate through all 2^12 sign combinations
    for mask in range(1 << n):
        signs = [1 if (mask >> i) & 1 == 0 else -1 for i in range(n)]
        
        coeffs = {}
        for i in range(n):
            e_i = exponents[i]
            s_i = signs[i]
            for j in range(i, n):
                e_sum = e_i + exponents[j]
                if i == j:
                    coeffs[e_sum] = coeffs.get(e_sum, 0) + 1
                else:
                    coeffs[e_sum] = coeffs.get(e_sum, 0) + 2 * s_i * signs[j]
        
        count = sum(1 for v in coeffs.values() if v != 0)
        if count < best_count:
            best_count = count
            best_signs = signs
            if count == 12: # Theoretical min for 12 terms? Actually min is 12 (if it were a monomial, but it's 12 terms). 
                # Wait, (1+x+...+x^11)^2 has 23 terms. 
                # Can we get 12? 
                # If P^2 = Q^2, then P=Q or P=-Q. 
                # If P has 12 terms, P^2 usually has many. 
                # The absolute minimum for a 12-term polynomial square is 12? 
                # No, (1-x)^2 = 1-2x+x^2 (3 terms). 
                # (1+x+x^2+x^3)^2 = 1+2x+3x^2+4x^3+3x^4+2x^5+x^6 (7 terms).
                # It is known that the minimum number of terms in the square of a k-term polynomial is k if k is a power of 2? No.
                # For k=12, we can try to get close to 12 or 13.
                pass
    
    return best_count, best_signs

def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN
    rng = random.Random(42)
    
    best_coefs = None
    best_sq_nz = 10**9
    
    def update(exponents, signs):
        nonlocal best_coefs, best_sq_nz
        max_exp = max(exponents)
        coefs = [0] * (max_exp + 1)
        for e, s in zip(exponents, signs):
            coefs[e] = s
        if len(coefs) > 41:
            return
        if coefs[-1] == 0:
            return
        if sum(1 for c in coefs if c != 0) != 12:
            return
        if any(c not in range(-64, 65) for c in coefs):
            return
            
        count = _square_nonzero_count_fast(exponents, signs)
        if count < best_sq_nz:
            best_sq_nz = count
            best_coefs = coefs

    # Strategy: Structured Factorization
    # We assume P(x) = A(x) * B(x) * C(x)
    # where A, B, C are 4-term polynomials.
    # To keep P sparse (12 terms), we need the supports of A, B, C to be "disjoint" in a way that doesn't create collisions, 
    # OR we use a specific structure like P(x) = sum_{i=0}^{11} s_i x^{e_i}.
    # Actually, the product of three 4-term polynomials generally has 64 terms. 
    # To get 12 terms, we need massive cancellation or a very specific structure.
    # A better structural insight: 
    # If we pick exponents from a small set, we get collisions.
    # Let's try to construct P(x) such that P(x)^2 has minimal terms.
    # Known result: The minimum number of nonzero terms in the square of a polynomial with k nonzero terms is at least k.
    # For k=12, can we achieve 12? 
    # If P(x) = 1 + x + x^2 + ... + x^{11}, P^2 has 23 terms.
    # If we use signs: P(x) = 1 - x + x^2 - ... + x^{11}.
    # P(x) = (1-x^{12})/(1+x). P(x)^2 = (1-x^{12})^2 / (1+x)^2.
    # (1-x^{12})^2 = 1 - 2x^{12} + x^{24}.
    # 1/(1+x)^2 = sum_{k=0}^\infty (-1)^k (k+1) x^k.
    # So P(x)^2 = (1 - 2x^{12} + x^{24}) * sum_{k=0}^{11} (-1)^k (k+1) x^k (truncated? No, infinite).
    # Wait, P(x) is finite.
    # Let's just search.
    
    # We will generate candidates by:
    # 1. Random exponent sets with high collision probability (small range).
    # 2. Specific patterns like geometric progressions or mixed progressions.
    
    candidates = []
    
    # 1. Arithmetic Progressions (already in seed, but let's include)
    for start in range(0, 30):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) < 41:
                candidates.append(exps)
                
    # 2. "Clustered" exponents: pick a base, then small random offsets
    # This increases the chance of sum collisions.
    for _ in range(2000):
        if time.time() > deadline:
            break
        base = rng.randint(0, 15)
        exps = set()
        while len(exps) < 12:
            e = base + rng.randint(0, 25)
            if e < 41:
                exps.add(e)
        exps = sorted(exps)
        candidates.append(exps)
        
    # 3. "Geometric-like" or "Exponential" spacing? 
    # e.g. 1, 2, 4, 8, 16... (Powers of 2). 
    # If exponents are powers of 2, sums are unique (binary representation). 
    # So no collisions. Bad for cancellation.
    # We want NON-unique sums.
    
    # 4. "Twin" exponents: pairs of close exponents.
    # e.g. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 is AP.
    # What if we use 0, 1, 1, 2, 2, 3...? No, exponents must be distinct for the polynomial to be well-defined as a sum of terms with distinct powers? 
    # The problem says "exactly 12 nonzero terms". It does NOT explicitly say exponents must be distinct.
    # However, if two terms have the same exponent, they combine into one term with coefficient c1+c2.
    # If c1+c2 != 0, it's 1 term. If c1+c2=0, it's 0 terms.
    # If we allow duplicate exponents, we can have "virtual" terms that cancel out or combine.
    # But the output is a coefficient list. 
    # "exactly 12 nonzero terms" in the INPUT polynomial.
    # If I put 1 at x^0 and 1 at x^0, the coefficient is 2. That's 1 term.
    # So to have 12 nonzero terms in the input, I need 12 distinct exponents with non-zero coefficients.
    # So exponents MUST be distinct.
    
    # Let's evaluate candidates
    for exps in candidates:
        if time.time() > deadline:
            break
        # Quick filter: unique sums
        sums = set()
        n = 12
        for i in range(n):
            for j in range(i, n):
                sums.add(exps[i] + exps[j])
        if len(sums) > 25:
            continue
            
        best_count, best_signs = _best_signs_for_exponents(exps)
        
        if best_count < best_sq_nz:
            best_sq_nz = best_count
            max_exp = max(exps)
            coefs = [0] * (max_exp + 1)
            for e, s in zip(exps, best_signs):
                coefs[e] = s
            best_coefs = coefs
            
    # Fallback
    if best_coefs is None:
        exps = list(range(12))
        best_count, best_signs = _best_signs_for_exponents(exps)
        best_sq_nz = best_count
        coefs = [0] * 12
        for e, s in zip(exps, best_signs):
            coefs[e] = s
        best_coefs = coefs
        
    # Local Search Refinement
    # We will perform a random restart local search on the exponents.
    if time.time() < deadline - 2.0 and best_coefs is not None:
        current_exps = [i for i, c in enumerate(best_coefs) if c != 0]
        current_signs = [best_coefs[i] for i in current_exps]
        
        # Simulated Annealing on exponents
        T = 10.0
        cooling = 0.99
        best_local_count = best_sq_nz
        best_local_exps = list(current_exps)
        best_local_signs = list(current_signs)
        
        iterations = 0
        while time.time() < deadline - 1.0 and T > 0.1:
            iterations += 1
            # Pick a random exponent to move
            i = rng.randint(0, 11)
            old_e = current_exps[i]
            new_e = old_e + rng.randint(-5, 5)
            
            if new_e < 0 or new_e >= 41:
                continue
            if new_e in current_exps:
                continue
                
            new_exps = list(current_exps)
            new_exps[i] = new_e
            new_exps.sort()
            
            # Optimize signs for this new exponent set
            new_count, new_signs = _best_signs_for_exponents(new_exps)
            
            # Metropolis acceptance
            delta = new_count - best_local_count
            if delta < 0 or rng.random() < pow(2.71828, -delta / T):
                current_exps = new_exps
                current_signs = new_signs
                if new_count < best_local_count:
                    best_local_count = new_count
                    best_local_exps = list(new_exps)
                    best_local_signs = list(new_signs)
                    if best_local_count < best_sq_nz:
                        best_sq_nz = best_local_count
                        max_exp = max(best_local_exps)
                        best_coefs = [0] * (max_exp + 1)
                        for e, s in zip(best_local_exps, best_local_signs):
                            best_coefs[e] = s
            T *= cooling
            
    # Final check
    if best_coefs is None or not best_coefs or best_coefs[-1] == 0:
        return [1] * 12
        
    return best_coefs

if __name__ == "__main__":
    res = construct()
    print(res)
