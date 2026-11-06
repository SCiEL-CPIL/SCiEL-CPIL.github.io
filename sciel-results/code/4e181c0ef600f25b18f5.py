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
    # Use a dictionary to accumulate coefficients
    # Key: sum of exponents, Value: sum of products of signs
    coeffs = {}
    for i in range(n):
        e_i = exponents[i]
        s_i = signs[i]
        for j in range(i, n):
            e_sum = e_i + exponents[j]
            s_prod = s_i * signs[j]
            if i == j:
                # Diagonal term: c_i^2 = 1
                coeffs[e_sum] = coeffs.get(e_sum, 0) + 1
            else:
                # Off-diagonal terms: 2 * c_i * c_j
                coeffs[e_sum] = coeffs.get(e_sum, 0) + 2 * s_prod
    
    # Count nonzeros
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
    
    # Optimization: Pre-compute the exponent sums for all pairs
    # There are 12*13/2 = 78 unique pair sums (i<=j)
    # But the value depends on signs.
    
    # Iterate through all 2^12 sign combinations
    # Represent signs as an integer bitmask for speed
    for mask in range(1 << n):
        # Extract signs
        # sign[i] = 1 if bit i is 0, -1 if bit i is 1 (arbitrary mapping)
        # Let's just use a list for clarity first, optimize if needed
        signs = [1 if (mask >> i) & 1 == 0 else -1 for i in range(n)]
        
        # Calculate square nonzero count
        # Fast path: use dict
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
            if count == 1: # Theoretical minimum for non-zero poly square is 1 (monomial), but we have 12 terms so min is higher. 12 terms -> min square terms is roughly 12 (if it's a monomial times something? No. (sum x^i)^2 has at least 12 terms if no collisions? Actually, if P is a monomial, 1 term. If P has 2 terms, 3 terms. If P has k terms, min terms in P^2 is k (if all cross terms cancel? No, cross terms are 2xy. They can cancel if signs are opposite. e.g. (1-x)^2 = 1-2x+x^2 (3 terms). (1+x)^2 = 1+2x+x^2. To get fewer terms, we need collisions. 
                # Minimum terms for square of k-term poly:
                # If exponents are in AP, terms are 2k-1.
                # Can we get fewer? 
                # (1 + x + ... + x^{k-1})^2 has 2k-1 terms.
                # What if we use negative signs?
                # (1 - x + x^2 - ...)^2?
                # For k=2: (1-x)^2 = 1 - 2x + x^2 (3 terms).
                # For k=3: (1-x+x^2)^2 = 1 - 2x + 3x^2 - 2x^3 + x^4 (5 terms).
                # It seems hard to get below 2k-1 for generic exponents.
                # However, if exponents collide in the square, we can reduce it.
                # e.g. P = 1 + x^2 + x^4. P^2 = 1 + 2x^2 + 3x^4 + 2x^6 + x^8 (5 terms).
                # P = 1 + x + x^2 + x^3. P^2 has 7 terms.
                # If we can force many collisions and cancellations, we can go lower.
                pass
    
    return best_count, best_signs

def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN
    rng = random.Random(42)
    
    best_coefs = None
    best_sq_nz = 10**9
    
    # Initial candidate: 12 consecutive terms
    # exponents = [0, 1, ..., 11]
    # Signs: all 1
    # Square has 23 terms.
    
    def update(exponents, signs):
        nonlocal best_coefs, best_sq_nz
        # Build coefficient list
        max_exp = max(exponents)
        coefs = [0] * (max_exp + 1)
        for e, s in zip(exponents, signs):
            coefs[e] = s
        # Pad to at least 1? No, just check validity
        if len(coefs) > 41:
            return
        if coefs[-1] == 0:
            return
        if sum(1 for c in coefs if c != 0) != 12:
            return
        # Check bounds
        if any(c not in range(-64, 65) for c in coefs):
            return
            
        # We already know the count from the search, but let's be safe
        # Actually, _best_signs_for_exponents returns the count.
        # We pass the count in.
        pass

    # Strategy 1: Random Exponent Sets + Optimal Signs
    # We will generate random sets of 12 exponents in [0, 40].
    # For each set, find the best sign pattern.
    # Keep the global best.
    
    # To speed up, we can skip sets that look "bad" (e.g., all distinct sums are unique, implying 78 terms in square before cancellation, and cancellation is hard).
    # A set with many collisions in the exponent sums is promising.
    
    def count_unique_sums(exponents):
        sums = set()
        n = len(exponents)
        for i in range(n):
            for j in range(i, n):
                sums.add(exponents[i] + exponents[j])
        return len(sums)
    
    # Theoretical max unique sums for 12 terms is 78.
    # If unique sums is 78, no collisions in exponents, so no cancellation possible between different pairs (i,j) and (k,l) unless they are the same pair.
    # Wait, if sums are unique, then each coefficient in the square is from a unique pair (i,j).
    # The value is 1 (if i=j) or 2*s_i*s_j (if i!=j).
    # These are all non-zero (1 or +/-2).
    # So if all 78 sums are unique, the square has exactly 78 non-zero terms.
    # To get fewer terms, we NEED collisions in the sums.
    # So we should prioritize exponent sets with FEW unique sums.
    
    # Minimum unique sums?
    # If exponents are 0,1,2,3,4,5,6,7,8,9,10,11.
    # Sums range from 0 to 22.
    # Unique sums: 0, 1, ..., 22. Total 23 unique sums.
    # So AP gives 23 terms in square (with all + signs).
    # Can we do better?
    # If we use signs to cancel?
    # In AP, sum k is formed by pairs (i, k-i).
    # For k=11 (middle), pairs are (0,11), (1,10), ..., (5,6).
    # Coeff is 2*(s_0 s_11 + s_1 s_10 + ... + s_5 s_6).
    # If we choose signs such that this sum is 0, we reduce the count.
    # For AP, can we make the interior coefficients zero?
    # This is equivalent to finding a sequence of signs s_0...s_11 such that 
    # sum_{i=0}^{k} s_i s_{k-i} = 0 for many k.
    # This is related to Barker sequences or similar.
    
    # Let's search for exponent sets with low unique sum count, and then optimize signs.
    
    candidates = []
    
    # Generate some structured sets
    # 1. Arithmetic Progressions
    for start in range(0, 30):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) < 41:
                candidates.append(exps)
                
    # 2. Random sets with low collision count
    # We want to minimize unique sums.
    # Random sets usually have ~78 unique sums.
    # We need to bias towards clustered exponents.
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        # Generate clustered exponents
        # Pick a base, then add small random offsets
        base = rng.randint(0, 20)
        exps = set()
        while len(exps) < 12:
            e = base + rng.randint(0, 20)
            if e < 41:
                exps.add(e)
        exps = sorted(exps)
        # Check if we already have this set? Probably not worth it.
        candidates.append(exps)
        
    # Evaluate all candidates
    for exps in candidates:
        if time.time() > deadline:
            break
        # Quick filter: if unique sums > 25, it's unlikely to beat the AP (23 terms).
        # But maybe signs can reduce it?
        # If unique sums is U, the max terms is U.
        # We want to minimize the number of non-zero coefficients among these U sums.
        # So we should focus on sets with small U.
        
        unique_sums = count_unique_sums(exps)
        if unique_sums > 30:
            continue
            
        best_count, best_signs = _best_signs_for_exponents(exps)
        
        if best_count < best_sq_nz:
            best_sq_nz = best_count
            # Construct coefs
            max_exp = max(exps)
            coefs = [0] * (max_exp + 1)
            for e, s in zip(exps, best_signs):
                coefs[e] = s
            best_coefs = coefs
            
    # If no good candidate found, fallback to AP with optimal signs
    if best_coefs is None:
        exps = list(range(12))
        best_count, best_signs = _best_signs_for_exponents(exps)
        best_sq_nz = best_count
        coefs = [0] * 12
        for e, s in zip(exps, best_signs):
            coefs[e] = s
        best_coefs = coefs
        
    # Local Search Refinement
    # Try to improve the best candidate by swapping exponents or flipping signs.
    # We have time left?
    
    if time.time() < deadline - 2.0 and best_coefs is not None:
        current_exps = [i for i, c in enumerate(best_coefs) if c != 0]
        current_signs = [best_coefs[i] for i in current_exps]
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try flipping signs
            for i in range(12):
                new_signs = list(current_signs)
                new_signs[i] *= -1
                count, _ = _best_signs_for_exponents(current_exps) # This re-optimizes all signs, which is fine
                # Actually, flipping one sign might not be optimal if others change.
                # So just re-run _best_signs_for_exponents on the same exponents?
                # No, that's what we already did.
                # So flipping signs is already covered by the initial optimization.
                # We need to change EXponents.
                
            # Try moving one exponent
            for i in range(12):
                old_e = current_exps[i]
                for new_e in range(0, 41):
                    if new_e == old_e:
                        continue
                    if new_e in current_exps:
                        continue
                    new_exps = list(current_exps)
                    new_exps[i] = new_e
                    new_exps.sort()
                    
                    # Quick check: unique sums
                    us = count_unique_sums(new_exps)
                    if us > best_sq_nz + 5: # Heuristic: if unique sums is much higher, unlikely to be better
                        continue
                        
                    count, signs = _best_signs_for_exponents(new_exps)
                    if count < best_sq_nz:
                        best_sq_nz = count
                        current_exps = new_exps
                        current_signs = signs
                        improved = True
                        # Update best_coefs
                        max_exp = max(current_exps)
                        best_coefs = [0] * (max_exp + 1)
                        for e, s in zip(current_exps, current_signs):
                            best_coefs[e] = s
                        break # Restart inner loop? Or just continue?
                if improved:
                    break
            
    # Final check
    if best_coefs is None or not best_coefs or best_coefs[-1] == 0:
        return [1] * 12
        
    return best_coefs

if __name__ == "__main__":
    res = construct()
    print(res)
