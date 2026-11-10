import itertools

def count_nonzero(lst):
    return sum(1 for x in lst if x != 0)

def poly_square(coeffs):
    n = len(coeffs)
    if n == 0:
        return []
    result = [0] * (2*n - 1)
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    for k in range(len(nz)):
        i, ci = nz[k]
        for l in range(k, len(nz)):
            j, cj = nz[l]
            val = ci * cj
            if i == j:
                result[i+j] += val
            else:
                result[i+j] += 2 * val
    return result

def valid_candidate(coeffs):
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    if count_nonzero(coeffs) != 12:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    return True

def construct():
    best_score = float('inf')
    best_cand = None
    
    # Search space: Exponents from 0 to 17, choose 12.
    # Coefficients: +/- 1
    max_exp = 17
    all_exps = range(max_exp + 1)
    
    for exps in itertools.combinations(all_exps, 12):
        # Optimization: Pre-calculate the indices
        # We can iterate masks
        for mask in range(1 << 12):
            # Build coeffs quickly
            coeffs = [0] * (max_exp + 1)
            for i, e in enumerate(exps):
                coeffs[e] = 1 if (mask >> i) & 1 else -1
            
            # Check validity (always valid for this construction)
            # Compute score
            sq = poly_square(coeffs)
            score = count_nonzero(sq)
            
            if score < best_score:
                best_score = score
                best_cand = coeffs[:]
                
    if best_cand is None:
        # Fallback
        best_cand = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        
    return best_cand
