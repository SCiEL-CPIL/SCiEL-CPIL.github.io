import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _square_nonzero_count(exponents, coefficients):
    """Compute P(x)^2 from sparse representation and count nonzero coefficients."""
    n = len(exponents)
    square = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            e = exponents[i] + exponents[j]
            c = coefficients[i] * coefficients[j]
            if i != j:
                c *= 2
            square[e] += c
    return sum(1 for c in square.values() if c != 0)


def _build_coeff_list(exponents, coefficients, max_degree=40):
    """Build a coefficient list in ascending degree from sparse representation."""
    max_exp = max(exponents)
    if max_exp > max_degree:
        return None
    result = [0] * (max_exp + 1)
    for e, c in zip(exponents, coefficients):
        result[e] = c
    # Verify exactly 12 nonzero terms
    nonzero = sum(1 for x in result if x != 0)
    if nonzero != 12:
        return None
    # Verify final coefficient nonzero
    if result[-1] == 0:
        return None
    # Verify entries in [-64, 64]
    if any(abs(x) > 64 for x in result):
        return None
    # Verify length <= 41
    if len(result) > 41:
        return None
    return result


def _sparse_to_dense(sparse):
    """Convert list of (exp, coeff) pairs to dense coefficient list."""
    if not sparse:
        return [0]
    max_exp = max(e for e, c in sparse)
    result = [0] * (max_exp + 1)
    for e, c in sparse:
        result[e] = c
    return result


def _poly_mul_sparse(a, b):
    """Multiply two sparse polynomials represented as lists of (exp, coeff)."""
    result = defaultdict(int)
    for ea, ca in a:
        for eb, cb in b:
            result[ea + eb] += ca * cb
    return [(e, c) for e, c in result.items() if c != 0]


def _poly_square_sparse(sparse):
    """Square a sparse polynomial and return sparse result."""
    n = len(sparse)
    result = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            e = sparse[i][0] + sparse[j][0]
            c = sparse[i][1] * sparse[j][1]
            if i != j:
                c *= 2
            result[e] += c
    return [(e, c) for e, c in result.items() if c != 0]


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Fallback: the seed program's answer
    fallback = [1] * 12
    fallback_score = 21  # (1+x+...+x^11)^2 has 21 nonzero terms

    best_score = fallback_score
    best_candidate = fallback

    # Strategy: Factorization-based search
    # P(x) = A(x) * B(x) where A and B are 6-term polynomials
    # We search over structured A and B to minimize nonzero terms in P^2
    
    # Generate a library of 6-term sparse polynomials with small coefficients
    # Focus on polynomials with exponents in [0, 20] to keep degree bounded
    
    def gen_6term_polys(max_exp=25, max_coeff=3):
        """Generate candidate 6-term polynomials with structured exponents."""
        polys = []
        # Pattern 1: arithmetic progressions
        for start in range(0, 8):
            for step in range(1, 6):
                exps = [start + i * step for i in range(6)]
                if max(exps) > max_exp:
                    continue
                # Try all sign patterns (2^6 = 64)
                for signs in itertools.product([-1, 1], repeat=6):
                    coeffs = list(signs)
                    sparse = [(e, c) for e, c in zip(exps, coeffs)]
                    polys.append(sparse)
        
        # Pattern 2: two arithmetic progressions (blocks)
        for s1 in range(0, 5):
            for s2 in range(5, 15):
                for step in range(1, 4):
                    exps1 = [s1 + i * step for i in range(3)]
                    exps2 = [s2 + i * step for i in range(3)]
                    exps = sorted(exps1 + exps2)
                    if len(set(exps)) != 6 or max(exps) > max_exp:
                        continue
                    for signs in itertools.product([-1, 1], repeat=6):
                        coeffs = list(signs)
                        sparse = [(e, c) for e, c in zip(exps, coeffs)]
                        polys.append(sparse)
        
        # Pattern 3: geometric-like (powers of 2)
        base_exps = [0, 1, 2, 4, 8, 16]
        for signs in itertools.product([-1, 1], repeat=6):
            coeffs = list(signs)
            sparse = [(e, c) for e, c in zip(base_exps, coeffs)]
            polys.append(sparse)
        
        # Pattern 4: mixed - low cluster + high single
        for base in range(0, 10):
            for step in range(1, 3):
                exps_low = [base + i * step for i in range(5)]
                for high in range(max(exps_low) + 2, max_exp + 1):
                    exps = exps_low + [high]
                    if len(set(exps)) != 6:
                        continue
                    for signs in itertools.product([-1, 1], repeat=6):
                        coeffs = list(signs)
                        sparse = [(e, c) for e, c in zip(exps, coeffs)]
                        polys.append(sparse)
        
        return polys

    # Generate library (limit size for speed)
    poly_library = gen_6term_polys(max_exp=20)
    
    # Deduplicate
    seen = set()
    unique_polys = []
    for p in poly_library:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            unique_polys.append(p)
    
    # Limit library size
    if len(unique_polys) > 500:
        unique_polys = unique_polys[:500]
    
    # For each pair (A, B), compute P = A*B and check square sparsity
    # But this is O(n^2) which could be large. Use smarter approach.
    
    # Alternative: directly search 12-term polynomials with structured exponents
    # using a smarter criterion: look for exponent sets where many pairs
    # (i,j) and (k,l) have e_i+e_j = e_k+e_l (collision), enabling cancellation.
    
    # Better strategy: Use the fact that if P(x) = sum c_i x^{e_i}, then
    # P(x)^2 = sum c_i^2 x^{2e_i} + 2*sum_{i<j} c_i c_j x^{e_i+e_j}
    # To minimize nonzero terms, we want:
    # 1. Many e_i + e_j = e_k + e_l collisions
    # 2. Signs chosen so that c_i c_j + c_k c_l = 0 for colliding pairs
    
    # Search over exponent sets with high "additive energy" (many collisions)
    # Then optimize signs via local search.
    
    def additive_energy(exps):
        """Count number of quadruples (i,j,k,l) with i<j, k<l, i!=k, e_i+e_j=e_k+e_l."""
        n = len(exps)
        sums = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                s = exps[i] + exps[j]
                sums[s].append((i, j))
        count = 0
        for s, pairs in sums.items():
            if len(pairs) > 1:
                count += len(pairs) * (len(pairs) - 1) // 2
        return count

    # Generate candidate exponent sets with high additive energy
    candidate_exps = []
    
    # Arithmetic progressions have high additive energy
    for start in range(0, 10):
        for step in range(1, 8):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                candidate_exps.append(exps)
    
    # Two interleaved arithmetic progressions
    for s1 in range(0, 5):
        for s2 in range(5, 15):
            for step in range(1, 5):
                exps1 = [s1 + i * step for i in range(6)]
                exps2 = [s2 + i * step for i in range(6)]
                exps = sorted(exps1 + exps2)
                if len(set(exps)) == 12 and max(exps) <= 40:
                    candidate_exps.append(exps)
    
    # Three interleaved
    for s1 in range(0, 4):
        for s2 in range(4, 12):
            for s3 in range(12, 25):
                for step in range(1, 4):
                    exps1 = [s1 + i * step for i in range(4)]
                    exps2 = [s2 + i * step for i in range(4)]
                    exps3 = [s3 + i * step for i in range(4)]
                    exps = sorted(exps1 + exps2 + exps3)
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        candidate_exps.append(exps)
    
    # Four interleaved groups of 3
    for s1 in range(0, 3):
        for s2 in range(3, 10):
            for s3 in range(10, 20):
                for s4 in range(20, 30):
                    for step in range(1, 3):
                        exps1 = [s1 + i * step for i in range(3)]
                        exps2 = [s2 + i * step for i in range(3)]
                        exps3 = [s3 + i * step for i in range(3)]
                        exps4 = [s4 + i * step for i in range(3)]
                        exps = sorted(exps1 + exps2 + exps3 + exps4)
                        if len(set(exps)) == 12 and max(exps) <= 40:
                            candidate_exps.append(exps)
    
    # Symmetric patterns: pairs summing to constant
    for total in range(12, 41):
        for i in range(6):
            for j in range(i + 1, 6):
                e1 = i
                e2 = total - i
                e3 = j
                e4 = total - j
                # Need 12 distinct exponents, use symmetric structure
                # Try: {a, b, c, d, total-d, total-c, total-b, total-a} + 4 more
                pass  # This pattern is complex, skip for now
    
    # Quadratic-like patterns
    for a in range(1, 4):
        for b in range(0, 10):
            for c in range(0, 20):
                exps = [a * i * i + b * i + c for i in range(12)]
                if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                    candidate_exps.append(sorted(exps))
    
    # Mixed: 2 clusters of 6 with different steps
    for s1 in range(0, 5):
        for step1 in range(1, 4):
            for s2 in range(8, 20):
                for step2 in range(1, 4):
                    exps1 = [s1 + i * step1 for i in range(6)]
                    exps2 = [s2 + i * step2 for i in range(6)]
                    exps = sorted(exps1 + exps2)
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        candidate_exps.append(exps)
    
    # Deduplicate
    seen_exps = set()
    unique_exps = []
    for exps in candidate_exps:
        key = tuple(exps)
        if key not in seen_exps:
            seen_exps.add(key)
            unique_exps.append(exps)
    
    # Sort by additive energy (descending) - higher energy means more cancellation opportunities
    # Use a refined metric that also considers coefficient flexibility
    def refined_score(exps):
        ae = additive_energy(exps)
        # Bonus for having many pairs with the same sum (enables sign-based cancellation)
        sums = defaultdict(int)
        for i in range(12):
            for j in range(i + 1, 12):
                sums[exps[i] + exps[j]] += 1
        max_collision = max(sums.values()) if sums else 0
        # Penalty for very high degree (harder to work with)
        degree_penalty = max(0, max(exps) - 30)
        return ae + max_collision * 2 - degree_penalty
    
    scored_exps = []
    for exps in unique_exps:
        rs = refined_score(exps)
        scored_exps.append((rs, exps))
    scored_exps.sort(key=lambda x: -x[0])
    
    # For each top exponent set, optimize signs
    def optimize_signs(exps, max_iter=200):
        """Local search to find best sign assignment."""
        # Start with all +1
        signs = [1] * 12
        current_score = _square_nonzero_count(exps, signs)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Single flip moves
            for i in range(12):
                signs[i] = -signs[i]
                new_score = _square_nonzero_count(exps, signs)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    signs[i] = -signs[i]
            # Double flip moves
            for i in range(12):
                for j in range(i + 1, 12):
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]
                    new_score = _square_nonzero_count(exps, signs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
            if not improved:
                break
        return current_score, signs
    
    def optimize_coeffs(exps, max_iter=100):
        """Local search to find best coefficient assignment in {-1, 1, -2, 2, -3, 3}."""
        # Start with all +1
        coeffs = [1] * 12
        current_score = _square_nonzero_count(exps, coeffs)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(12):
                old_val = coeffs[i]
                for new_val in [-3, -2, -1, 1, 2, 3]:
                    if new_val == old_val:
                        continue
                    coeffs[i] = new_val
                    new_score = _square_nonzero_count(exps, coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    elif new_score == current_score:
                        # Keep the change if it doesn't worsen (for diversity)
                        pass
                    else:
                        coeffs[i] = old_val
                        break
            if not improved:
                break
        return current_score, coeffs

    # Try top exponent sets
    for ae, exps in scored_exps[:300]:
        if time.time() > deadline:
            break
        
        # First try sign-only optimization (fast)
        score, signs = optimize_signs(exps, max_iter=30)
        
        if score < best_score:
            best_score = score
            candidate = _build_coeff_list(exps, signs)
            if candidate is not None:
                best_candidate = candidate
        
        # If we have time, also try coefficient optimization
        if time.time() < deadline - 5:
            score2, coeffs = optimize_coeffs(exps, max_iter=20)
            
            if score2 < best_score:
                best_score = score2
                candidate = _build_coeff_list(exps, coeffs)
                if candidate is not None:
                    best_candidate = candidate
    
    # Also try the factorization approach for a few pairs
    if time.time() < deadline - 10:
        # Try P = A * B where A, B are 6-term
        # Sample a subset of the poly library
        sample_size = min(50, len(unique_polys))
        import random
        random.seed(123)
        sampled = random.sample(unique_polys, sample_size)
        
        for i in range(sample_size):
            if time.time() > deadline:
                break
            for j in range(i + 1, sample_size):
                if time.time() > deadline:
                    break
                A = sampled[i]
                B = sampled[j]
                # P = A * B
                P_sparse = _poly_mul_sparse(A, B)
                # Check if P has exactly 12 nonzero terms
                if len(P_sparse) != 12:
                    continue
                # Check degree
                if P_sparse[-1][0] > 40:
                    continue
                # Compute square
                square_sparse = _poly_square_sparse(P_sparse)
                score = len(square_sparse)
                
                if score < best_score:
                    best_score = score
                    exps = [e for e, c in P_sparse]
                    coeffs = [c for e, c in P_sparse]
                    candidate = _build_coeff_list(exps, coeffs)
                    if candidate is not None:
                        best_candidate = candidate
    
    # If we found something better than fallback, use it
    if best_score < fallback_score:
        return best_candidate
    
    # Otherwise return fallback
    return fallback

# EVOLVE-BLOCK-END
