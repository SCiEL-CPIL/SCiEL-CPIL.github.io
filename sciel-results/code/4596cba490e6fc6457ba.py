"""
Structural approach: exploit algebraic identities and collision patterns.
Key ideas:
1. If P = Q * R where Q, R are sparse, P^2 = Q^2 * R^2.
2. Cyclotomic-like patterns: exponents in AP with coefficients from roots of unity.
3. Direct collision maximization: choose exponents so many pairs share sums.
4. Coefficient cancellation: at each exponent, sum of contributing products = 0.
"""
import time
import random
import numpy as np

def _square_nonzero_count(poly):
    """Count non-zero terms in poly^2."""
    if not poly:
        return 0
    sq = np.convolve(poly, poly)
    return int(np.count_nonzero(sq))

def _is_valid(poly):
    """Check all constraints."""
    if len(poly) == 0 or len(poly) > 41:
        return False
    if poly[-1] == 0:
        return False
    nz = sum(1 for x in poly if x != 0)
    if nz != 12:
        return False
    if any(abs(x) > 64 for x in poly):
        return False
    return True

def _generate_cyclotomic_candidates():
    """
    Generate candidates based on cyclotomic polynomial structures.
    For a prime p, the p-th cyclotomic polynomial has p-1 terms.
    Phi_p(x) = (x^p - 1)/(x - 1) = 1 + x + x^2 + ... + x^{p-1}
    
    But we need 12 terms. Let's use products of cyclotomic-like factors.
    
    Key identity: (1 + x + x^2 + ... + x^{k-1})^2 has a triangular structure.
    
    More useful: (1 - x + x^2 - x^3 + ... + x^{11})^2
    For alternating signs on AP with step 1:
    P(x) = sum_{i=0}^{11} (-1)^i x^i
    P(x) = (1 - (-x)^12) / (1 + x) = (1 + x^12)/(1+x)
    P(x)^2 = (1 + x^12)^2 / (1+x)^2 = (1 + 2x^12 + x^24) / (1 + 2x + x^2)
    This is a rational function, not a polynomial! So this doesn't directly work.
    
    Better: use P(x) = (1 + x^a + x^b + ...) where the exponents form a group structure.
    
    Let's try: P(x) = (1 + x + x^2)(1 + x^3 + x^6)(1 + x^9)
    This has 3*3*2 = 18 terms, too many.
    
    P(x) = (1 + x^2 + x^4)(1 + x^3 + x^6 + x^9)
    = 1 + x^2 + x^4 + x^3 + x^5 + x^7 + x^6 + x^8 + x^10
    That's 9 terms.
    
    Let's try products that give exactly 12 terms.
    """
    candidates = []
    
    # Product of two polynomials
    # (1 + x^a + x^b) * (1 + x^c + x^d + x^e + x^f) = 15 terms, too many
    # (1 + x^a + x^b + x^c) * (1 + x^d + x^e) = 6 terms
    # (1 + x^a + x^b + x^c + x^d) * (1 + x^e + x^f + x^g) = 15 terms
    
    # Let's try: (1 + x + x^2 + x^3) * (1 + x^4 + x^8)
    # = 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11
    # That's 12 terms! Exponents: 0,1,2,3,4,5,6,7,8,9,10,11
    # So P(x) = (1+x+x^2+x^3)(1+x^4+x^8)
    # Let's verify: (1+x+x^2+x^3)(1+x^4+x^8)
    # = 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11
    # Yes! 12 terms.
    
    poly = [0] * 41
    for i in range(12):
        poly[i] = 1
    candidates.append(poly[:])
    
    # Now try with signs: (1 - x + x^2 - x^3)(1 + x^4 + x^8)
    # = 1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^10 - x^11
    poly2 = [0] * 41
    for i in range(12):
        poly2[i] = 1 if i % 2 == 0 else -1
    candidates.append(poly2[:])
    
    # (1 + x - x^2 + x^3)(1 + x^4 + x^8)
    poly3 = [0] * 41
    for i in range(4):
        poly3[i] = [1, 1, -1, 1][i]
    for i in range(4, 12, 4):
        poly3[i] = [1, 1, -1, 1][i % 4]
    # Wait, this isn't a product structure. Let me be more careful.
    
    # General: (a0 + a1*x + a2*x^2 + a3*x^3) * (1 + x^4 + x^8)
    # = a0 + a1*x + a2*x^2 + a3*x^3 + a0*x^4 + a1*x^5 + a2*x^6 + a3*x^7 + a0*x^8 + a1*x^9 + a2*x^10 + a3*x^11
    # So coefficients are: [a0, a1, a2, a3, a0, a1, a2, a3, a0, a1, a2, a3]
    
    for a0 in [-2, -1, 1, 2]:
        for a1 in [-2, -1, 1, 2]:
            for a2 in [-2, -1, 1, 2]:
                for a3 in [-2, -1, 1, 2]:
                    if a0 == 0 or a1 == 0 or a2 == 0 or a3 == 0:
                        continue
                    poly = [0] * 41
                    coeffs = [a0, a1, a2, a3]
                    for i in range(12):
                        poly[i] = coeffs[i % 4]
                    if _is_valid(poly):
                        candidates.append(poly[:])
    
    # Another structure: (1 + x^a)(1 + x^b)(1 + x^c)(1 + x^d)(1 + x^e)(1 + x^f)
    # This gives 2^6 = 64 terms, way too many.
    
    # (1 + x + x^2)(1 + x^3 + x^6)(1 + x^9)
    # = (1 + x + x^2)(1 + x^3 + x^6)(1 + x^9)
    # First: (1 + x + x^2)(1 + x^3 + x^6) = 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8
    # Then: (1 + x + ... + x^8)(1 + x^9) = 1 + x + ... + x^8 + x^9 + x^10 + ... + x^17
    # That's 18 terms.
    
    # Let's try: (1 + x^2 + x^4 + x^6)(1 + x^3 + x^9)
    # = 1 + x^2 + x^4 + x^6 + x^3 + x^5 + x^7 + x^9 + x^6 + x^8 + x^10 + x^12
    # Wait: x^6 * x^9 = x^15, not x^12. Let me redo.
    # (1 + x^2 + x^4 + x^6)(1 + x^3 + x^9)
    # = 1 + x^3 + x^9 + x^2 + x^5 + x^11 + x^4 + x^7 + x^13 + x^6 + x^9 + x^15
    # Exponents: 0, 2, 3, 4, 5, 6, 7, 9, 9, 11, 13, 15
    # x^9 appears twice! So coefficient at x^9 is 1+1=2.
    # That's 11 distinct exponents, but one has coefficient 2.
    # Total non-zero terms: 11. Not 12.
    
    # Let me try: (1 + x^2 + x^4 + x^6)(1 + x^3 + x^10)
    # = 1 + x^3 + x^10 + x^2 + x^5 + x^12 + x^4 + x^7 + x^14 + x^6 + x^9 + x^16
    # Exponents: 0, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 16 - all distinct!
    # 12 terms.
    poly4 = [0] * 41
    for e in [0, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 16]:
        poly4[e] = 1
    candidates.append(poly4[:])
    
    # With signs: (1 - x^2 + x^4 - x^6)(1 + x^3 + x^10)
    poly5 = [0] * 41
    for e, c in zip([0, 2, 4, 6], [1, -1, 1, -1]):
        poly5[e] = c
    for e, c in zip([3, 10], [1, 1]):
        poly5[e] = c
    # Product:
    # 1*1 + 1*x^3 + 1*x^10
    # -x^2*1 + -x^2*x^3 + -x^2*x^10 = -x^2 - x^5 - x^12
    # x^4*1 + x^4*x^3 + x^4*x^10 = x^4 + x^7 + x^14
    # -x^6*1 + -x^6*x^3 + -x^6*x^10 = -x^6 - x^9 - x^16
    # Exponents: 0, 3, 10, 2, 5, 12, 4, 7, 14, 6, 9, 16
    # Coefficients: 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1
    poly5b = [0] * 41
    for e, c in zip([0, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 16],
                     [1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1]):
        poly5b[e] = c
    candidates.append(poly5b[:])
    
    return candidates

def _generate_collision_candidates():
    """
    Generate candidates designed for maximum collision in P^2.
    
    Key: if exponents form an arithmetic progression, sums e_i + e_j form a 
    structured set. For AP with step d, the sums cover a range with step d.
    
    For 12 terms in AP: e_i = a + i*d for i=0..11
    Sums: 2a + (i+j)*d for 0 <= i <= j <= 11
    Number of distinct sums: from 2a to 2a+22d, step d, so 23 values.
    Diagonals: 2a, 2a+2d, ..., 2a+22d (12 values)
    Off-diagonals: 2a+d, 2a+3d, ..., 2a+21d (11 values)
    
    Total 23 positions. To minimize non-zero, we want cancellation.
    
    If all coefficients are 1: P^2 has all positive coefficients, no cancellation.
    If alternating signs: some cancellation.
    
    Let's try: P(x) = sum_{i=0}^{11} c_i x^{i*d} where c_i are chosen for cancellation.
    """
    candidates = []
    
    for d in [1, 2, 3]:
        for a in [0]:
            # Try all sign patterns with small coefficients
            # Focus on patterns that might create cancellation
            patterns = [
                [1]*12,
                [1 if i%2==0 else -1 for i in range(12)],
                [1 if i%3==0 else -1 for i in range(12)],
                [1 if i%4==0 else -1 for i in range(12)],
                [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
            ]
            for pat in patterns:
                poly = [0] * 41
                for i in range(12):
                    e = a + i * d
                    if e < 41:
                        poly[e] = pat[i]
                if _is_valid(poly):
                    candidates.append(poly[:])
    
    return candidates

def _local_search(poly, max_iters=3000, deadline=None):
    """
    Local search with smart moves:
    1. Change coefficient value (from {-2,-1,1,2} or small range)
    2. Shift a term's exponent
    3. Swap two terms
    4. Add/remove... no, must keep exactly 12 terms.
    """
    current = poly[:]
    current_score = _square_nonzero_count(current)
    best = current[:]
    best_score = current_score
    
    nz_indices = [i for i, v in enumerate(current) if v != 0]
    
    temp = 2.0
    cooling = 0.9995
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Change coefficient value
            idx = random.choice(nz_indices)
            old_val = current[idx]
            new_val = random.choice([-3, -2, -1, 1, 2, 3])
            if new_val == old_val or abs(new_val) > 64:
                continue
            current[idx] = new_val
            new_score = _square_nonzero_count(current)
            delta = new_score - current_score
            if delta < 0 or (temp > 0.01 and random.random() < np.exp(-delta / temp)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = current[:]
            else:
                current[idx] = old_val
                
        elif move_type < 0.7:
            # Shift exponent
            idx = random.choice(nz_indices)
            val = current[idx]
            shift = random.choice([-2, -1, 1, 2])
            new_idx = idx + shift
            if new_idx < 0 or new_idx >= len(current) or new_idx == idx:
                continue
            if current[new_idx] != 0:
                continue
            current[idx] = 0
            current[new_idx] = val
            new_score = _square_nonzero_count(current)
            delta = new_score - current_score
            if delta < 0 or (temp > 0.01 and random.random() < np.exp(-delta / temp)):
                current_score = new_score
                nz_indices.remove(idx)
                nz_indices.append(new_idx)
                if new_score < best_score:
                    best_score = new_score
                    best = current[:]
            else:
                current[new_idx] = 0
                current[idx] = val
                
        elif move_type < 0.85:
            # Swap positions of two terms
            if len(nz_indices) < 2:
                continue
            i1, i2 = random.sample(nz_indices, 2)
            v1, v2 = current[i1], current[i2]
            current[i1], current[i2] = v2, v1
            new_score = _square_nonzero_count(current)
            delta = new_score - current_score
            if delta < 0 or (temp > 0.01 and random.random() < np.exp(-delta / temp)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = current[:]
            else:
                current[i1], current[i2] = v1, v2
                
        else:
            # Random restart: pick a new random exponent for one term
            idx = random.choice(nz_indices)
            val = current[idx]
            new_idx = random.randint(0, len(current) - 1)
            if new_idx == idx or current[new_idx] != 0:
                continue
            current[idx] = 0
            current[new_idx] = val
            new_score = _square_nonzero_count(current)
            delta = new_score - current_score
            if delta < 0 or (temp > 0.01 and random.random() < np.exp(-delta / temp)):
                current_score = new_score
                nz_indices.remove(idx)
                nz_indices.append(new_idx)
                if new_score < best_score:
                    best_score = new_score
                    best = current[:]
            else:
                current[new_idx] = 0
                current[idx] = val
                
        temp *= cooling
    
    return best, best_score

def _try_factorization():
    """
    Try to find P = A * B where A and B are sparse, and P^2 is very sparse.
    
    If A has m terms and B has n terms, P has at most m*n terms.
    We need m*n = 12, so (m,n) in {(1,12),(2,6),(3,4),(4,3),(6,2),(12,1)}.
    
    For (2,6): A = a0 + a1*x^p, B = b0 + b1*x^q + ... + b5*x^r
    P = a0*b0 + (a0*b1 + a1*b0)*x^min(p,q) + ...
    
    Let's focus on A with 2 terms, B with 6 terms.
    A(x) = 1 + s*x^a  (s = ±1, ±2)
    B(x) = sum_{j=0}^{5} c_j * x^{e_j}
    
    P(x) = B(x) + s*x^a * B(x)
    P^2 = B^2 + 2s*x^a*B^2 + s^2*x^{2a}*B^2
    
    For P^2 to be sparse, we need B^2, x^a*B^2, x^{2a}*B^2 to have overlapping supports
    with cancellation.
    """
    candidates = []
    
    for s in [-2, -1, 1, 2]:
        for a in range(1, 20):
            # B has 6 terms
            # Try B with exponents in AP
            for step in [1, 2, 3]:
                for start in range(0, 5):
                    exps = [start + j * step for j in range(6)]
                    if max(exps) + a >= 41:
                        continue
                    # Try different coefficient patterns for B
                    for b_coeffs in [[1]*6, [1 if j%2==0 else -1 for j in range(6)]]:
                        B = [0] * 41
                        for j in range(6):
                            B[exps[j]] = b_coeffs[j]
                        # P = B + s * x^a * B
                        P = [0] * 41
                        for i in range(41):
                            if B[i] != 0:
                                P[i] += B[i]
                                if i + a < 41:
                                    P[i + a] += s * B[i]
                        if _is_valid(P):
                            candidates.append(P[:])
    
    # Also try A with 3 terms, B with 4 terms
    for a0 in [-1, 1]:
        for a1 in [-1, 1]:
            for a2 in [-1, 1]:
                for p in range(1, 5):
                    for q in range(p + 1, 10):
                        for r in range(1, 15):
                            # A = a0 + a1*x^p + a2*x^q
                            # B = b0 + b1*x^r + b2*x^{2r} + b3*x^{3r}
                            for b0 in [-1, 1]:
                                for b1 in [-1, 1]:
                                    for b2 in [-1, 1]:
                                        for b3 in [-1, 1]:
                                            A = [0] * 41
                                            A[0] = a0
                                            A[p] = a1
                                            A[q] = a2
                                            B = [0] * 41
                                            B[0] = b0
                                            B[r] = b1
                                            B[2*r] = b2
                                            B[3*r] = b3
                                            # P = A * B
                                            P = [0] * 41
                                            for i in range(41):
                                                if A[i] == 0:
                                                    continue
                                                for j in range(41 - i):
                                                    if B[j] != 0:
                                                        P[i + j] += A[i] * B[j]
                                            if _is_valid(P):
                                                candidates.append(P[:])
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate candidates from multiple structural sources
    all_candidates = []
    
    # 1. Cyclotomic/product structures
    all_candidates.extend(_generate_cyclotomic_candidates())
    
    # 2. Collision-optimized AP structures
    all_candidates.extend(_generate_collision_candidates())
    
    # 3. Factorization-based
    all_candidates.extend(_try_factorization())
    
    # 4. Random structured candidates
    for _ in range(100):
        # Random exponents with some structure
        exps = sorted(random.sample(range(0, 41), 12))
        signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
        coeffs = [1 if random.random() > 0.3 else 2 for _ in range(12)]
        poly = [0] * 41
        for i in range(12):
            poly[exps[i]] = signs[i] * coeffs[i]
        if _is_valid(poly):
            all_candidates.append(poly[:])
    
    if not all_candidates:
        return [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in all_candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    best_poly = None
    best_score = 1000
    
    # First pass: quick evaluation
    for cand in unique_candidates:
        if time.time() > deadline - 5.0:
            break
        score = _square_nonzero_count(cand)
        if score < best_score:
            best_score = score
            best_poly = cand[:]
    
    # Second pass: local search on best candidates
    # Sort by initial score
    scored = []
    for cand in unique_candidates:
        if time.time() > deadline - 10.0:
            break
        score = _square_nonzero_count(cand)
        scored.append((score, cand))
    scored.sort(key=lambda x: x[0])
    
    # Local search on top candidates
    for score, cand in scored[:50]:
        if time.time() > deadline - 5.0:
            break
        cand_deadline = time.time() + min(3.0, deadline - time.time() - 2.0)
        if cand_deadline <= time.time():
            break
        refined, refined_score = _local_search(cand, max_iters=2000, deadline=cand_deadline)
        if refined_score < best_score:
            best_score = refined_score
            best_poly = refined[:]
        
        if best_score <= 13:
            break
    
    if best_poly is None:
        return [1] * 12
    
    return best_poly

