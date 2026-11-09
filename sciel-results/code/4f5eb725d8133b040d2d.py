import time
import random
import numpy as np
from itertools import product as iproduct

def count_nonzero(coeffs):
    """Count nonzero terms in a coefficient list."""
    if isinstance(coeffs, list):
        return sum(1 for c in coeffs if c != 0)
    return np.count_nonzero(coeffs)

def compute_square(coeffs):
    """Compute the exact square of a polynomial using numpy convolution."""
    arr = np.array(coeffs, dtype=np.int64)
    return np.convolve(arr, arr)

def poly_mul(A, B):
    """Multiply two polynomials given as coefficient lists."""
    if not A or not B:
        return []
    result = [0] * (len(A) + len(B) - 1)
    for i, a in enumerate(A):
        if a == 0:
            continue
        for j, b in enumerate(B):
            if b == 0:
                continue
            result[i + j] += a * b
    return result

def poly_square_sparse(coeffs):
    """Compute square of a sparse polynomial, returning only nonzero positions and values."""
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    if not nz:
        return {}
    # For each pair (i,j) with i<=j, the coefficient at position i+j gets contribution
    # c_i * c_j * (2 if i!=j else 1)
    sq = {}
    for idx, (i, ci) in enumerate(nz):
        for j, cj in nz[idx:]:
            pos = i + j
            if i == j:
                sq[pos] = sq.get(pos, 0) + ci * ci
            else:
                sq[pos] = sq.get(pos, 0) + 2 * ci * cj
    # Remove zeros
    return {k: v for k, v in sq.items() if v != 0}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    random.seed(42)
    
    # Initial candidate
    best_coeffs = [1] * 12
    best_square = compute_square(best_coeffs)
    best_nonzero = count_nonzero(best_square)
    
    def evaluate(coeffs):
        square = compute_square(coeffs)
        return count_nonzero(square), square
    
    # === STRATEGY 1: Product-based construction ===
    # P(x) = A(x) * B(x) where A and B are small sparse polynomials
    # If A has m terms and B has n terms, P has at most m*n terms
    # We want P to have exactly 12 nonzero terms
    
    # Try products of small polynomials
    # A has k terms, B has l terms, k*l >= 12, and P = A*B has exactly 12 nonzeros
    
    # Start with very small A and B
    small_vals = [-2, -1, 1, 2]
    
    # Enumerate small polynomials A with up to 4 terms
    def gen_small_polys(max_terms, max_degree, vals):
        """Generate small sparse polynomials."""
        polys = []
        # 1-term polys
        for d in range(max_degree + 1):
            for v in vals:
                c = [0] * (d + 1)
                c[d] = v
                polys.append(c)
        # 2-term polys
        for d1 in range(max_degree + 1):
            for d2 in range(d1 + 1, max_degree + 1):
                for v1 in vals:
                    for v2 in vals:
                        c = [0] * (d2 + 1)
                        c[d1] = v1
                        c[d2] = v2
                        polys.append(c)
        # 3-term polys
        for d1 in range(max_degree + 1):
            for d2 in range(d1 + 1, max_degree + 1):
                for d3 in range(d2 + 1, max_degree + 1):
                    for v1 in vals:
                        for v2 in vals:
                            for v3 in vals:
                                c = [0] * (d3 + 1)
                                c[d1] = v1
                                c[d2] = v2
                                c[d3] = v3
                                polys.append(c)
        return polys
    
    # Generate candidate A polynomials (small, up to 3 terms, degree up to 20)
    A_candidates = gen_small_polys(3, 20, small_vals)
    
    # For each A, find B such that A*B has exactly 12 nonzero terms
    # and A*B has degree <= 40
    # B should also be small
    
    B_candidates = gen_small_polys(3, 20, small_vals)
    
    # This could be large, so let's be selective
    # Filter: A*B must have exactly 12 nonzero terms, degree <= 40, coeffs in [-64,64]
    
    checked = 0
    for A in A_candidates:
        if time.time() > deadline - 20:
            break
        # A must have at least 1 nonzero term
        if all(c == 0 for c in A):
            continue
        a_nz = count_nonzero(A)
        if a_nz > 4:
            continue
        for B in B_candidates:
            if time.time() > deadline - 20:
                break
            if all(c == 0 for c in B):
                continue
            b_nz = count_nonzero(B)
            if b_nz > 4:
                continue
            if a_nz * b_nz < 12:
                continue
            # Compute P = A * B
            P = poly_mul(A, B)
            if len(P) > 41:
                continue
            if P[-1] == 0:
                continue
            p_nz = count_nonzero(P)
            if p_nz != 12:
                continue
            # Check coeffs in range
            if any(abs(c) > 64 for c in P):
                continue
            # Evaluate
            nz, sq = evaluate(P)
            if nz < best_nonzero:
                best_nonzero = nz
                best_coeffs = P[:]
                best_square = sq
            checked += 1
    
    # === STRATEGY 2: Targeted local search on the best found so far ===
    current_coeffs = best_coeffs[:]
    current_nz = best_nonzero
    current_square = best_square
    
    mutation_values = [-2, -1, 1, 2, -3, -4, 3, 4]
    
    # Local search with focus on coefficient sign flips and small changes
    while time.time() < deadline:
        # Try flipping signs of individual terms
        L = len(current_coeffs)
        nonzero_idx = [i for i in range(L) if current_coeffs[i] != 0]
        if not nonzero_idx:
            break
        
        idx = random.choice(nonzero_idx)
        old_val = current_coeffs[idx]
        
        # Try negating or small changes
        candidates = []
        if old_val > 0:
            candidates.append(-old_val)
        else:
            candidates.append(-old_val)
        # Also try small perturbations
        for delta in [-2, -1, 1, 2]:
            nv = old_val + delta
            if nv != 0 and -64 <= nv <= 64 and nv != old_val:
                candidates.append(nv)
        
        if not candidates:
            continue
        
        new_val = random.choice(candidates)
        current_coeffs[idx] = new_val
        nz, sq = evaluate(current_coeffs)
        
        if nz < current_nz:
            current_nz = nz
            current_square = sq
            if nz < best_nonzero:
                best_nonzero = nz
                best_coeffs = current_coeffs[:]
                best_square = sq
        else:
            # Revert
            current_coeffs[idx] = old_val
    
    # === STRATEGY 3: Try specific algebraic identities ===
    # P(x) = (1 + x + x^k)(1 + x + x^m)(1 + x + x^n) gives 27 terms max
    # but with cancellations can be fewer. We need exactly 12.
    
    # Try P = (1 + a*x^p + b*x^q) * (1 + c*x^r + d*x^s)
    # This gives at most 9 terms, so we need more factors or terms
    
    # Try P = (1 + x^a + x^b) * (1 + x^c + x^d + x^e)
    # 3 * 5 = 15 terms max, can reduce to 12 with cancellations
    
    for a in range(1, 15):
        if time.time() > deadline - 5:
            break
        for b in range(a + 1, 20):
            for c in range(1, 15):
                for d in range(c + 1, 20):
                    for e in range(d + 1, 25):
                        # P = (1 + x^a + x^b)(1 + x^c + x^d + x^e)
                        A = [0] * (b + 1)
                        A[0] = 1
                        A[a] = 1
                        A[b] = 1
                        B = [0] * (e + 1)
                        B[0] = 1
                        B[c] = 1
                        B[d] = 1
                        B[e] = 1
                        P = poly_mul(A, B)
                        if len(P) > 41 or P[-1] == 0:
                            continue
                        p_nz = count_nonzero(P)
                        if p_nz != 12:
                            continue
                        if any(abs(v) > 64 for v in P):
                            continue
                        nz, sq = evaluate(P)
                        if nz < best_nonzero:
                            best_nonzero = nz
                            best_coeffs = P[:]
                            best_square = sq
    
    # === STRATEGY 4: Random restarts with structured initialization ===
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        # Random 12-term polynomial with small coefficients
        L = random.randint(12, 41)
        last_idx = L - 1
        other_indices = random.sample(range(L - 1), 11)
        non_zero_indices = set(other_indices) | {last_idx}
        coeffs = [0] * L
        for idx in non_zero_indices:
            coeffs[idx] = random.choice([-2, -1, 1, 2])
        nz, sq = evaluate(coeffs)
        if nz < best_nonzero:
            best_nonzero = nz
            best_coeffs = coeffs[:]
            best_square = sq
    
    return best_coeffs
