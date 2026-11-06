import time
import itertools
from collections import defaultdict
import random

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
    if not exponents:
        return None
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

def _poly_mul_sparse(a, b):
    """Multiply two sparse polynomials represented as lists of (exp, coeff)."""
    result = defaultdict(int)
    for ea, ca in a:
        for eb, cb in b:
            result[ea + eb] += ca * cb
    return [(e, c) for e, c in result.items() if c != 0]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Fallback: the seed program's answer
    # (1+x+...+x^11)^2 has 21 nonzero terms.
    # A better fallback: 1 - x + x^2 - ... + x^10 - x^11 (12 terms)
    # Square: (1-x^12)/(1-x)^2. 
    # Let's just use a known good one if we find one quickly, else fallback to simple.
    fallback = [1] * 12
    fallback_score = 21
    best_score = fallback_score
    best_candidate = fallback

    # Strategy: P(x) = A(x) * B(x) * C(x)
    # A, B, C are 4-term polynomials.
    # We search for structured exponent sets for A, B, C such that their sumset
    # has exactly 12 elements (or close, allowing for internal cancellations to reduce it).
    # Then we optimize the coefficients (signs/magnitudes) of A, B, C to minimize nonzero terms in P^2.

    def gen_4term_factors(max_exp=15):
        """Generate candidate 4-term polynomials with structured exponents."""
        polys = []
        # Pattern 1: Arithmetic progressions
        for start in range(0, 5):
            for step in range(1, 5):
                exps = [start + i * step for i in range(4)]
                if max(exps) > max_exp:
                    continue
                for signs in itertools.product([-1, 1], repeat=4):
                    coeffs = list(signs)
                    sparse = [(e, c) for e, c in zip(exps, coeffs)]
                    polys.append(sparse)
        
        # Pattern 2: Two pairs (geometric-ish or clustered)
        for s1 in range(0, 5):
            for d1 in range(1, 4):
                e1 = [s1, s1 + d1]
                for s2 in range(5, 12):
                    for d2 in range(1, 4):
                        e2 = [s2, s2 + d2]
                        exps = sorted(e1 + e2)
                        if len(set(exps)) != 4 or max(exps) > max_exp:
                            continue
                        for signs in itertools.product([-1, 1], repeat=4):
                            coeffs = list(signs)
                            sparse = [(e, c) for e, c in zip(exps, coeffs)]
                            polys.append(sparse)
                            
        # Pattern 3: Powers of 2 (sparse)
        base_exps = [0, 1, 2, 4]
        for signs in itertools.product([-1, 1], repeat=4):
            coeffs = list(signs)
            sparse = [(e, c) for e, c in zip(base_exps, coeffs)]
            polys.append(sparse)
            
        # Deduplicate
        seen = set()
        unique_polys = []
        for p in polys:
            key = tuple(sorted(p))
            if key not in seen:
                seen.add(key)
                unique_polys.append(p)
        return unique_polys

    # Generate a small library of factors
    # We don't need too many, just diverse structures
    factor_library = gen_4term_factors(max_exp=10)
    
    # Limit library size for speed
    if len(factor_library) > 100:
        # Randomly sample to keep diversity but limit size
        random.seed(42)
        factor_library = random.sample(factor_library, 100)

    # Precompute exponents for each factor
    # factor_exps[i] is the list of exponents for factor_library[i]
    
    def try_factor_triple(A_idx, B_idx, C_idx, max_iter=20):
        """
        Try a specific triple of factors.
        Optimize coefficients of A, B, C to minimize nonzero terms in P^2.
        """
        A_exp = [e for e, c in factor_library[A_idx]]
        B_exp = [e for e, c in factor_library[B_idx]]
        C_exp = [e for e, c in factor_library[C_idx]]
        
        # Check if sumset size is reasonable
        # Sumset of 4+4+4 can be up to 48, but we want ~12.
        # If the sumset is too large, it's unlikely to have only 12 nonzero terms
        # unless there are massive cancellations, which is hard to find.
        # However, we can still try.
        
        # Initialize coefficients with random signs
        # We will use local search to flip signs and change magnitudes.
        
        best_local_score = float('inf')
        best_local_coeffs = None
        
        # Try a few random initializations
        for _ in range(5):
            if time.time() > deadline:
                return None, float('inf')
            
            # Start with small integer coefficients
            # Using [-1, 1] is a good start, maybe [-2, 2]
            A_c = [random.choice([-1, 1]) for _ in range(4)]
            B_c = [random.choice([-1, 1]) for _ in range(4)]
            C_c = [random.choice([-1, 1]) for _ in range(4)]
            
            current_score = float('inf')
            current_A, current_B, current_C = A_c, B_c, C_c
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                improved = False
                
                # Flip signs in A
                for i in range(4):
                    current_A[i] = -current_A[i]
                    P_sparse = _poly_mul_sparse(
                        _poly_mul_sparse(zip(A_exp, current_A), zip(B_exp, current_B)),
                        zip(C_exp, current_C)
                    )
                    if len(P_sparse) == 12 and max(e for e, c in P_sparse) <= 40:
                        P_exp = [e for e, c in P_sparse]
                        P_c = [c for e, c in P_sparse]
                        score = _square_nonzero_count(P_exp, P_c)
                        if score < current_score:
                            current_score = score
                            improved = True
                    else:
                        # If structure breaks (e.g., terms cancel out making it not 12 terms),
                        # we might want to penalize or just ignore.
                        # But we need EXACTLY 12 nonzero terms in P.
                        # If P has != 12 terms, it's invalid.
                        # So we only accept moves that keep P having 12 terms.
                        pass
                    current_A[i] = -current_A[i]
                    
                # Flip signs in B
                for i in range(4):
                    current_B[i] = -current_B[i]
                    P_sparse = _poly_mul_sparse(
                        _poly_mul_sparse(zip(A_exp, current_A), zip(B_exp, current_B)),
                        zip(C_exp, current_C)
                    )
                    if len(P_sparse) == 12 and max(e for e, c in P_sparse) <= 40:
                        P_exp = [e for e, c in P_sparse]
                        P_c = [c for e, c in P_sparse]
                        score = _square_nonzero_count(P_exp, P_c)
                        if score < current_score:
                            current_score = score
                            improved = True
                    current_B[i] = -current_B[i]
                    
                # Flip signs in C
                for i in range(4):
                    current_C[i] = -current_C[i]
                    P_sparse = _poly_mul_sparse(
                        _poly_mul_sparse(zip(A_exp, current_A), zip(B_exp, current_B)),
                        zip(C_exp, current_C)
                    )
                    if len(P_sparse) == 12 and max(e for e, c in P_sparse) <= 40:
                        P_exp = [e for e, c in P_sparse]
                        P_c = [c for e, c in P_sparse]
                        score = _square_nonzero_count(P_exp, P_c)
                        if score < current_score:
                            current_score = score
                            improved = True
                    current_C[i] = -current_C[i]
                    
                if not improved:
                    break
            
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_coeffs = (current_A, current_B, current_C)
                
        if best_local_score == float('inf'):
            return None, float('inf')
            
        # Reconstruct P from best coeffs
        A_c, B_c, C_c = best_local_coeffs
        P_sparse = _poly_mul_sparse(
            _poly_mul_sparse(zip(A_exp, A_c), zip(B_exp, B_c)),
            zip(C_exp, C_c)
        )
        
        if len(P_sparse) != 12:
            return None, float('inf')
            
        P_exp = [e for e, c in P_sparse]
        P_c = [c for e, c in P_sparse]
        
        return P_exp, best_local_score

    # Iterate over factor triples
    # To keep it fast, we don't do all O(N^3). We sample.
    # N=100, N^3 is 1,000,000. Too many.
    # We can fix C and iterate A, B. Or just random sampling.
    
    # Let's try a structured subset first: 
    # Use factors with similar "density" or structure.
    
    # Sample random triples
    random.seed(123)
    num_samples = 5000 # Limit samples to fit in time
    sampled_indices = list(range(len(factor_library)))
    
    for _ in range(num_samples):
        if time.time() > deadline:
            break
            
        i, j, k = random.sample(sampled_indices, 3)
        
        # Quick check: Max exponent of sum
        max_e_A = max(e for e, c in factor_library[i])
        max_e_B = max(e for e, c in factor_library[j])
        max_e_C = max(e for e, c in factor_library[k])
        if max_e_A + max_e_B + max_e_C > 40:
            continue
            
        exps, score = try_factor_triple(i, j, k, max_iter=10)
        
        if exps is not None and score < best_score:
            best_score = score
            # Rebuild candidate list
            # We need the actual coefficients to build the list
            # try_factor_triple returns exps, but we need coeffs.
            # Let's modify try_factor_triple to return the candidate list directly or coeffs.
            
            # For now, let's just store the best score and re-run to get coeffs if needed.
            # Better: return the candidate list from the helper.
            
            pass # Logic update needed below

    # Refined loop to capture candidate
    best_candidate_found = None
    
    for _ in range(num_samples):
        if time.time() > deadline:
            break
            
        i, j, k = random.sample(sampled_indices, 3)
        
        max_e_A = max(e for e, c in factor_library[i])
        max_e_B = max(e for e, c in factor_library[j])
        max_e_C = max(e for e, c in factor_library[k])
        if max_e_A + max_e_B + max_e_C > 40:
            continue
            
        # Inline the optimization to return the full candidate
        A_exp = [e for e, c in factor_library[i]]
        B_exp = [e for e, c in factor_library[j]]
        C_exp = [e for e, c in factor_library[k]]
        
        best_local_score = float('inf')
        best_local_candidate = None
        
        for _ in range(3): # Fewer restarts for speed
            if time.time() > deadline:
                break
            
            A_c = [random.choice([-1, 1]) for _ in range(4)]
            B_c = [random.choice([-1, 1]) for _ in range(4)]
            C_c = [random.choice([-1, 1]) for _ in range(4)]
            
            current_score = float('inf')
            
            for _ in range(15): # Local search steps
                if time.time() > deadline:
                    break
                
                improved = False
                
                # Flip A
                for idx in range(4):
                    A_c[idx] = -A_c[idx]
                    P_sparse = _poly_mul_sparse(
                        _poly_mul_sparse(zip(A_exp, A_c), zip(B_exp, B_c)),
                        zip(C_exp, C_c)
                    )
                    if len(P_sparse) == 12 and max(e for e, c in P_sparse) <= 40:
                        P_exp = [e for e, c in P_sparse]
                        P_c = [c for e, c in P_sparse]
                        score = _square_nonzero_count(P_exp, P_c)
                        if score < current_score:
                            current_score = score
                            improved = True
                    A_c[idx] = -A_c[idx]
                    
                # Flip B
                for idx in range(4):
                    B_c[idx] = -B_c[idx]
                    P_sparse = _poly_mul_sparse(
                        _poly_mul_sparse(zip(A_exp, A_c), zip(B_exp, B_c)),
                        zip(C_exp, C_c)
                    )
                    if len(P_sparse) == 12 and max(e for e, c in P_sparse) <= 40:
                        P_exp = [e for e, c in P_sparse]
                        P_c = [c for e, c in P_sparse]
                        score = _square_nonzero_count(P_exp, P_c)
                        if score < current_score:
                            current_score = score
                            improved = True
                    B_c[idx] = -B_c[idx]
                    
                # Flip C
                for idx in range(4):
                    C_c[idx] = -C_c[idx]
                    P_sparse = _poly_mul_sparse(
                        _poly_mul_sparse(zip(A_exp, A_c), zip(B_exp, B_c)),
                        zip(C_exp, C_c)
                    )
                    if len(P_sparse) == 12 and max(e for e, c in P_sparse) <= 40:
                        P_exp = [e for e, c in P_sparse]
                        P_c = [c for e, c in P_sparse]
                        score = _square_nonzero_count(P_exp, P_c)
                        if score < current_score:
                            current_score = score
                            improved = True
                    C_c[idx] = -C_c[idx]
                    
                if not improved:
                    break
            
            if current_score < best_local_score:
                best_local_score = current_score
                if best_local_score < best_score:
                    # Build candidate
                    P_sparse = _poly_mul_sparse(
                        _poly_mul_sparse(zip(A_exp, A_c), zip(B_exp, B_c)),
                        zip(C_exp, C_c)
                    )
                    if len(P_sparse) == 12:
                        P_exp = [e for e, c in P_sparse]
                        P_c = [c for e, c in P_sparse]
                        candidate = _build_coeff_list(P_exp, P_c)
                        if candidate:
                            best_score = best_local_score
                            best_candidate = candidate

    # If we didn't find anything better than fallback, return fallback
    if best_score < fallback_score:
        return best_candidate
    else:
        return fallback

# EVOLVE-BLOCK-END
