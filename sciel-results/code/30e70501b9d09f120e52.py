import time
import random
import numpy as np
from itertools import product

def _square_nonzero_fast(exps, coeffs):
    """Compute nonzero count of square using vectorized numpy."""
    n = len(exps)
    if n == 0:
        return 0
    e = np.array(exps, dtype=np.int32)
    c = np.array(coeffs, dtype=np.int64)
    
    # Build all pairs (i, j) with i <= j
    i_idx, j_idx = np.triu_indices(n, k=0)
    degrees = e[i_idx] + e[j_idx]
    prods = c[i_idx] * c[j_idx]
    diag_mask = (i_idx == j_idx)
    prods[~diag_mask] *= 2
    
    max_deg = int(degrees.max()) if len(degrees) > 0 else 0
    sq = np.bincount(degrees, weights=prods, minlength=max_deg + 1).astype(np.int64)
    return int(np.sum(sq != 0))

def _make_list_from_exps_coeffs(exps, coeffs):
    """Convert parallel lists to coefficient list in ascending degree."""
    max_e = max(exps) if exps else 0
    lst = [0] * (max_e + 1)
    for e, c in zip(exps, coeffs):
        if e <= 40:
            lst[e] = c
    while len(lst) > 1 and lst[-1] == 0:
        lst.pop()
    return lst

def _validate(exps, coeffs):
    """Check validity."""
    if len(exps) != 12:
        return False
    seen = set()
    for e, c in zip(exps, coeffs):
        if c == 0 or c < -64 or c > 64:
            return False
        if e < 0 or e > 40:
            return False
        if e in seen:
            return False
        seen.add(e)
    return True

def _search_factor_structure(deadline, rng):
    """
    Search for P = A * B where A and B each have 6 terms.
    P^2 = A^2 * B^2. We want A^2 * B^2 to have few nonzero terms.
    
    Strategy: Find A with 6 terms such that A^2 has few terms,
    and B with 6 terms such that B^2 has few terms,
    then A^2 * B^2 will have few terms.
    """
    best_exps = None
    best_coeffs = None
    best_score = float('inf')
    
    # First, find good 6-term polynomials A where A^2 is sparse
    # A^2 has at most 21 terms (6*7/2). We want fewer.
    # For A^2 to have k terms, many e_i+e_j must collide and cancel.
    
    # Try geometric progressions and other structured sets
    candidate_6term_sets = []
    
    # Geometric: 0, d, 2d, 3d, 4d, 5d
    for d in range(1, 8):
        exps = list(range(0, 6*d, d))
        if max(exps) <= 40:
            candidate_6term_sets.append(exps)
    
    # Symmetric: 0, a, 2a, 3a, 4a, 5a (same as above)
    # Try: 0, 1, 2, 3, 4, 12 (last jump)
    for d in range(1, 6):
        for jump in range(5, 15):
            exps = [0, 1, 2, 3, 4, 4 + jump]
            if max(exps) <= 40 and len(set(exps)) == 6:
                candidate_6term_sets.append(sorted(exps))
    
    # Try: 0, a, 2a, 3a, 4a, 5a+b
    for a in range(1, 5):
        for b in range(1, 10):
            exps = [0, a, 2*a, 3*a, 4*a, 5*a + b]
            if max(exps) <= 40 and len(set(exps)) == 6:
                candidate_6term_sets.append(sorted(exps))
    
    # Random 6-term sets
    for _ in range(200):
        exps = sorted(rng.sample(range(0, 41), 6))
        candidate_6term_sets.append(exps)
    
    # For each 6-term set, find coefficient patterns that minimize A^2 nonzero terms
    good_As = []  # (exps, coeffs, score)
    
    for exps in candidate_6term_sets:
        if len(set(exps)) != 6 or max(exps) > 40:
            continue
        # Try small coefficient patterns
        best_a_score = float('inf')
        best_a_coeffs = None
        # Try all-1, alternating, and some random
        patterns = [
            [1]*6,
            [1 if i%2==0 else -1 for i in range(6)],
            [-1 if i%2==0 else 1 for i in range(6)],
            [1,1,1,1,1,-1],
            [1,1,-1,1,1,-1],
            [1,-1,1,-1,1,-1],
        ]
        for _ in range(20):
            patterns.append([rng.choice([-2,-1,1,2]) for _ in range(6)])
        
        for coeffs in patterns:
            if any(c == 0 for c in coeffs):
                continue
            score = _square_nonzero_fast(exps, coeffs)
            if score < best_a_score:
                best_a_score = score
                best_a_coeffs = coeffs[:]
        
        if best_a_coeffs is not None:
            good_As.append((exps, best_a_coeffs, best_a_score))
    
    # Sort by A^2 sparsity
    good_As.sort(key=lambda x: x[2])
    
    # Now try P = A * B for pairs of good A's
    # P has 12 terms (union of exponents from A and B, assuming disjoint)
    # P^2 = A^2 * B^2
    
    # For speed, limit to top candidates
    top_As = good_As[:50]
    
    for i in range(len(top_As)):
        if time.time() > deadline - 10:
            break
        exps_a, coeffs_a, _ = top_As[i]
        for j in range(i, len(top_As)):
            if time.time() > deadline - 5:
                break
            exps_b, coeffs_b, _ = top_As[j]
            
            # Check if exponents are disjoint (for P to have 12 terms)
            set_a = set(exps_a)
            set_b = set(exps_b)
            if set_a & set_b:
                continue
            
            exps_p = sorted(exps_a + exps_b)
            if len(exps_p) != 12 or max(exps_p) > 40:
                continue
            
            # Coefficients: P = A*B, so coeff at e_a+e_b is c_a*c_b
            # But P itself has 12 terms at positions exps_p
            # Wait, this is wrong. P = A*B means P has terms at e_a_i + e_b_j
            # That's 36 terms, not 12.
            
            # I need to reconsider. If P has 12 terms, and P = A*B where A has 6 and B has 6,
            # then P has up to 36 terms unless there's massive collision.
            # This approach is flawed for getting 12-term P.
            
            # Alternative: P = A^2 where A has 6 terms. Then P has up to 21 terms.
            # We need P to have exactly 12 terms, so A^2 must have exactly 12 terms.
            # Then P^2 = A^4.
            
            pass
    
    # Better approach: P = A^2 where A has 6 terms and A^2 has exactly 12 terms
    # Then P^2 = A^4, and we want A^4 to be sparse.
    
    for exps_a, coeffs_a, a2_score in good_As:
        if time.time() > deadline - 5:
            break
        if a2_score != 12:
            continue
        
        # P = A^2 has 12 terms. Find P's exponents and coefficients.
        # Compute A^2
        n = 6
        e = np.array(exps_a, dtype=np.int32)
        c = np.array(coeffs_a, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n, k=0)
        degrees = e[i_idx] + e[j_idx]
        prods = c[i_idx] * c[j_idx]
        diag_mask = (i_idx == j_idx)
        prods[~diag_mask] *= 2
        max_deg = int(degrees.max())
        sq = np.bincount(degrees, weights=prods, minlength=max_deg + 1).astype(np.int64)
        
        # Get nonzero positions and coefficients
        nz_pos = np.where(sq != 0)[0]
        nz_coeff = sq[nz_pos]
        
        if len(nz_pos) != 12:
            continue
        if max(nz_pos) > 40:
            continue
        if np.any(np.abs(nz_coeff) > 64):
            continue
        
        exps_p = [int(x) for x in nz_pos]
        coeffs_p = [int(x) for x in nz_coeff]
        
        # Now compute P^2 = A^4
        score = _square_nonzero_fast(exps_p, coeffs_p)
        
        if score < best_score:
            best_score = score
            best_exps = exps_p[:]
            best_coeffs = coeffs_p[:]
    
    return best_exps, best_coeffs, best_score

def _search_direct_12term(deadline, rng):
    """Direct search for 12-term polynomials with sparse squares."""
    best_exps = None
    best_coeffs = None
    best_score = float('inf')
    
    # Try structured 12-term sets
    candidate_sets = []
    
    # AP: 0,1,...,11
    candidate_sets.append(list(range(12)))
    
    # Geometric-like
    for d in range(1, 4):
        exps = list(range(0, 12*d, d))
        if max(exps) <= 40:
            candidate_sets.append(exps)
    
    # Two interleaved APs
    for d1 in range(1, 4):
        for d2 in range(1, 4):
            if d1 == d2:
                continue
            exps = list(range(0, 6*d1, d1)) + list(range(1, 6*d2+1, d2))
            if len(set(exps)) == 12 and max(exps) <= 40:
                candidate_sets.append(sorted(exps))
    
    # Symmetric: 0,1,2,3,4,5,6,7,8,9,10,20
    for tail in range(12, 41):
        exps = list(range(12))
        exps[-1] = tail
        if len(set(exps)) == 12:
            candidate_sets.append(sorted(exps))
    
    # Random
    for _ in range(500):
        exps = sorted(rng.sample(range(0, 41), 12))
        candidate_sets.append(exps)
    
    # For each set, optimize coefficients
    for exps in candidate_sets:
        if time.time() > deadline - 5:
            break
        if len(set(exps)) != 12 or max(exps) > 40:
            continue
        
        best_c_score = float('inf')
        best_c_coeffs = None
        
        # Try structured patterns
        patterns = [
            [1]*12,
            [1 if i%2==0 else -1 for i in range(12)],
            [-1 if i%2==0 else 1 for i in range(12)],
            [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1],
            [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
            [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
        ]
        for _ in range(50):
            patterns.append([rng.choice([-2,-1,1,2]) for _ in range(12)])
        
        for coeffs in patterns:
            if any(c == 0 for c in coeffs):
                continue
            if any(abs(c) > 64 for c in coeffs):
                continue
            score = _square_nonzero_fast(exps, coeffs)
            if score < best_c_score:
                best_c_score = score
                best_c_coeffs = coeffs[:]
        
        if best_c_coeffs is not None and best_c_score < best_score:
            best_score = best_c_score
            best_exps = exps[:]
            best_coeffs = best_c_coeffs[:]
    
    return best_exps, best_coeffs, best_score

def _hill_climb_general(exps, coeffs, deadline, rng):
    """General hill climbing allowing exponent and coefficient changes."""
    current_exps = exps[:]
    current_coeffs = coeffs[:]
    current_score = _square_nonzero_fast(current_exps, current_coeffs)
    best_exps = current_exps[:]
    best_coeffs = current_coeffs[:]
    best_score = current_score
    
    no_improve = 0
    
    while time.time() < deadline and no_improve < 3000:
        move_type = rng.random()
        new_exps = current_exps[:]
        new_coeffs = current_coeffs[:]
        
        if move_type < 0.35:
            # Change coefficient
            idx = rng.randrange(12)
            new_c = rng.choice([-8, -4, -3, -2, -1, 1, 2, 3, 4, 8])
            new_coeffs[idx] = new_c
        elif move_type < 0.55:
            # Move exponent by small amount
            idx = rng.randrange(12)
            old_e = new_exps[idx]
            shift = rng.choice([-3, -2, -1, 1, 2, 3])
            new_e = old_e + shift
            if new_e < 0 or new_e > 40 or new_e in new_exps:
                continue
            new_exps[idx] = new_e
            new_exps.sort()
        elif move_type < 0.7:
            # Flip sign
            idx = rng.randrange(12)
            new_coeffs[idx] = -new_coeffs[idx]
        elif move_type < 0.85:
            # Change two coefficients
            i, j = rng.sample(range(12), 2)
            new_coeffs[i] = rng.choice([-2, -1, 1, 2])
            new_coeffs[j] = rng.choice([-2, -1, 1, 2])
        else:
            # Change one coefficient to a larger value
            idx = rng.randrange(12)
            new_coeffs[idx] = rng.choice([-16, -8, -4, 4, 8, 16])
        
        if not _validate(new_exps, new_coeffs):
            continue
        
        new_score = _square_nonzero_fast(new_exps, new_coeffs)
        
        if new_score <= current_score:
            current_exps = new_exps
            current_coeffs = new_coeffs
            current_score = new_score
            if new_score < best_score:
                best_exps = new_exps[:]
                best_coeffs = new_coeffs[:]
                best_score = new_score
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Accept worse with small probability
            if rng.random() < 0.03:
                current_exps = new_exps
                current_coeffs = new_coeffs
                current_score = new_score
            no_improve += 1

    return best_exps, best_coeffs, best_score

def _exhaustive_sign_search(exps, deadline, rng):
    """Try all 2^12 sign patterns for fixed exponents."""
    best_score = float('inf')
    best_coeffs = None
    
    # Use absolute values of current best as magnitudes
    # We'll try with magnitudes [1]*12 and [1,1,1,1,1,1,2,2,2,2,2,2] etc.
    mag_sets = [
        [1]*12,
        [1,1,1,1,1,1,2,2,2,2,2,2],
        [1,1,1,1,1,1,1,1,1,1,2,2],
    ]
    
    for mags in mag_sets:
        for signs in product([1, -1], repeat=12):
            if time.time() > deadline:
                return best_score, best_coeffs
            trial_coeffs = [mags[i] * signs[i] for i in range(12)]
            score = _square_nonzero_fast(exps, trial_coeffs)
            if score < best_score:
                best_score = score
                best_coeffs = trial_coeffs[:]
    
    return best_score, best_coeffs

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    rng = random.Random(42)
    
    best_exps = None
    best_coeffs = None
    best_score = float('inf')
    
    # Phase 1: Factor-based search (P = A^2 where A has 6 terms)
    if time.time() < deadline - 30:
        e1, c1, s1 = _search_factor_structure(deadline - 20, rng)
        if e1 is not None and s1 < best_score:
            best_exps, best_coeffs, best_score = e1, c1, s1
    
    # Phase 2: Direct 12-term search
    if time.time() < deadline - 20:
        e2, c2, s2 = _search_direct_12term(deadline - 10, rng)
        if e2 is not None and s2 < best_score:
            best_exps, best_coeffs, best_score = e2, c2, s2
    
    # Phase 3: Ensure we have a valid candidate
    if best_exps is None:
        best_exps = list(range(12))
        best_coeffs = [1] * 12
        best_score = _square_nonzero_fast(best_exps, best_coeffs)
    
    # Phase 4: Exhaustive sign search on best exponents
    if time.time() < deadline - 15:
        s_sign, c_sign = _exhaustive_sign_search(best_exps, deadline - 10, rng)
        if c_sign is not None and s_sign < best_score:
            best_score = s_sign
            best_coeffs = c_sign[:]
    
    # Phase 5: General hill climbing from best
    if time.time() < deadline - 10:
        e5, c5, s5 = _hill_climb_general(best_exps, best_coeffs, deadline - 5, rng)
        if s5 < best_score:
            best_exps, best_coeffs, best_score = e5, c5, s5
    
    # Phase 6: Another sign search after hill climbing
    if time.time() < deadline - 5:
        s_sign2, c_sign2 = _exhaustive_sign_search(best_exps, deadline - 2, rng)
        if c_sign2 is not None and s_sign2 < best_score:
            best_score = s_sign2
            best_coeffs = c_sign2[:]
    
    # Phase 7: Final hill climb
    if time.time() < deadline - 2:
        e7, c7, s7 = _hill_climb_general(best_exps, best_coeffs, deadline - 1, rng)
        if s7 < best_score:
            best_exps, best_coeffs, best_score = e7, c7, s7
    
    # Convert to list
    result = _make_list_from_exps_coeffs(best_exps, best_coeffs)
    
    if len(result) > 41:
        result = result[:41]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    nz = sum(1 for c in result if c != 0)
    if nz != 12:
        # Fallback
        result = [1] * 12
    
    return result
