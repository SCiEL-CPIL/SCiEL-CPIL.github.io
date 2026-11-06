import time
import random
import numpy as np
from itertools import combinations, product

T_START = time.time()
DEADLINE = 85.0

def _square_nonzero_count(exps, coeffs):
    """Compute nonzero count of square using vectorized numpy."""
    n = len(exps)
    if n == 0:
        return 0
    e = np.array(exps, dtype=np.int64)
    c = np.array(coeffs, dtype=np.int64)
    
    i_idx, j_idx = np.triu_indices(n, k=0)
    degrees = e[i_idx] + e[j_idx]
    prods = c[i_idx] * c[j_idx]
    diag_mask = (i_idx == j_idx)
    prods[~diag_mask] *= 2
    
    max_deg = int(degrees.max()) if len(degrees) > 0 else 0
    sq = np.bincount(degrees, weights=prods, minlength=max_deg + 1).astype(np.int64)
    return int(np.sum(sq != 0))

def _valid_candidate(coeffs):
    if not isinstance(coeffs, list):
        return False
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    for v in coeffs:
        if v < -64 or v > 64:
            return False
    nnz = sum(1 for v in coeffs if v != 0)
    if nnz != 12:
        return False
    return True

def _make_list_from_exps_coeffs(exps, coeffs):
    """Convert parallel lists to coefficient list in ascending degree."""
    if len(exps) != len(coeffs):
        return None
    if len(exps) != 12:
        return None
    seen = set()
    for e, c in zip(exps, coeffs):
        if c == 0 or c < -64 or c > 64:
            return None
        if e < 0 or e > 40:
            return None
        if e in seen:
            return None
        seen.add(e)
    max_e = max(exps)
    lst = [0] * (max_e + 1)
    for e, c in zip(exps, coeffs):
        lst[e] = c
    while len(lst) > 1 and lst[-1] == 0:
        lst.pop()
    return lst

def construct():
    best_count = 100
    best_coeffs = None
    end_time = T_START + DEADLINE
    rng = random.Random(42)
    
    def expired():
        return time.time() > end_time
    
    def update_best(exps, coeffs):
        nonlocal best_count, best_coeffs
        if len(exps) != 12:
            return
        c = _square_nonzero_count(exps, coeffs)
        if c < best_count:
            lst = _make_list_from_exps_coeffs(exps, coeffs)
            if lst is not None:
                best_count = c
                best_coeffs = lst

    def _hill_climb(exps, coeffs, deadline, rng_local, allow_exponent_change=True):
        """Hill climbing with optional exponent changes."""
        current_exps = exps[:]
        current_coeffs = coeffs[:]
        current_score = _square_nonzero_count(current_exps, current_coeffs)
        best_exps = current_exps[:]
        best_coeffs = current_coeffs[:]
        best_score = current_score
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve < max_no_improve:
            move_type = rng_local.random()
            new_exps = current_exps[:]
            new_coeffs = current_coeffs[:]
            
            if move_type < 0.4:
                idx = rng_local.randrange(12)
                new_c = rng_local.choice([-8, -4, -3, -2, -1, 1, 2, 3, 4, 8])
                new_coeffs[idx] = new_c
            elif move_type < 0.6:
                idx = rng_local.randrange(12)
                new_coeffs[idx] = -new_coeffs[idx]
            elif move_type < 0.75 and allow_exponent_change:
                idx = rng_local.randrange(12)
                shift = rng_local.choice([-3, -2, -1, 1, 2, 3])
                new_e = new_exps[idx] + shift
                if 0 <= new_e <= 40 and new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    continue
            else:
                i, j = rng_local.sample(range(12), 2)
                new_coeffs[i] = rng_local.choice([-4, -2, -1, 1, 2, 4])
                new_coeffs[j] = rng_local.choice([-4, -2, -1, 1, 2, 4])
            
            valid = True
            seen = set()
            for e, c in zip(new_exps, new_coeffs):
                if c == 0 or c < -64 or c > 64 or e < 0 or e > 40 or e in seen:
                    valid = False
                    break
                seen.add(e)
            if not valid:
                continue
            
            new_score = _square_nonzero_count(new_exps, new_coeffs)
            
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
                if rng_local.random() < 0.03:
                    current_exps = new_exps
                    current_coeffs = new_coeffs
                    current_score = new_score
                no_improve += 1
        
        return best_exps, best_coeffs, best_score

    # --- Phase 1: Product-of-two-factors structural search ---
    # P = A * B where A has 6 terms, B has 6 terms
    # P^2 = A^2 + 2AB + B^2
    # Key insight: if A and B have well-separated exponents, we can get
    # systematic cancellations in the 2AB cross terms.
    
    def _try_factor_pair(a_exps, a_coeffs, b_exps, b_coeffs):
        """Try P = A*B and compute square nonzero count."""
        # Compute P coefficients via convolution
        p_exps = []
        p_coeffs = []
        for ae, ac in zip(a_exps, a_coeffs):
            for be, bc in zip(b_exps, b_coeffs):
                p_exps.append(ae + be)
                p_coeffs.append(ac * bc)
        
        # Sort and merge same-degree terms
        if not p_exps:
            return None, 999
        
        pairs = sorted(zip(p_exps, p_coeffs))
        merged_exps = []
        merged_coeffs = []
        i = 0
        while i < len(pairs):
            e = pairs[i][0]
            c = 0
            while i < len(pairs) and pairs[i][0] == e:
                c += pairs[i][1]
                i += 1
            if c != 0:
                merged_exps.append(e)
                merged_coeffs.append(c)
        
        if len(merged_exps) != 12:
            return None, 999
        
        # Check bounds
        for e, c in zip(merged_exps, merged_coeffs):
            if e < 0 or e > 40 or c < -64 or c > 64:
                return None, 999
        
        score = _square_nonzero_count(merged_exps, merged_coeffs)
        return (merged_exps, merged_coeffs), score

    # Strategy: A = sum of 6 terms with exponents {a0, a0+g, ..., a0+5g}
    #           B = sum of 6 terms with exponents {b0, b0+h, ..., b0+5h}
    # Then P has exponents {a0+i*g + b0+j*h for i,j in 0..5}
    # For P to have exactly 12 terms, we need exactly 12 distinct sums
    # with nonzero coefficients.
    
    # Key structural idea: if g and h are "incommensurate" enough,
    # most a_i + b_j are distinct (36 terms), but we need exactly 12.
    # So we need heavy collisions: many (i,j) pairs giving same sum.
    
    # Alternative: A and B share structure. E.g., A = x^p * C, B = x^q * C
    # where C has 6 terms. Then P = x^(p+q) * C^2.
    # C^2 has at most 21 terms (6*7/2). We need C^2 to have exactly 12 terms.
    # This reduces to finding a 6-term polynomial whose square has 12 terms!
    
    def _find_6term_square_12():
        """Find 6-term polynomials whose square has exactly 12 nonzero terms."""
        results = []
        # Try arithmetic progressions for the 6 terms
        for gap in range(1, 15):
            for start in range(0, 41 - 5*gap + 1):
                if expired():
                    break
                exps = [start + i*gap for i in range(6)]
                # Try all sign patterns (2^6 = 64)
                for mask in range(64):
                    if expired():
                        break
                    coeffs = [1 if (mask >> i) & 1 else -1 for i in range(6)]
                    score = _square_nonzero_count(exps, coeffs)
                    if score == 12:
                        results.append((exps[:], coeffs[:], score))
                # Try some magnitude variations
                for mag_pattern in [[1]*6, [1,1,1,-1,-1,-1], [1,-1,1,-1,1,-1], [2,1,1,1,1,1]]:
                    if expired():
                        break
                    score = _square_nonzero_count(exps, mag_pattern)
                    if score == 12:
                        results.append((exps[:], mag_pattern[:], score))
        
        # Try geometric-like progressions
        for base in [2, 3]:
            for start in range(0, 5):
                if expired():
                    break
                exps = [start + base**i for i in range(6)]
                if max(exps) > 40:
                    continue
                for mask in range(64):
                    if expired():
                        break
                    coeffs = [1 if (mask >> i) & 1 else -1 for i in range(6)]
                    score = _square_nonzero_count(exps, coeffs)
                    if score == 12:
                        results.append((exps[:], coeffs[:], score))
        
        return results
    
    # Phase 1a: Find 6-term polys with square having 12 terms
    six_term_results = _find_6term_square_12()
    
    # For each such C, P = x^s * C^2 gives a 12-term polynomial
    # with P^2 = x^(2s) * C^4
    # We need C^4 to have few nonzero terms.
    for exps6, coeffs6, _ in six_term_results:
        if expired():
            break
        # P = C^2 (shift by 0)
        # Compute C^2
        c2_exps = []
        c2_coeffs = []
        for i in range(6):
            for j in range(i, 6):
                e = exps6[i] + exps6[j]
                c = coeffs6[i] * coeffs6[j]
                if i != j:
                    c *= 2
                c2_exps.append(e)
                c2_coeffs.append(c)
        
        # Merge
        pairs = sorted(zip(c2_exps, c2_coeffs))
        merged_exps = []
        merged_coeffs = []
        i = 0
        while i < len(pairs):
            e = pairs[i][0]
            c = 0
            while i < len(pairs) and pairs[i][0] == e:
                c += pairs[i][1]
                i += 1
            if c != 0:
                merged_exps.append(e)
                merged_coeffs.append(c)
        
        if len(merged_exps) == 12:
            for s in range(0, 41 - max(merged_exps) + 1):
                if expired():
                    break
                shifted_exps = [e + s for e in merged_exps]
                if max(shifted_exps) > 40:
                    break
                # P^2 = x^(2s) * C^4, so nonzero count of P^2 = nonzero count of C^4
                # Compute C^4 = (C^2)^2
                score = _square_nonzero_count(merged_exps, merged_coeffs)
                update_best(shifted_exps, merged_coeffs)
    
    # Phase 1b: Direct product search - A and B with different structures
    # A has exponents in low range, B in high range, with overlap creating cancellations
    
    for a_gap in [1, 2, 3, 4]:
        for b_gap in [1, 2, 3, 4]:
            for a_start in range(0, 10):
                for b_start in range(10, 30):
                    if expired():
                        break
                    a_exps = [a_start + i*a_gap for i in range(6)]
                    b_exps = [b_start + i*b_gap for i in range(6)]
                    if max(a_exps) > 40 or max(b_exps) > 40:
                        continue
                    
                    # Try sign patterns for A and B
                    for a_mask in range(64):
                        if expired():
                            break
                        a_coeffs = [1 if (a_mask >> i) & 1 else -1 for i in range(6)]
                        for b_mask in range(64):
                            if expired():
                                break
                            b_coeffs = [1 if (b_mask >> i) & 1 else -1 for i in range(6)]
                            result, score = _try_factor_pair(a_exps, a_coeffs, b_exps, b_coeffs)
                            if result is not None and score < best_count:
                                best_count = score
                                lst = _make_list_from_exps_coeffs(result[0], result[1])
                                if lst is not None:
                                    best_coeffs = lst
    
    # Phase 1c: Structured initial candidates (original approach)
    for gap in range(1, 8):
        for start in range(0, 41 - 11*gap + 1):
            if expired():
                break
            pos = [start + i*gap for i in range(12)]
            update_best(pos, [1]*12)
            update_best(pos, [1 if i%2==0 else -1 for i in range(12)])
            update_best(pos, [1]*6 + [-1]*6)
    
    for a in range(0, 15):
        for b in range(a+8, 35):
            if expired():
                break
            pos = list(range(a, a+6)) + list(range(b, b+6))
            if max(pos) < 41:
                update_best(pos, [1]*12)
                update_best(pos, [1]*6 + [-1]*6)
    
    # --- Phase 2: Random initialization with hill climbing ---
    for restart in range(20):
        if expired():
            break
        
        while True:
            if expired():
                break
            pos = sorted(rng.sample(range(41), 12))
            if max(pos) < 41:
                break
        
        for pattern in range(5):
            if expired():
                break
            
            if pattern == 0:
                coeffs = [1]*12
            elif pattern == 1:
                coeffs = [1 if i%2==0 else -1 for i in range(12)]
            elif pattern == 2:
                coeffs = [1]*6 + [-1]*6
            elif pattern == 3:
                coeffs = [rng.choice([-1, 1]) for _ in range(12)]
            else:
                coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
            
            new_exps, new_coeffs, new_score = _hill_climb(
                pos, coeffs, end_time, rng, allow_exponent_change=(restart < 10)
            )
            
            if new_score < best_count:
                best_count = new_score
                lst = _make_list_from_exps_coeffs(new_exps, new_coeffs)
                if lst is not None:
                    best_coeffs = lst
    
    # --- Phase 3: Fine-tune best solution ---
    if best_coeffs is not None and not expired():
        pos = [i for i, v in enumerate(best_coeffs) if v != 0]
        vals = [v for v in best_coeffs if v != 0]
        
        if len(pos) == 12:
            for run in range(3):
                if expired():
                    break
                new_exps, new_coeffs, new_score = _hill_climb(
                    pos, vals, end_time, rng, allow_exponent_change=True
                )
                if new_score < best_count:
                    best_count = new_score
                    lst = _make_list_from_exps_coeffs(new_exps, new_coeffs)
                    if lst is not None:
                        best_coeffs = lst
    
    # --- Phase 4: Exhaustive sign search on best positions ---
    if best_coeffs is not None and not expired():
        pos = [i for i, v in enumerate(best_coeffs) if v != 0]
        vals = [v for v in best_coeffs if v != 0]
        
        if len(pos) == 12:
            mags = [abs(v) for v in vals]
            for mask in range(4096):
                if expired():
                    break
                signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
                new_vals = [mags[i] * signs[i] for i in range(12)]
                c = _square_nonzero_count(pos, new_vals)
                if c < best_count:
                    best_count = c
                    lst = _make_list_from_exps_coeffs(pos, new_vals)
                    if lst is not None:
                        best_coeffs = lst
    
    # Fallback
    if best_coeffs is None:
        best_coeffs = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        k = len(best_coeffs) - 1
        while k >= 0 and best_coeffs[k] == 0:
            k -= 1
        best_coeffs = best_coeffs[:k+1]
    
    return best_coeffs
