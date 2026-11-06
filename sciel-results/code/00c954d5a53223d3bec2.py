import time
import random
import numpy as np
from itertools import combinations

T_START = time.time()
DEADLINE = 80.0  # Safety margin before 90s kill

def _square_nonzero_fast(exps, coeffs):
    """Compute nonzero count of square using vectorized numpy."""
    n = len(exps)
    if n == 0:
        return 0
    e = np.array(exps, dtype=np.int64)
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

def _valid_candidate(coeffs):
    """Check all constraints on the candidate list."""
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

def _square_nonzero_count(coeffs):
    """Return number of nonzero coefficients in the exact square of coeff list."""
    if not _valid_candidate(coeffs):
        return None
    exps = [i for i, v in enumerate(coeffs) if v != 0]
    vals = [v for v in coeffs if v != 0]
    return _square_nonzero_fast(exps, vals)

def _try_polynomial(coeffs):
    """If valid, return (count, coeffs); else (None, None)."""
    if not _valid_candidate(coeffs):
        return None, None
    exps = [i for i, v in enumerate(coeffs) if v != 0]
    vals = [v for v in coeffs if v != 0]
    return _square_nonzero_fast(exps, vals), coeffs

def construct():
    best_count = 100
    best_coeffs = None
    end_time = T_START + DEADLINE
    rng = random.Random(42)
    
    def expired():
        return time.time() > end_time
    
    def build_poly(positions, values):
        coeffs = [0] * 41
        for p, v in zip(positions, values):
            if 0 <= p < 41:
                coeffs[p] = v
            else:
                return None
        k = 40
        while k >= 0 and coeffs[k] == 0:
            k -= 1
        if k < 0:
            return None
        return coeffs[:k+1]
    
    def update_best(coeffs):
        nonlocal best_count, best_coeffs
        if coeffs is None:
            return
        c, _ = _try_polynomial(coeffs)
        if c is not None and c < best_count:
            best_count = c
            best_coeffs = list(coeffs)
    
    def make_exps_coeffs(exps, coeffs):
        """Convert parallel lists to coefficient list in ascending degree."""
        if not _valid_candidate_exps(exps, coeffs):
            return None
        max_e = max(exps) if exps else 0
        lst = [0] * (max_e + 1)
        for e, c in zip(exps, coeffs):
            if e <= 40:
                lst[e] = c
        while len(lst) > 1 and lst[-1] == 0:
            lst.pop()
        return lst
    
    def _valid_candidate_exps(exps, coeffs):
        """Check validity of parallel exponent/coefficient lists."""
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
    
    # --- Phase 1: Factor-based construction with sign search ---
    # Key insight: if P = A * B where A and B are sparse, then P^2 = A^2 * B^2
    # The square structure can create systematic cancellations
    
    # Try various factor structures
    factor_structures = []
    
    # Geometric progression factors: A = sum a_i * r^i
    for r in [2, 3, 4, 5]:
        for a_len in range(2, 7):
            for b_len in range(2, 7):
                if a_len + b_len - 1 <= 12:
                    # A has a_len terms, B has b_len terms
                    # P = A*B has at most a_len*b_len terms
                    if a_len * b_len <= 12:
                        factor_structures.append((a_len, b_len, r))
    
    # Arithmetic progression factors
    for a_gap in [1, 2, 3]:
        for b_gap in [1, 2, 3]:
            for a_len in range(2, 7):
                for b_len in range(2, 7):
                    if a_len * b_len <= 12:
                        factor_structures.append((a_len, b_len, 'ap', a_gap, b_gap))
    
    # Evaluate factor-based candidates
    for struct in factor_structures:
        if expired():
            break
        
        if len(struct) == 3:
            a_len, b_len, r = struct
        else:
            a_len, b_len, _, a_gap, b_gap = struct
        
        # Generate A and B exponent sets
        if len(struct) == 3:
            a_exps = [i * r for i in range(a_len)]
            b_exps = [i * r for i in range(b_len)]
        else:
            a_exps = [i * a_gap for i in range(a_len)]
            b_exps = [i * b_gap for i in range(b_len)]
        
        # P exponents are all sums a_i + b_j
        p_exps = sorted(set(a + b for a in a_exps for b in b_exps))
        
        if len(p_exps) > 12:
            continue
        
        # If we have fewer than 12 exponents, pad with zeros or skip
        if len(p_exps) < 12:
            # Try to add more exponents
            continue
        
        # For each valid exponent set, try sign patterns
        # The coefficients of P are determined by the factor coefficients
        # For simplicity, try all 1s first, then sign variations
        
        base_coeffs = [1] * len(p_exps)
        update_best(make_exps_coeffs(p_exps, base_coeffs))
        
        # Try sign patterns
        if len(p_exps) <= 12:
            for mask in range(4096):
                if expired():
                    break
                signs = [1 if (mask >> i) & 1 else -1 for i in range(len(p_exps))]
                new_coeffs = [signs[i] for i in range(len(p_exps))]
                update_best(make_exps_coeffs(p_exps, new_coeffs))
    
    # --- Phase 2: Structured position sets with coefficient search ---
    
    position_candidates = []
    
    # Arithmetic progressions
    for gap in range(1, 8):
        for start in range(0, 41 - 11*gap + 1):
            pos = [start + i*gap for i in range(12)]
            position_candidates.append(pos)
    
    # Two-block structures
    for a in range(0, 20):
        for b in range(a+6, 41):
            pos = list(range(a, a+6)) + list(range(b, b+6))
            if max(pos) < 41:
                position_candidates.append(pos)
    
    # Symmetric positions
    for center in range(10, 31):
        for spread in range(1, 20):
            pos = []
            for k in range(1, 20):
                if center - k >= 0:
                    pos.append(center - k)
                if center + k < 41:
                    pos.append(center + k)
                if len(pos) >= 12:
                    break
            if len(pos) == 12:
                pos.sort()
                position_candidates.append(pos)
    
    # Random positions biased toward clusters
    for _ in range(200):
        base = rng.randint(0, 30)
        pos = set()
        for _ in range(12):
            p = base + rng.randint(-5, 5)
            if 0 <= p < 41:
                pos.add(p)
            else:
                pos.add(rng.randint(0, 40))
        if len(pos) == 12:
            position_candidates.append(sorted(pos))
    
    # Deduplicate
    seen_pos = set()
    unique_pos = []
    for ps in position_candidates:
        key = tuple(ps)
        if key not in seen_pos:
            seen_pos.add(key)
            unique_pos.append(ps)
    position_candidates = unique_pos
    
    # Evaluate position sets with sign patterns
    sign_patterns = []
    sign_patterns.append([1]*12)
    sign_patterns.append([1 if i%2==0 else -1 for i in range(12)])
    sign_patterns.append([1]*6 + [-1]*6)
    for mask in range(0, 4096, 32):
        sign_patterns.append([1 if (mask >> i) & 1 else -1 for i in range(12)])
    
    for ps in position_candidates:
        if expired():
            break
        for signs in sign_patterns:
            if expired():
                break
            update_best(build_poly(ps, signs))
    
    # --- Phase 3: Hill climbing with fixed exponents ---
    
    def _hill_climb_ap_fixed(exps, coeffs, deadline, rng_local):
        """Hill climbing with fixed exponents, optimizing coefficients."""
        current_exps = exps[:]
        current_coeffs = coeffs[:]
        current_score = _square_nonzero_fast(current_exps, current_coeffs)
        best_exps = current_exps[:]
        best_coeffs = current_coeffs[:]
        best_score = current_score
        
        no_improve = 0
        max_iters = 5000
        
        while time.time() < deadline and no_improve < 1000:
            move_type = rng_local.random()
            new_exps = current_exps[:]
            new_coeffs = current_coeffs[:]
            
            if move_type < 0.5:
                idx = rng_local.randrange(12)
                new_c = rng_local.choice([-4, -3, -2, -1, 1, 2, 3, 4, -8, 8])
                if new_c == 0:
                    new_c = 1
                new_coeffs[idx] = new_c
            elif move_type < 0.8:
                idx = rng_local.randrange(12)
                new_coeffs[idx] = -new_coeffs[idx]
            else:
                i, j = rng_local.sample(range(12), 2)
                new_coeffs[i] = rng_local.choice([-2, -1, 1, 2])
                new_coeffs[j] = rng_local.choice([-2, -1, 1, 2])
            
            if not _valid_candidate_exps(new_exps, new_coeffs):
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
                if rng_local.random() < 0.05:
                    current_exps = new_exps
                    current_coeffs = new_coeffs
                    current_score = new_score
                no_improve += 1
        
        return best_exps, best_coeffs, best_score
    
    # Run hill climbing on best candidates
    if best_coeffs and not expired():
        pos = [i for i, v in enumerate(best_coeffs) if v != 0]
        vals = [v for v in best_coeffs if v != 0]
        
        if len(pos) == 12:
            # Try hill climbing on several variants
            for variant in range(5):
                if expired():
                    break
                
                # Create variant with different coefficient magnitudes
                variant_coeffs = []
                for v in vals:
                    mag = abs(v)
                    if mag == 0:
                        mag = 1
                    variant_coeffs.append(mag * (1 if rng.random() < 0.5 else -1))
                
                new_exps, new_coeffs, new_score = _hill_climb_ap_fixed(
                    pos, variant_coeffs, end_time, rng
                )
                
                if new_score < best_count:
                    best_count = new_score
                    best_coeffs = make_exps_coeffs(new_exps, new_coeffs)
    
    # --- Phase 4: General hill climbing allowing exponent changes ---
    
    def _hill_climb_general(exps, coeffs, deadline, rng_local):
        """General hill climbing allowing exponent changes."""
        current_exps = exps[:]
        current_coeffs = coeffs[:]
        current_score = _square_nonzero_fast(current_exps, current_coeffs)
        best_exps = current_exps[:]
        best_coeffs = current_coeffs[:]
        best_score = current_score
        
        no_improve = 0
        
        while time.time() < deadline and no_improve < 2000:
            move_type = rng_local.random()
            new_exps = current_exps[:]
            new_coeffs = current_coeffs[:]
            
            if move_type < 0.4:
                idx = rng_local.randrange(12)
                new_c = rng_local.choice([-8, -4, -3, -2, -1, 1, 2, 3, 4, 8])
                if new_c == 0:
                    new_c = 1
                new_coeffs[idx] = new_c
            elif move_type < 0.6:
                idx = rng_local.randrange(12)
                old_e = new_exps[idx]
                shift = rng_local.choice([-2, -1, 1, 2])
                new_e = old_e + shift
                if new_e < 0 or new_e > 40 or new_e in new_exps:
                    continue
                new_exps[idx] = new_e
                new_exps.sort()
            elif move_type < 0.7:
                idx = rng_local.randrange(12)
                new_coeffs[idx] = -new_coeffs[idx]
            else:
                i, j = rng_local.sample(range(12), 2)
                new_coeffs[i] = rng_local.choice([-2, -1, 1, 2])
                new_coeffs[j] = rng_local.choice([-2, -1, 1, 2])
            
            if not _valid_candidate_exps(new_exps, new_coeffs):
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
                if rng_local.random() < 0.02:
                    current_exps = new_exps
                    current_coeffs = new_coeffs
                    current_score = new_score
                no_improve += 1
        
        return best_exps, best_coeffs, best_score
    
    # Run general hill climbing
    if best_coeffs and not expired():
        pos = [i for i, v in enumerate(best_coeffs) if v != 0]
        vals = [v for v in best_coeffs if v != 0]
        
        if len(pos) == 12:
            new_exps, new_coeffs, new_score = _hill_climb_general(
                pos, vals, end_time, rng
            )
            
            if new_score < best_count:
                best_count = new_score
                best_coeffs = make_exps_coeffs(new_exps, new_coeffs)
    
    # --- Phase 5: Exhaustive sign search on best positions ---
    if best_coeffs and not expired():
        pos = [i for i, v in enumerate(best_coeffs) if v != 0]
        vals = [v for v in best_coeffs if v != 0]
        if len(pos) == 12:
            mags = [abs(v) for v in vals]
            for mask in range(4096):
                if expired():
                    break
                signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
                new_vals = [mags[i] * signs[i] for i in range(12)]
                update_best(build_poly(pos, new_vals))
    
    if best_coeffs is None:
        best_coeffs = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        k = len(best_coeffs) - 1
        while k >= 0 and best_coeffs[k] == 0:
            k -= 1
        best_coeffs = best_coeffs[:k+1]
    
    return best_coeffs
