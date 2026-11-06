import time
import itertools

T_START = time.time()
DEADLINE = 85.0  # Safety margin from 90s

def _square_nonzero_count(coeffs):
    """Return number of nonzero coefficients in the exact square of coeff list."""
    n = len(coeffs)
    if n == 0:
        return 0
    sq = [0] * (2 * n - 1)
    # Optimization: only iterate over non-zero terms
    non_zero_indices = [i for i, c in enumerate(coeffs) if c != 0]
    for i_idx, i in enumerate(non_zero_indices):
        a = coeffs[i]
        for j_idx, j in enumerate(non_zero_indices):
            if j_idx < i_idx:
                continue
            b = coeffs[j]
            if i == j:
                sq[i + j] += a * a
            else:
                sq[i + j] += 2 * a * b
    count = 0
    for v in sq:
        if v != 0:
            count += 1
    return count

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

def construct():
    best_count = 25
    best_coeffs = None
    end_time = T_START + DEADLINE

    def expired():
        return time.time() > end_time

    # Helper to update best
    def update_best(coeffs):
        nonlocal best_count, best_coeffs
        if not _valid_candidate(coeffs):
            return
        cnt = _square_nonzero_count(coeffs)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = list(coeffs)

    # Initial seed: 12 consecutive ones
    seed_coeffs = [1] * 12
    update_best(seed_coeffs)

    # Generate position sets
    position_sets = []
    
    # 1. Consecutive blocks
    for start in range(0, 30):
        position_sets.append(list(range(start, start + 12)))
        
    # 2. Symmetric sets around a center
    # Center can be integer or half-integer
    for center in range(5, 35):
        for half in range(1, 10):
            # 6 terms on left, 6 on right
            pos = [center - half - i for i in range(6)] + [center + half + i for i in range(6)]
            pos = sorted(set(pos))
            if len(pos) == 12 and min(pos) >= 0 and max(pos) <= 40:
                position_sets.append(pos)
                
    # 3. Two blocks of 6
    for gap in range(0, 15):
        for start in range(0, 25):
            pos1 = list(range(start, start + 6))
            pos2 = list(range(start + 6 + gap, start + 12 + gap))
            if max(pos2) <= 40:
                position_sets.append(pos1 + pos2)
                
    # 4. Powers of 2 variants
    p2 = [1, 2, 4, 8, 16, 32]
    for offset in range(0, 10):
        pos = [offset + x for x in p2]
        if max(pos) <= 40:
            position_sets.append(pos)
            
    # 5. Fibonacci variants
    fib = [1, 2, 3, 5, 8, 13, 21, 34]
    # Pick 6 from fib
    for combo in itertools.combinations(range(8), 6):
        pos = [fib[i] for i in combo]
        if max(pos) <= 40:
            position_sets.append(sorted(pos))
            
    # 6. Random-ish deterministic sets
    import hashlib
    for seedi in range(0, 500):
        h = hashlib.md5(str(seedi).encode()).digest()
        pos = []
        used = set()
        idx = 0
        while len(pos) < 12 and idx < 64:
            byte = h[idx % len(h)]
            p = byte % 41
            if p not in used:
                used.add(p)
                pos.append(p)
            idx += 1
        if len(pos) == 12:
            position_sets.append(sorted(pos))

    # De-duplicate
    seen = set()
    unique_pos = []
    for ps in position_sets:
        key = tuple(ps)
        if key not in seen:
            seen.add(key)
            unique_pos.append(ps)
    position_sets = unique_pos

    # Local search for each position set
    # We optimize signs first (coeffs = +/- 1), then try small magnitudes
    
    def optimize_signs(positions):
        """Local search over signs for fixed positions."""
        # Initialize with all +1
        coeffs = [0] * 41
        for p in positions:
            coeffs[p] = 1
        # Trim
        k = 40
        while k >= 0 and coeffs[k] == 0:
            k -= 1
        cur_coeffs = coeffs[:k+1]
        cur_count = _square_nonzero_count(cur_coeffs)
        
        improved = True
        # Limit iterations to prevent getting stuck in long loops on bad sets
        max_iter = 50
        it = 0
        while improved and not expired() and it < max_iter:
            it += 1
            improved = False
            for idx, p in enumerate(positions):
                if expired():
                    return cur_coeffs, cur_count
                old_val = coeffs[p]
                new_val = -old_val
                coeffs[p] = new_val
                
                k = 40
                while k >= 0 and coeffs[k] == 0:
                    k -= 1
                cand_coeffs = coeffs[:k+1]
                cand_count = _square_nonzero_count(cand_coeffs)
                
                if cand_count < cur_count:
                    cur_count = cand_count
                    cur_coeffs = cand_coeffs
                    improved = True
                else:
                    coeffs[p] = old_val
        return cur_coeffs, cur_count

    for ps in position_sets:
        if expired():
            break
        
        # Sign optimization
        opt_coeffs, opt_count = optimize_signs(ps)
        if opt_count < best_count:
            best_count = opt_count
            best_coeffs = list(opt_coeffs)
            
        # Try small magnitude variations on the best sign pattern found
        if opt_count < best_count + 2: # Only worth trying if close to best
             base_signs = [opt_coeffs[p] if opt_coeffs[p] != 0 else 1 for p in ps]
             # Try changing one coefficient to +/- 2 or +/- 3
             for i in range(12):
                 if expired():
                     break
                 for mag in [2, -2, 3, -3]:
                     # Skip if same as current
                     if base_signs[i] == mag:
                         continue
                     test_coeffs = [0] * 41
                     for j, p in enumerate(ps):
                         val = mag if j == i else base_signs[j]
                         test_coeffs[p] = val
                     k = 40
                     while k >= 0 and test_coeffs[k] == 0:
                         k -= 1
                     tc = test_coeffs[:k+1]
                     tc_count = _square_nonzero_count(tc)
                     if tc_count < best_count:
                         best_count = tc_count
                         best_coeffs = list(tc)
                         # Update base for further single-step changes? 
                         # Keep it simple for now.

    # Final check and return
    if best_coeffs is None:
        best_coeffs = [1] * 12
        
    if _valid_candidate(best_coeffs):
        return best_coeffs
    return [1] * 12
