import time
import random
from itertools import product

def _count_nonzero(arr):
    return sum(1 for x in arr if x != 0)

def _square_nonzero_count_sparse(exponents, coeffs):
    """Compute nonzero count in square of sparse polynomial."""
    if not exponents:
        return 0
    max_deg = exponents[-1]
    square = [0] * (2 * max_deg + 1)
    n = len(exponents)
    for i in range(n):
        ei = exponents[i]
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            ej = exponents[j]
            cj = coeffs[j]
            if cj == 0:
                continue
            square[ei + ej] += ci * cj
    return _count_nonzero(square)

def _build_full_coeff_list(exponents, coeffs):
    """Build full coefficient list in ascending degree."""
    if not exponents:
        return [1]
    max_deg = max(exponents)
    full = [0] * (max_deg + 1)
    for e, c in zip(exponents, coeffs):
        full[e] = c
    return full

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    best_count = float('inf')
    best_coeffs = None

    def eval_square_sparse(exponents, coeffs):
        """Compute nonzero count in square of sparse polynomial."""
        return _square_nonzero_count_sparse(exponents, coeffs)

    def try_update(count, exponents, coeffs):
        nonlocal best_count, best_coeffs
        if count < best_count:
            full = _build_full_coeff_list(exponents, coeffs)
            if len(full) > 41:
                return False
            if full[-1] == 0:
                # Find last nonzero
                for i in range(len(full) - 1, -1, -1):
                    if full[i] != 0:
                        full = full[:i + 1]
                        break
                if not full or full[-1] == 0:
                    return False
            nz = sum(1 for x in full if x != 0)
            if nz != 12:
                return False
            if any(abs(x) > 64 for x in full):
                return False
            best_count = count
            best_coeffs = full
            return True
        return False

    def to_sparse_list(full_coeffs):
        """Convert full coeff list to (exponents, coeffs) sparse representation."""
        exps = []
        coefs = []
        for i, c in enumerate(full_coeffs):
            if c != 0:
                exps.append(i)
                coefs.append(c)
        return exps, coefs

    # =====================================================================
    # Phase 1: Brute force signs for 12 consecutive terms (0..11)
    # This is fast: 2^12 = 4096 evaluations
    # =====================================================================
    for bits in range(1 << 12):
        if time.time() > deadline - 40:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        count = _square_nonzero_count_sparse(list(range(12)), coeffs)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 2: 2-term × 6-term structured search
    # P = (a + b*x^d) * R where R has 6 terms
    # This creates natural cancellation patterns
    # =====================================================================
    random.seed(42)

    # Phase 2a: Systematic search for 2-term * 6-term structures
    # Q = a + b*x^d, R has 6 terms at positions in [0, d-1]
    # For no overlap between R and x^d*R, need d >= 6
    for d in range(6, 25):
        if time.time() > deadline - 30:
            break
        for qa in [-2, -1, 1, 2]:
            for qb in [-2, -1, 1, 2]:
                if time.time() > deadline - 30:
                    break
                # R must have 6 terms; when multiplied by (qa + qb*x^d),
                # we get qa*R + qb*x^d*R
                # For exactly 12 nonzero terms, we need no overlap
                # i.e., all exponents of R and all exponents of R+d must be distinct
                
                # Try R with consecutive exponents in [0, d-1]
                for r_start in range(0, max(1, d - 5)):
                    if time.time() > deadline - 30:
                        break
                    r_exps = list(range(r_start, r_start + 6))
                    if r_exps[-1] >= d:
                        continue
                    
                    # Search coefficients for R: each in {-2, -1, 1, 2}
                    # That's 4^6 = 4096 combinations
                    for rc in product([-2, -1, 1, 2], repeat=6):
                        if time.time() > deadline - 30:
                            break
                        # Build P = (qa + qb*x^d) * R
                        # P exponents: r_exps and [e+d for e in r_exps]
                        p_exps = r_exps + [e + d for e in r_exps]
                        p_coeffs = [qa * c for c in rc] + [qb * c for c in rc]
                        
                        # Check exactly 12 nonzero terms
                        if _count_nonzero(p_coeffs) != 12:
                            continue
                        
                        # Check all coefficients in [-64, 64]
                        if any(abs(c) > 64 for c in p_coeffs):
                            continue
                        
                        # Check max degree <= 40
                        if p_exps[-1] > 40:
                            continue
                        
                        count = _square_nonzero_count_sparse(p_exps, p_coeffs)
                        if count < best_count:
                            try_update(count, p_exps, p_coeffs)
                            if best_count == 12:
                                return best_coeffs

    # Phase 2b: Try with random R coefficients for broader search
    for d in range(6, 30):
        if time.time() > deadline - 25:
            break
        for qa in [-2, -1, 1, 2]:
            for qb in [-2, -1, 1, 2]:
                if time.time() > deadline - 25:
                    break
                for _ in range(100):
                    if time.time() > deadline - 25:
                        break
                    # Random 6 exponents from [0, d-1]
                    if d < 6:
                        continue
                    r_exps = sorted(random.sample(range(d), 6))
                    rc = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
                    
                    p_exps = r_exps + [e + d for e in r_exps]
                    p_coeffs = [qa * c for c in rc] + [qb * c for c in rc]
                    
                    if _count_nonzero(p_coeffs) != 12:
                        continue
                    if any(abs(c) > 64 for c in p_coeffs):
                        continue
                    if p_exps[-1] > 40:
                        continue
                    
                    count = _square_nonzero_count_sparse(p_exps, p_coeffs)
                    if count < best_count:
                        try_update(count, p_exps, p_coeffs)
                        if best_count == 12:
                            return best_coeffs

    # =====================================================================
    # Phase 3: Structural factor search (2x2x3 from current program)
    # =====================================================================
    def convolve_sparse(poly1, poly2):
        res = {}
        for e1, c1 in poly1:
            for e2, c2 in poly2:
                e = e1 + e2
                res[e] = res.get(e, 0) + c1 * c2
        return [(e, c) for e, c in res.items() if c != 0]

    candidate_factors_pool = []
    for k in range(1, 11):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                if a == 0 or b == 0:
                    continue
                candidate_factors_pool.append([(0, a), (k, b)])
    for k in range(1, 8):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                for c in [-2, -1, 1, 2]:
                    if a == 0 or b == 0 or c == 0:
                        continue
                    candidate_factors_pool.append([(0, a), (k, b), (2 * k, c)])
                    if k + 1 <= 15:
                        candidate_factors_pool.append([(0, a), (k, b), (k + 1, c)])

    pool_2 = [f for f in candidate_factors_pool if len(f) == 2]
    pool_3 = [f for f in candidate_factors_pool if len(f) == 3]

    pair_products = []
    for i in range(len(pool_2)):
        for j in range(i, len(pool_2)):
            prod = convolve_sparse(pool_2[i], pool_2[j])
            if len(prod) <= 4:
                pair_products.append(prod)

    seen_polys = set()
    for pp in pair_products:
        if time.time() > deadline - 20:
            break
        for c in pool_3:
            max_e_pp = max(e for e, _ in pp)
            max_e_c = max(e for e, _ in c)
            if max_e_pp + max_e_c > 40:
                continue
            p_poly = convolve_sparse(pp, c)
            if len(p_poly) != 12:
                continue
            p_sorted = sorted(p_poly)
            key = tuple(p_sorted)
            if key in seen_polys:
                continue
            seen_polys.add(key)
            exps = [e for e, _ in p_sorted]
            coefs = [c for _, c in p_sorted]
            count = _square_nonzero_count_sparse(exps, coefs)
            try_update(count, exps, coefs)
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 4: Hill climb local search
    # =====================================================================
    def hill_climb(exps, coefs, time_budget=2.0):
        nonlocal best_count, best_coeffs
        t0 = time.time()
        exps = list(exps)
        coefs = list(coefs)
        count = _square_nonzero_count_sparse(exps, coefs)
        try_update(count, exps, coefs)
        no_improve_count = 0
        vals = [-4, -3, -2, -1, 1, 2, 3, 4, -8, 8, -16, 16]

        while time.time() - t0 < time_budget:
            improved = False
            for _ in range(5):
                idx = random.randrange(12)
                e = exps[idx]
                c = coefs[idx]
                if random.random() < 0.7:
                    new_c = random.choice(vals)
                    if new_c != c and abs(new_c) <= 64:
                        coefs2 = list(coefs)
                        coefs2[idx] = new_c
                        if _count_nonzero(coefs2) != 12:
                            continue
                        c2 = _square_nonzero_count_sparse(exps, coefs2)
                        if c2 < count:
                            coefs = coefs2
                            count = c2
                            try_update(count, exps, coefs)
                            improved = True
                            no_improve_count = 0
                            break
                else:
                    shift = random.choice([-2, -1, 1, 2])
                    new_e = e + shift
                    if 0 <= new_e <= 40:
                        exps2 = list(exps)
                        exps2.remove(e)
                        if new_e not in exps2:
                            exps2.append(new_e)
                            exps2.sort()
                            c2 = _square_nonzero_count_sparse(exps2, coefs)
                            if c2 < count:
                                exps = exps2
                                count = c2
                                try_update(count, exps, coefs)
                                improved = True
                                no_improve_count = 0
                                break
            if not improved:
                no_improve_count += 1
                if no_improve_count > 30:
                    for _ in range(2):
                        idx = random.randrange(12)
                        new_c = random.choice([-8, -4, -2, -1, 1, 2, 4, 8])
                        if abs(new_c) <= 64:
                            coefs[idx] = new_c
                    no_improve_count = 0
                    count = _square_nonzero_count_sparse(exps, coefs)
                    try_update(count, exps, coefs)
            if best_count == 12:
                break
        return exps, coefs, count

    # Start hill climb from best found so far
    if best_coeffs is not None:
        exps, coefs = to_sparse_list(best_coeffs)
        if len(exps) == 12:
            hill_climb(exps, coefs, time_budget=5.0)
            if best_count == 12:
                return best_coeffs

    # Random starts
    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        if random.random() < 0.5:
            exps = list(range(12))
        else:
            exps = sorted(random.sample(range(40), 12))
        coefs = [random.choice(vals) for _ in range(12)]
        for i in range(12):
            if coefs[i] == 0:
                coefs[i] = 1
        hill_climb(exps, coefs, time_budget=1.0)
        if best_count == 12:
            return best_coeffs
        if attempts % 10 == 0:
            base_exp = random.randint(1, 3)
            exps = [i * base_exp for i in range(12)]
            if exps[-1] > 40:
                continue
            coefs = [random.choice(vals) for _ in range(12)]
            hill_climb(exps, coefs, time_budget=0.5)
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 5: Final polish with tighter moves
    # =====================================================================
    if best_coeffs is not None:
        exps, coefs = to_sparse_list(best_coeffs)
        if len(exps) == 12:
            def hill_climb_polish(exps, coefs, time_budget=5.0):
                nonlocal best_count, best_coeffs
                t0 = time.time()
                exps = list(exps)
                coefs = list(coefs)
                count = _square_nonzero_count_sparse(exps, coefs)
                try_update(count, exps, coefs)
                vals = [-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64]

                while time.time() - t0 < time_budget:
                    improved = False
                    for _ in range(10):
                        idx = random.randrange(12)
                        c = coefs[idx]
                        new_c = random.choice(vals)
                        if new_c != c:
                            coefs2 = list(coefs)
                            coefs2[idx] = new_c
                            if _count_nonzero(coefs2) != 12:
                                continue
                            c2 = _square_nonzero_count_sparse(exps, coefs2)
                            if c2 < count:
                                coefs = coefs2
                                count = c2
                                try_update(count, exps, coefs)
                                improved = True
                                break
                    if not improved:
                        idx = random.randrange(12)
                        e = exps[idx]
                        shift = random.choice([-1, 1])
                        new_e = e + shift
                        if 0 <= new_e <= 40:
                            exps2 = list(exps)
                            exps2.remove(e)
                            if new_e not in exps2:
                                exps2.append(new_e)
                                exps2.sort()
                                c2 = _square_nonzero_count_sparse(exps2, coefs)
                                if c2 < count:
                                    exps = exps2
                                    count = c2
                                    try_update(count, exps, coefs)
                                    improved = True
                    if best_count == 12:
                        break
                return exps, coefs, count

            hill_climb_polish(exps, coefs, time_budget=5.0)
            if best_count == 12:
                return best_coeffs

    # Fallback
    if best_coeffs is None:
        best_coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        if len(best_coeffs) > 41:
            best_coeffs = best_coeffs[:41]
        if best_coeffs[-1] == 0:
            for i in range(len(best_coeffs) - 1, -1, -1):
                if best_coeffs[i] != 0:
                    best_coeffs = best_coeffs[:i + 1]
                    break
            if not best_coeffs:
                best_coeffs = [1]
        nz = [i for i, c in enumerate(best_coeffs) if c != 0]
        if len(nz) != 12:
            best_coeffs = [0] * 41
            for i in range(12):
                best_coeffs[i] = 1

    return best_coeffs
