"""Combined program: sparse squares of twelve-term integer polynomials.
Merges the current program's local-search/hill-climbing machinery with
structured-search ideas, and adds two new mechanisms:
  (1) Incremental square-delta updates for O(24) local-search steps.
  (2) Targeted collision-degree cancellation search (exponent sets whose
      pair-sum multiset has many collisions, so +/-1 coefficients cancel).
"""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    best_poly = None
    best_nonzero_in_square = float('inf')

    def count_nonzero(coeff_list):
        return sum(1 for c in coeff_list if c != 0)

    def square_nonzero_count(exponents, coeffs):
        n = len(exponents)
        sq = defaultdict(int)
        for i in range(n):
            ei = exponents[i]
            ci = coeffs[i]
            for j in range(i, n):
                ej = exponents[j]
                cj = coeffs[j]
                deg = ei + ej
                if i == j:
                    sq[deg] += ci * cj
                else:
                    sq[deg] += 2 * ci * cj
        return sum(1 for v in sq.values() if v != 0)

    def validate(poly_coeff_list):
        if len(poly_coeff_list) > 41:
            return False
        if len(poly_coeff_list) == 0:
            return False
        if poly_coeff_list[-1] == 0:
            return False
        for c in poly_coeff_list:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if count_nonzero(poly_coeff_list) != 12:
            return False
        return True

    def try_exponents_coeffs(exponents, coeffs):
        nonlocal best_poly, best_nonzero_in_square
        if len(exponents) != 12:
            return False
        if len(set(exponents)) != 12:
            return False
        if any(e < 0 or e >= 41 for e in exponents):
            return False
        if any(c == 0 or abs(c) > 64 for c in coeffs):
            return False
        nz = square_nonzero_count(exponents, coeffs)
        if nz < best_nonzero_in_square:
            best_nonzero_in_square = nz
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            best_poly = poly
            return True
        return False

    # ------------------------------------------------------------------
    # Phase 1: Structured AP exponents with many coefficient patterns
    # ------------------------------------------------------------------
    for d in range(1, 8):
        if time.time() > deadline - 30:
            break
        max_a = min(41 - 11 * d, 15)
        for a in range(0, max_a + 1):
            if time.time() > deadline - 30:
                break
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] >= 41:
                continue

            patterns = []
            patterns.append([1 if i % 2 == 0 else -1 for i in range(12)])
            patterns.append([-1 if i % 2 == 0 else 1 for i in range(12)])
            patterns.append([1 if (i // 2) % 2 == 0 else -1 for i in range(12)])
            patterns.append([-1 if (i // 2) % 2 == 0 else 1 for i in range(12)])
            patterns.append([1 if (i // 3) % 2 == 0 else -1 for i in range(12)])
            patterns.append([-1 if (i // 3) % 2 == 0 else 1 for i in range(12)])
            patterns.append([1 if (i // 4) % 2 == 0 else -1 for i in range(12)])
            patterns.append([-1 if (i // 4) % 2 == 0 else 1 for i in range(12)])
            patterns.append([1 if (i // 6) % 2 == 0 else -1 for i in range(12)])
            patterns.append([-1 if (i // 6) % 2 == 0 else 1 for i in range(12)])
            patterns.append([1] * 12)
            patterns.append([-1] * 12)
            patterns.append([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
            patterns.append([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
            patterns.append([1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1])
            patterns.append([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])
            patterns.append([2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2])
            patterns.append([1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2])
            patterns.append([1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1])
            patterns.append([3, -1, -1, 1, 1, -3, 1, 1, -1, -1, 1, -3])
            # Palindromic / near-palindromic (from inspiration)
            patterns.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            patterns.append([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
            patterns.append([1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1])

            for cp in patterns:
                if time.time() > deadline - 30:
                    break
                try_exponents_coeffs(exponents, cp)

    # ------------------------------------------------------------------
    # Phase 2: Targeted collision-degree cancellation search.
    # Exponent sets whose pair-sum multiset has many collisions allow
    # +/-1 coefficients to cancel many square terms. We enumerate
    # structured families (symmetric, two-AP, clustered) and for each
    # do a fast randomized sign search.
    # ------------------------------------------------------------------
    random.seed(98765)

    def collision_count(exponents):
        """Number of pair-sum degrees that have >=2 distinct (i,j) pairs."""
        from collections import Counter
        cnt = Counter()
        n = len(exponents)
        for i in range(n):
            for j in range(i, n):
                cnt[exponents[i] + exponents[j]] += 1
        return sum(1 for v in cnt.values() if v >= 2)

    def sign_search(exponents, num_signs=2000):
        """Randomized +/-1 sign search with a few magnitude tweaks."""
        nonlocal best_poly, best_nonzero_in_square
        n = 12
        best_local_nz = float('inf')
        best_local_coeffs = None
        for _ in range(num_signs):
            if time.time() > deadline - 20:
                return
            coeffs = [random.choice([-1, 1]) for _ in range(n)]
            nz = square_nonzero_count(exponents, coeffs)
            if nz < best_local_nz:
                best_local_nz = nz
                best_local_coeffs = list(coeffs)
            if nz < best_nonzero_in_square:
                best_nonzero_in_square = nz
                max_deg = max(exponents)
                poly = [0] * (max_deg + 1)
                for e, c in zip(exponents, coeffs):
                    poly[e] = c
                best_poly = poly
        # A few magnitude tweaks around the best local
        if best_local_coeffs is not None:
            for _ in range(100):
                if time.time() > deadline - 20:
                    return
                c = list(best_local_coeffs)
                idx = random.randint(0, 11)
                c[idx] = random.choice([-2, -1, 1, 2])
                nz = square_nonzero_count(exponents, c)
                if nz < best_local_nz:
                    best_local_nz = nz
                    best_local_coeffs = list(c)
                if nz < best_nonzero_in_square:
                    best_nonzero_in_square = nz
                    max_deg = max(exponents)
                    poly = [0] * (max_deg + 1)
                    for e, cc in zip(exponents, c):
                        poly[e] = cc
                    best_poly = poly

    # Symmetric exponent families (high collision)
    sym_candidates = []
    for center in range(6, 35):
        for span in range(3, 18):
            # Symmetric: center +/- offsets
            offsets = sorted(random.sample(range(1, span + 1), 6))
            exps = sorted([center - o for o in offsets] + [center + o for o in offsets])
            if len(set(exps)) == 12 and all(0 <= e < 41 for e in exps):
                sym_candidates.append(exps)
    # Two-AP families
    for d1 in range(1, 4):
        for d2 in range(1, 4):
            for a1 in range(0, 12):
                for gap in range(3, 25):
                    a2 = a1 + gap
                    e1 = [a1 + i * d1 for i in range(6)]
                    e2 = [a2 + i * d2 for i in range(6)]
                    exps = sorted(e1 + e2)
                    if len(set(exps)) == 12 and all(0 <= e < 41 for e in exps):
                        sym_candidates.append(exps)

    # Keep top candidates by collision count, then sign-search them
    sym_candidates = list(set(tuple(c) for c in sym_candidates))
    sym_candidates = [list(c) for c in sym_candidates]
    sym_candidates.sort(key=lambda e: -collision_count(e))
    sym_candidates = sym_candidates[:400]

    for exps in sym_candidates:
        if time.time() > deadline - 20:
            break
        sign_search(exps, num_signs=800)

    # ------------------------------------------------------------------
    # Phase 3: Randomized broad search
    # ------------------------------------------------------------------
    def random_search(num_trials=60000):
        nonlocal best_poly, best_nonzero_in_square
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break
            r = random.random()
            if r < 0.25:
                d = random.randint(1, 5)
                max_a = max(0, 40 - 11 * d)
                a = random.randint(0, max_a)
                exponents = [a + i * d for i in range(12)]
                if exponents[-1] >= 41:
                    continue
            elif r < 0.45:
                d1 = random.randint(1, 3)
                d2 = random.randint(1, 3)
                a1 = random.randint(0, 10)
                gap = random.randint(4, 15)
                a2 = a1 + gap
                exponents = [a1 + i * d1 for i in range(6)] + [a2 + i * d2 for i in range(6)]
                if any(e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            elif r < 0.6:
                center = random.randint(6, 34)
                offsets = sorted(random.sample(range(1, 20), 6))
                exponents = [center - o for o in offsets] + [center + o for o in offsets]
                if any(e < 0 or e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            elif r < 0.75:
                window = random.randint(12, 25)
                base = random.randint(0, max(0, 40 - window))
                exponents = sorted(random.sample(range(base, min(base + window, 41)), 12))
            else:
                exponents = sorted(random.sample(range(41), 12))

            cstyle = random.random()
            if cstyle < 0.40:
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
            elif cstyle < 0.55:
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            elif cstyle < 0.70:
                pat = random.choice([
                    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                    [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
                    [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
                    [2, -1, -1, 1, 1, -2, 2, -1, -1, 1, 1, -2],
                    [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
                ])
                coeffs = list(pat)
            elif cstyle < 0.85:
                coeffs = [random.randint(-3, 3) for _ in range(12)]
                for i in range(12):
                    while coeffs[i] == 0:
                        coeffs[i] = random.randint(1, 3) * random.choice([-1, 1])
            else:
                coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]

            if all(abs(c) <= 64 and c != 0 for c in coeffs):
                try_exponents_coeffs(exponents, coeffs)

    random_search(60000)

    # ------------------------------------------------------------------
    # Phase 4: Local search with INCREMENTAL square-delta updates.
    # For a single coefficient change at index k, the square changes only
    # at degrees e_k+e_k (diagonal) and e_k+e_j for j != k (cross terms).
    # That's 1 + 11 = 12 degree updates, each O(1). For an exponent move
    # it's 12 + 12 = 24 updates. This is ~3x faster than the full O(78)
    # recompute, enabling more hill-climbing steps.
    # ------------------------------------------------------------------
    if best_poly is not None and time.time() < deadline - 5:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]

        def build_square(exponents, coeffs):
            sq = defaultdict(int)
            n = len(exponents)
            for i in range(n):
                ei = exponents[i]
                ci = coeffs[i]
                for j in range(i, n):
                    ej = exponents[j]
                    cj = coeffs[j]
                    deg = ei + ej
                    if i == j:
                        sq[deg] += ci * cj
                    else:
                        sq[deg] += 2 * ci * cj
            return sq

        def nz_from_square(sq):
            return sum(1 for v in sq.values() if v != 0)

        for restart in range(8):
            if time.time() > deadline - 3:
                break

            cur_exp = list(best_exp)
            cur_coef = list(best_coef)
            cur_sq = build_square(cur_exp, cur_coef)
            cur_nz = nz_from_square(cur_sq)

            for it in range(8000):
                if time.time() > deadline - 3:
                    break

                r = random.random()
                if r < 0.5:
                    # Change a coefficient (incremental update)
                    idx = random.randint(0, 11)
                    new_c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 12])
                    if new_c == cur_coef[idx]:
                        continue
                    old_c = cur_coef[idx]
                    ek = cur_exp[idx]
                    # Diagonal: degree 2*ek changes from old_c^2 to new_c^2
                    # Cross terms: degree ek+ej changes by 2*(new_c - old_c)*cj
                    new_sq = dict(cur_sq)
                    # Apply diagonal delta
                    delta_diag = new_c * new_c - old_c * old_c
                    new_sq[2 * ek] = new_sq.get(2 * ek, 0) + delta_diag
                    # Apply cross-term deltas
                    for j in range(12):
                        if j == idx:
                            continue
                        deg = ek + cur_exp[j]
                        delta = 2 * (new_c - old_c) * cur_coef[j]
                        new_sq[deg] = new_sq.get(deg, 0) + delta
                    # Clean up zeros
                    new_sq = {k: v for k, v in new_sq.items() if v != 0}
                    new_nz = len(new_sq)
                    if new_nz < cur_nz:
                        cur_nz = new_nz
                        cur_coef = list(cur_coef)
                        cur_coef[idx] = new_c
                        cur_sq = new_sq
                        if new_nz < best_nonzero_in_square:
                            best_nonzero_in_square = new_nz
                            max_deg = max(cur_exp)
                            poly_new = [0] * (max_deg + 1)
                            for e, c in zip(cur_exp, cur_coef):
                                poly_new[e] = c
                            best_poly = poly_new
                            best_exp = list(cur_exp)
                            best_coef = list(cur_coef)
                else:
                    # Move an exponent (full recompute, less frequent)
                    idx = random.randint(0, 11)
                    shift = random.choice([-3, -2, -1, 1, 2, 3, 4, 5])
                    new_e = cur_exp[idx] + shift
                    if 0 <= new_e < 41 and new_e not in cur_exp:
                        new_exp = list(cur_exp)
                        new_exp[idx] = new_e
                        new_exp.sort()
                        new_coef = [cur_coef[cur_exp.index(e)] for e in new_exp]
                        new_nz = square_nonzero_count(new_exp, new_coef)
                        if new_nz < cur_nz:
                            cur_nz = new_nz
                            cur_exp = new_exp
                            cur_coef = new_coef
                            cur_sq = build_square(cur_exp, cur_coef)
                            if new_nz < best_nonzero_in_square:
                                best_nonzero_in_square = new_nz
                                max_deg = max(cur_exp)
                                poly_new = [0] * (max_deg + 1)
                                for e, c in zip(cur_exp, cur_coef):
                                    poly_new[e] = c
                                best_poly = poly_new
                                best_exp = list(cur_exp)
                                best_coef = list(cur_coef)

    # Final fallback
    if best_poly is None:
        best_poly = [1] * 12

    # Final validation
    if not validate(best_poly):
        best_poly = [1] * 12

    return best_poly

# EVOLVE-BLOCK-END
