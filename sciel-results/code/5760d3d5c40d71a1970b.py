"""Rewritten program: focus on finding sparse squares of 12-term polynomials."""
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
        """
        Compute the number of nonzero terms in the exact square of a polynomial
        given by exponents (sorted, length 12) and coeffs (length 12).
        """
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

    # Phase 1: Structured AP exponents with many coefficient patterns
    # Key insight: arithmetic progressions of exponents create regular overlap patterns
    for d in range(1, 7):
        if time.time() > deadline - 20:
            break
        max_a = min(41 - 11 * d, 12)
        for a in range(0, max_a + 1):
            if time.time() > deadline - 20:
                break
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] >= 41:
                continue

            # Generate many coefficient patterns
            patterns = []
            # Alternating signs
            patterns.append([1 if i % 2 == 0 else -1 for i in range(12)])
            patterns.append([-1 if i % 2 == 0 else 1 for i in range(12)])
            # Blocks of 2
            patterns.append([1 if (i // 2) % 2 == 0 else -1 for i in range(12)])
            patterns.append([-1 if (i // 2) % 2 == 0 else 1 for i in range(12)])
            # Blocks of 3
            patterns.append([1 if (i // 3) % 2 == 0 else -1 for i in range(12)])
            patterns.append([-1 if (i // 3) % 2 == 0 else 1 for i in range(12)])
            # Blocks of 4
            patterns.append([1 if (i // 4) % 2 == 0 else -1 for i in range(12)])
            patterns.append([-1 if (i // 4) % 2 == 0 else 1 for i in range(12)])
            # Blocks of 6
            patterns.append([1 if (i // 6) % 2 == 0 else -1 for i in range(12)])
            patterns.append([-1 if (i // 6) % 2 == 0 else 1 for i in range(12)])
            # All same
            patterns.append([1] * 12)
            patterns.append([-1] * 12)
            # Mixed magnitudes
            patterns.append([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
            patterns.append([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
            patterns.append([1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1])
            patterns.append([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])
            patterns.append([2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2])
            patterns.append([1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2])
            patterns.append([1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1])
            patterns.append([3, -1, -1, 1, 1, -3, 1, 1, -1, -1, 1, -3])

            for cp in patterns:
                if time.time() > deadline - 20:
                    break
                try_exponents_coeffs(exponents, cp)

    # Phase 2: Randomized search with diverse strategies
    random.seed(12345)

    def random_search(num_trials=60000):
        nonlocal best_poly, best_nonzero_in_square
        for trial in range(num_trials):
            if time.time() > deadline - 25:
                break

            r = random.random()
            if r < 0.25:
                # AP-like exponents
                d = random.randint(1, 5)
                max_a = max(0, 40 - 11 * d)
                a = random.randint(0, max_a)
                exponents = [a + i * d for i in range(12)]
                if exponents[-1] >= 41:
                    continue
            elif r < 0.45:
                # Two-cluster: 6 terms in each of two AP groups
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
                # Symmetric around center
                center = random.randint(6, 34)
                offsets = sorted(random.sample(range(1, 20), 6))
                exponents = [center - o for o in offsets] + [center + o for o in offsets]
                if any(e < 0 or e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            elif r < 0.75:
                # Clustered in a small window
                window = random.randint(12, 25)
                base = random.randint(0, max(0, 40 - window))
                exponents = sorted(random.sample(range(base, min(base + window, 41)), 12))
            else:
                # Fully random
                exponents = sorted(random.sample(range(41), 12))

            # Choose coefficients
            cstyle = random.random()
            if cstyle < 0.35:
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
            elif cstyle < 0.55:
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            elif cstyle < 0.70:
                # Patterned
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
                # Small integer coefficients
                coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]

            if all(abs(c) <= 64 and c != 0 for c in coeffs):
                try_exponents_coeffs(exponents, coeffs)

    random_search(80000)

    # Phase 3: Local search / hill climbing
    if best_poly is not None and time.time() < deadline - 10:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]

        for restart in range(5):
            if time.time() > deadline - 5:
                break

            cur_exp = list(best_exp)
            cur_coef = list(best_coef)
            cur_nz = square_nonzero_count(cur_exp, cur_coef)

            for it in range(3000):
                if time.time() > deadline - 5:
                    break

                exp_copy = list(cur_exp)
                coef_copy = list(cur_coef)

                r = random.random()
                if r < 0.45:
                    # Change a coefficient
                    idx = random.randint(0, 11)
                    new_c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 12])
                    if new_c != coef_copy[idx]:
                        coef_copy[idx] = new_c
                elif r < 0.75:
                    # Move an exponent
                    idx = random.randint(0, 11)
                    shift = random.choice([-3, -2, -1, 1, 2, 3, 4, 5])
                    new_e = exp_copy[idx] + shift
                    if 0 <= new_e < 41 and new_e not in exp_copy:
                        exp_copy[idx] = new_e
                        exp_copy.sort()
                    else:
                        continue
                elif r < 0.85:
                    # Swap two exponents
                    i, j = random.sample(range(12), 2)
                    exp_copy[i], exp_copy[j] = exp_copy[j], exp_copy[i]
                    exp_copy.sort()
                else:
                    # Change two coefficients
                    i, j = random.sample(range(12), 2)
                    coef_copy[i] = random.choice([-3, -2, -1, 1, 2, 3, 4])
                    coef_copy[j] = random.choice([-3, -2, -1, 1, 2, 3, 4])

                if len(set(exp_copy)) != 12:
                    continue
                if any(e < 0 or e >= 41 for e in exp_copy):
                    continue
                if any(c == 0 or abs(c) > 64 for c in coef_copy):
                    continue

                nz = square_nonzero_count(exp_copy, coef_copy)
                if nz < cur_nz:
                    cur_nz = nz
                    cur_exp = list(exp_copy)
                    cur_coef = list(coef_copy)

                    if nz < best_nonzero_in_square:
                        best_nonzero_in_square = nz
                        max_deg = max(cur_exp)
                        poly_new = [0] * (max_deg + 1)
                        for e, c in zip(cur_exp, cur_coef):
                            poly_new[e] = c
                        best_poly = poly_new
                        best_exp = list(cur_exp)
                        best_coef = list(cur_coef)

    # Phase 4: Targeted search for very sparse squares
    # Theoretical minimum: for 12 terms, the square has at most 12*13/2 = 78 terms
    # With cancellation, we can get much lower. Known results suggest ~16-20 is achievable.
    # Try to find patterns that maximize cancellation.

    if time.time() < deadline - 8:
        # Try specific known-good structures
        # For a polynomial with terms at positions forming a group structure,
        # the square can have fewer terms due to cancellation.

        # Try: exponents as union of two APs with specific relationship
        for d1 in range(1, 4):
            if time.time() > deadline - 5:
                break
            for d2 in range(1, 4):
                if time.time() > deadline - 5:
                    break
                for a1 in range(0, 10):
                    if time.time() > deadline - 5:
                        break
                    for gap in range(3, 20):
                        if time.time() > deadline - 5:
                            break
                        a2 = a1 + gap
                        exp1 = [a1 + i * d1 for i in range(6)]
                        exp2 = [a2 + i * d2 for i in range(6)]
                        exponents = sorted(exp1 + exp2)
                        if len(set(exponents)) != 12 or any(e >= 41 for e in exponents):
                            continue

                        # Try sign patterns that promote cancellation at overlap degrees
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                if time.time() > deadline - 5:
                                    break
                                # First 6 terms have sign s1, last 6 have sign s2
                                coeffs = [s1] * 6 + [s2] * 6
                                try_exponents_coeffs(exponents, coeffs)

                                # Alternating within groups
                                coeffs2 = [s1 if i % 2 == 0 else -s1 for i in range(6)] + \
                                          [s2 if i % 2 == 0 else -s2 for i in range(6)]
                                try_exponents_coeffs(exponents, coeffs2)

    # Final fallback
    if best_poly is None:
        best_poly = [1] * 12

    # Final validation
    if not validate(best_poly):
        best_poly = [1] * 12

    return best_poly

# EVOLVE-BLOCK-END
