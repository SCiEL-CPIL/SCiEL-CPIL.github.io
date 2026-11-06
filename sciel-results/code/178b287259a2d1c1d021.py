"""Sparse squares of twelve-term integer polynomials – stronger search."""
import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s kill

    best_poly = None
    best_nz = float('inf')

    def count_nonzero(coeff_list):
        return sum(1 for c in coeff_list if c != 0)

    def square_nonzero(exponents, coeffs):
        """Compute exact square and return nonzero coefficient count."""
        n = len(exponents)
        if n == 0:
            return 0
        square_dict = defaultdict(int)
        for i in range(n):
            ei = exponents[i]
            ci = coeffs[i]
            for j in range(i, n):
                deg = ei + exponents[j]
                if i == j:
                    square_dict[deg] += ci * ci
                else:
                    square_dict[deg] += 2 * ci * coeffs[j]
        return sum(1 for v in square_dict.values() if v != 0)

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
        nonlocal best_poly, best_nz
        if len(exponents) != 12:
            return
        if any(c == 0 or abs(c) > 64 for c in coeffs):
            return
        if any(e < 0 or e >= 41 for e in exponents):
            return
        if len(set(exponents)) != 12:
            return
        nz = square_nonzero(exponents, coeffs)
        if nz < best_nz:
            best_nz = nz
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            best_poly = poly

    # Phase 1: Structured AP patterns
    patterns = [
        [1]*12,
        [1 if i%2==0 else -1 for i in range(12)],
        [1 if (i//2)%2==0 else -1 for i in range(12)],
        [1 if (i//3)%2==0 else -1 for i in range(12)],
        [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
        [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
        [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
        [1,2,1,2,1,2,1,2,1,2,1,2],
        [1,-2,1,-2,1,-2,1,-2,1,-2,1,-2],
        [1,1,1,1,-1,-1,-1,-1,1,1,1,1],
        [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
        [1,1,-1,1,-1,-1,1,-1,1,1,-1,1],
        [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
        [1,-1,1,1,-1,-1,1,-1,1,1,-1,-1],
    ]
    for d in range(1, 12):
        for a in range(0, max(0, 41 - 11*d)):
            if time.time() > deadline - 20:
                break
            exponents = [a + i*d for i in range(12)]
            if exponents[-1] >= 41:
                continue
            for cp in patterns:
                try_exponents_coeffs(exponents, cp)
                try_exponents_coeffs(exponents, [-c for c in cp])
        if time.time() > deadline - 20:
            break

    # Phase 2: Simulated Annealing
    random.seed(12345)

    def sa_search(max_iters=200000):
        nonlocal best_poly, best_nz
        # Initialize with a random valid config
        degree = random.randint(15, 40)
        exponents = sorted(random.sample(range(degree+1), 12))
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        current_nz = square_nonzero(exponents, coeffs)

        # SA parameters
        T = 5.0
        T_min = 0.1
        cooling = (T_min / T) ** (1.0 / max(1, max_iters))
        reheat_every = 30000
        reheat_T = 3.0

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Reheat
            if it > 0 and it % reheat_every == 0:
                T = reheat_T

            # Generate neighbor
            r = random.random()
            exp_copy = list(exponents)
            coef_copy = list(coeffs)

            if r < 0.35:
                # Change a coefficient
                idx = random.randint(0, 11)
                new_c = random.choice([-4,-3,-2,-1,1,2,3,4,5,6,8,12])
                if new_c != coef_copy[idx]:
                    coef_copy[idx] = new_c
            elif r < 0.65:
                # Move an exponent
                idx = random.randint(0, 11)
                shift = random.choice([-4,-3,-2,-1,1,2,3,4])
                new_e = exp_copy[idx] + shift
                if 0 <= new_e < 41 and new_e not in exp_copy:
                    exp_copy[idx] = new_e
                    exp_copy.sort()
                else:
                    continue
            elif r < 0.8:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                exp_copy[i], exp_copy[j] = exp_copy[j], exp_copy[i]
                exp_copy.sort()
            else:
                # Change two coefficients
                i, j = random.sample(range(12), 2)
                coef_copy[i] = random.choice([-3,-2,-1,1,2,3])
                coef_copy[j] = random.choice([-3,-2,-1,1,2,3])

            if any(c == 0 or abs(c) > 64 for c in coef_copy):
                continue
            if len(set(exp_copy)) != 12:
                continue

            new_nz = square_nonzero(exp_copy, coef_copy)
            delta = new_nz - current_nz

            if delta <= 0 or random.random() < np.exp(-delta / T):
                exponents = exp_copy
                coeffs = coef_copy
                current_nz = new_nz

                if new_nz < best_nz:
                    best_nz = new_nz
                    max_deg = max(exponents)
                    poly = [0] * (max_deg + 1)
                    for e, c in zip(exponents, coeffs):
                        poly[e] = c
                    best_poly = poly

            T *= cooling

    sa_search(200000)

    # Phase 3: Targeted local search from best found
    if best_poly is not None and time.time() < deadline - 8:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]
        local_nz = square_nonzero(best_exp, best_coef)

        # Greedy local search
        for it in range(50000):
            if time.time() > deadline - 5:
                break
            improved = False
            r = random.random()
            exp_copy = list(best_exp)
            coef_copy = list(best_coef)

            if r < 0.5:
                idx = random.randint(0, 11)
                for nc in [-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]:
                    if nc == coef_copy[idx]:
                        continue
                    coef_copy[idx] = nc
                    nz = square_nonzero(exp_copy, coef_copy)
                    if nz < local_nz:
                        local_nz = nz
                        best_coef = coef_copy
                        coef_copy[idx] = nc
                        improved = True
                        if nz < best_nz:
                            best_nz = nz
                            max_deg = max(exp_copy)
                            poly = [0] * (max_deg + 1)
                            for e, c in zip(exp_copy, coef_copy):
                                poly[e] = c
                            best_poly = poly
                        break
                    coef_copy[idx] = best_coef[idx]
            else:
                idx = random.randint(0, 11)
                for shift in [-3,-2,-1,1,2,3]:
                    new_e = exp_copy[idx] + shift
                    if 0 <= new_e < 41 and new_e not in exp_copy:
                        exp_copy[idx] = new_e
                        exp_copy.sort()
                        nz = square_nonzero(exp_copy, coef_copy)
                        if nz < local_nz:
                            local_nz = nz
                            best_exp = exp_copy
                            improved = True
                            if nz < best_nz:
                                best_nz = nz
                                max_deg = max(exp_copy)
                                poly = [0] * (max_deg + 1)
                                for e, c in zip(exp_copy, coef_copy):
                                    poly[e] = c
                                best_poly = poly
                            break
                        exp_copy = list(best_exp)
                    else:
                        continue
            if not improved and random.random() < 0.1:
                break

    # Phase 4: Product-of-two-6-term strategy
    if time.time() < deadline - 5:
        # If P = A*B where A and B each have 6 terms, P^2 = A^2*B^2
        # But we need P to have exactly 12 terms, so A*B must have 12 terms
        # Try: A has 6 terms, B has 6 terms, A*B has 12 terms
        # Then P^2 = A^2 * B^2, and we count nonzero terms in A^2*B^2
        # Actually P^2 = (A*B)^2 = A^2*B^2
        # This is the same as before. Let's try a different angle.
        # Try polynomials where P = Q + R with Q having 6 terms and R having 6 terms
        # P^2 = Q^2 + 2QR + R^2
        # If supports of Q^2, QR, R^2 overlap in specific ways, cancellations occur

        # Try structured: Q = sum of 6 terms with coefficients +1, R = sum of 6 terms with coefficients -1
        # P^2 = Q^2 - 2QR + R^2
        # If Q and R have the same exponents, P = Q - Q = 0, not useful
        # If Q and R have disjoint exponents, P^2 = Q^2 + R^2 (no cross terms cancel)
        # We need partial overlap for cancellations

        for d in range(1, 8):
            if time.time() > deadline - 3:
                break
            for a in range(0, max(0, 41 - 11*d)):
                if time.time() > deadline - 3:
                    break
                # 12 exponents in AP
                exps = [a + i*d for i in range(12)]
                if exps[-1] >= 41:
                    continue
                # Split into first 6 and last 6
                for c_val in [1, 2, 3]:
                    # First 6 positive, last 6 negative
                    coefs = [c_val]*6 + [-c_val]*6
                    try_exponents_coeffs(exps, coefs)
                    # Alternating
                    coefs2 = [c_val if i%2==0 else -c_val for i in range(12)]
                    try_exponents_coeffs(exps, coefs2)

    # Ensure we have a valid result
    if best_poly is None or not validate(best_poly):
        # Default: 12 terms all with coefficient 1
        best_poly = [1]*12

    return best_poly

# EVOLVE-BLOCK-END
