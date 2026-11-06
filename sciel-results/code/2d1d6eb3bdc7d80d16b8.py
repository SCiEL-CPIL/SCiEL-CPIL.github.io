"""Sparse squares of 12-term integer polynomials – AP-focused optimization."""
import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85.0

    def square_nonzero_exact(exponents, coeffs):
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=82)
        return int(np.count_nonzero(result))

    def is_valid(exp, coef):
        if len(exp) != 12 or len(coef) != 12:
            return False
        if len(set(exp)) != 12:
            return False
        if any(e < 0 or e >= 41 for e in exp):
            return False
        if any(c == 0 or abs(c) > 64 for c in coef):
            return False
        return True

    def make_poly(exp, coef):
        max_deg = max(exp)
        poly = [0] * (max_deg + 1)
        for e, c in zip(exp, coef):
            poly[e] = int(c)
        return poly

    best_exp = None
    best_coef = None
    best_nz = float('inf')

    def try_candidate(exp, coef):
        nonlocal best_exp, best_coef, best_nz
        if not is_valid(exp, coef):
            return
        nz = square_nonzero_exact(exp, coef)
        if nz < best_nz:
            best_nz = nz
            best_exp = list(exp)
            best_coef = list(coef)

    # Phase 1: Focused search on Arithmetic Progressions (AP)
    # For AP exponents, the square's sparsity is determined by the coefficient vector's autocorrelation.
    # We optimize the coefficients for each AP.
    
    # Small set of promising coefficient patterns to seed the local search
    seed_coefs = [
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        [1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1],
        [2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2],
        [1, 1, -2, -1, 1, 1, -2, -1, 1, 1, -2, -1],
        [1, 3, -1, -3, 1, 3, -1, -3, 1, 3, -1, -3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    def local_search_ap(exp, init_coef, max_iters, dl):
        cur_coef = list(init_coef)
        cur_nz = square_nonzero_exact(exp, cur_coef)
        best_c = list(cur_coef)
        best_n = cur_nz
        
        for _ in range(max_iters):
            if time.time() > dl:
                break
            idx = random.randint(0, 11)
            # Perturb coefficient
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            new_c = cur_coef[idx] + delta
            if new_c == 0 or abs(new_c) > 64:
                continue
            new_coef = list(cur_coef)
            new_coef[idx] = new_c
            
            new_nz = square_nonzero_exact(exp, new_coef)
            if new_nz < cur_nz:
                cur_nz = new_nz
                cur_coef = new_coef
                if cur_nz < best_n:
                    best_n = cur_nz
                    best_c = list(cur_coef)
            # Accept worse with small probability (simulated annealing lite)
            elif new_nz - cur_nz <= 2 and random.random() < 0.1:
                cur_nz = new_nz
                cur_coef = new_coef
                
        return best_c, best_n

    # Iterate over AP steps and starts
    for d in range(1, 8):
        if time.time() > deadline - 30:
            break
        max_a = 40 - 11 * d
        if max_a < 0:
            continue
        for a in range(0, max_a + 1):
            if time.time() > deadline - 30:
                break
            exp = [a + i * d for i in range(12)]
            
            # Try each seed coefficient pattern with local search
            for sc in seed_coefs:
                if time.time() > deadline - 30:
                    break
                # Run a short local search
                bc, bn = local_search_ap(exp, sc, 500, deadline - 30)
                try_candidate(exp, bc)

    # Phase 2: Broader SA on the best found so far
    def simulated_annealing(initial_exp, initial_coef, max_iters, t_start, t_end, dl):
        cur_exp = list(initial_exp)
        cur_coef = list(initial_coef)
        cur_nz = square_nonzero_exact(cur_exp, cur_coef)

        best_e = list(cur_exp)
        best_c = list(cur_coef)
        best_n = cur_nz

        n = 12
        for it in range(max_iters):
            if time.time() > dl:
                break

            progress = it / max(max_iters - 1, 1)
            temp = t_start * (t_end / t_start) ** progress

            r = random.random()
            new_exp = list(cur_exp)
            new_coef = list(cur_coef)

            if r < 0.30:
                idx = random.randint(0, n - 1)
                new_c = random.choice([-8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8])
                if new_c == cur_coef[idx]:
                    continue
                new_coef[idx] = new_c
            elif r < 0.50:
                i, j = random.sample(range(n), 2)
                new_coef[i] = random.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 6])
                new_coef[j] = random.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 6])
            elif r < 0.65:
                idx = random.randint(0, n - 1)
                shift = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                new_e = cur_exp[idx] + shift
                if 0 <= new_e < 41 and new_e not in cur_exp:
                    new_exp[idx] = new_e
                    new_exp.sort()
                else:
                    continue
            elif r < 0.78:
                i, j = random.sample(range(n), 2)
                si = random.choice([-3, -2, -1, 1, 2, 3])
                sj = random.choice([-3, -2, -1, 1, 2, 3])
                new_ei = cur_exp[i] + si
                new_ej = cur_exp[j] + sj
                if (0 <= new_ei < 41 and 0 <= new_ej < 41 and
                        new_ei not in cur_exp and new_ej not in cur_exp and
                        new_ei != new_ej):
                    new_exp[i] = new_ei
                    new_exp[j] = new_ej
                    new_exp.sort()
                else:
                    continue
            elif r < 0.88:
                i, j = random.sample(range(n), 2)
                new_exp[i], new_exp[j] = new_exp[j], new_exp[i]
                new_exp.sort()
            else:
                idx = random.randint(0, n - 1)
                new_exp[idx] = random.randint(0, 40)
                new_exp.sort()
                if len(set(new_exp)) != n:
                    continue
                for _ in range(random.randint(1, 4)):
                    ci = random.randint(0, n - 1)
                    new_coef[ci] = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])

            if not is_valid(new_exp, new_coef):
                continue

            new_nz = square_nonzero_exact(new_exp, new_coef)

            delta = new_nz - cur_nz
            if delta <= 0:
                cur_exp = new_exp
                cur_coef = new_coef
                cur_nz = new_nz
            else:
                if temp > 0 and random.random() < np.exp(-delta / max(temp, 0.01)):
                    cur_exp = new_exp
                    cur_coef = new_coef
                    cur_nz = new_nz

            if cur_nz < best_n:
                best_n = cur_nz
                best_e = list(cur_exp)
                best_c = list(cur_coef)

        return best_e, best_c, best_n

    if best_exp is not None and time.time() < deadline - 5:
        e, c, nz = simulated_annealing(best_exp, best_coef, 15000, 5.0, 0.01, deadline - 3)
        if nz < best_nz:
            best_nz = nz
            best_exp = e
            best_coef = c

    # Final validation
    def validate(poly):
        if not poly or len(poly) > 41:
            return False
        if poly[-1] == 0:
            return False
        for c in poly:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in poly if c != 0) != 12:
            return False
        return True

    if best_exp is None or not is_valid(best_exp, best_coef):
        best_exp = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39, 40]
        best_coef = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    poly = make_poly(best_exp, best_coef)

    if not validate(poly):
        poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return poly
