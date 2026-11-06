"""Structural redesign: fast sparse convolution + simulated annealing with coefficient optimization."""
import time
import math
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    def elapsed_ok():
        return time.time() < deadline

    def square_nonzero_fast(exponents, coefficients):
        """Fast square using numpy convolution. Returns nonzero count."""
        max_deg = max(exponents)
        arr = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exponents, coefficients):
            arr[e] = c
        sq = np.convolve(arr, arr)
        return int(np.count_nonzero(sq))

    def square_nonzero_sparse(exponents, coefficients):
        """Sparse square: count nonzero terms without full convolution."""
        n = len(exponents)
        sq = defaultdict(int)
        for i in range(n):
            ci = coefficients[i]
            ei = exponents[i]
            for j in range(i, n):
                e = ei + exponents[j]
                if i == j:
                    sq[e] += ci * ci
                else:
                    sq[e] += 2 * ci * coefficients[j]
        return sum(1 for v in sq.values() if v != 0)

    def make_poly(exponents, coefficients):
        max_deg = max(exponents) if exponents else 0
        coeffs = [0] * (max_deg + 1)
        for e, c in zip(exponents, coefficients):
            coeffs[e] += c
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if not coeffs or coeffs[-1] == 0:
            return False
        nz = sum(1 for c in coeffs if c != 0)
        if nz != 12:
            return False
        for c in coeffs:
            if c < -64 or c > 64:
                return False
        return True

    best_score = 10**9
    best_exponents = None
    best_coefficients = None

    def consider(exponents, coefficients):
        nonlocal best_score, best_exponents, best_coefficients
        if len(exponents) != 12:
            return
        if len(set(exponents)) != 12:
            return
        if max(exponents) > 40:
            return
        for c in coefficients:
            if c == 0 or c < -64 or c > 64:
                return
        nz = square_nonzero_sparse(exponents, coefficients)
        if nz < best_score:
            best_score = nz
            best_exponents = list(exponents)
            best_coefficients = list(coefficients)

    def local_polish(exponents, coefficients, max_iters=2000):
        """Local search: try small changes to reduce nonzero count."""
        exps = list(exponents)
        coeffs = list(coefficients)
        current_nz = square_nonzero_sparse(exps, coeffs)
        best_local_nz = current_nz
        best_local_e = list(exps)
        best_local_c = list(coeffs)

        rng = random.Random(hash(tuple(exps)) % (2**31))

        for it in range(max_iters):
            if not elapsed_ok():
                break
            i = rng.randint(0, 11)
            move_type = rng.random()

            if move_type < 0.6:
                # Exponent move
                delta = rng.choice([-3, -2, -1, 1, 2, 3, -5, 5, -7, 7])
                new_e = exps[i] + delta
                if new_e < 0 or new_e > 40:
                    continue
                others = set(exps[:i] + exps[i+1:])
                if new_e in others:
                    continue
                exps[i] = new_e
                new_nz = square_nonzero_sparse(exps, coeffs)
                if new_nz <= current_nz:
                    current_nz = new_nz
                    if new_nz < best_local_nz:
                        best_local_nz = new_nz
                        best_local_e = list(exps)
                        best_local_c = list(coeffs)
                else:
                    exps[i] -= delta
            else:
                # Coefficient move: try all small coefficient values
                old_c = coeffs[i]
                new_c = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, -6, 6, -8, 8])
                if new_c == old_c or new_c == 0:
                    continue
                coeffs[i] = new_c
                new_nz = square_nonzero_sparse(exps, coeffs)
                if new_nz <= current_nz:
                    current_nz = new_nz
                    if new_nz < best_local_nz:
                        best_local_nz = new_nz
                        best_local_e = list(exps)
                        best_local_c = list(coeffs)
                else:
                    coeffs[i] = old_c

        return best_local_e, best_local_c, best_local_nz

    def simulated_annealing(init_exponents, init_coefficients, max_iters=5000, t0=1.0, t_end=0.01):
        exponents = list(init_exponents)
        coefficients = list(init_coefficients)
        current_nz = square_nonzero_sparse(exponents, coefficients)
        best_local_nz = current_nz
        best_local_e = list(exponents)
        best_local_c = list(coefficients)

        rng = random.Random(hash(tuple(exponents)) % (2**31))

        for it in range(max_iters):
            if not elapsed_ok():
                break
            t = t0 * (t_end / t0) ** (it / max_iters)

            move_type = rng.random()
            i = rng.randint(0, 11)

            if move_type < 0.5:
                # Exponent move
                delta = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -8, 8, -12, 12])
                new_e = exponents[i] + delta
                if new_e < 0 or new_e > 40:
                    continue
                others = set(exponents[:i] + exponents[i+1:])
                if new_e in others:
                    continue
                exponents[i] = new_e
                new_nz = square_nonzero_sparse(exponents, coefficients)
                delta_e = new_nz - current_nz
                if delta_e < 0 or rng.random() < math.exp(-delta_e / max(t, 1e-10)):
                    current_nz = new_nz
                    if new_nz < best_local_nz:
                        best_local_nz = new_nz
                        best_local_e = list(exponents)
                        best_local_c = list(coefficients)
                else:
                    exponents[i] -= delta
            elif move_type < 0.8:
                # Coefficient move
                old_c = coefficients[i]
                new_c = rng.choice([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16, -32, 32])
                if new_c == old_c or new_c == 0:
                    continue
                coefficients[i] = new_c
                new_nz = square_nonzero_sparse(exponents, coefficients)
                delta_c = new_nz - current_nz
                if delta_c < 0 or rng.random() < math.exp(-delta_c / max(t, 1e-10)):
                    current_nz = new_nz
                    if new_nz < best_local_nz:
                        best_local_nz = new_nz
                        best_local_e = list(exponents)
                        best_local_c = list(coefficients)
                else:
                    coefficients[i] = old_c
            else:
                # Swap two exponents
                j = rng.randint(0, 11)
                if j == i:
                    continue
                exponents[i], exponents[j] = exponents[j], exponents[i]
                new_nz = square_nonzero_sparse(exponents, coefficients)
                delta_s = new_nz - current_nz
                if delta_s < 0 or rng.random() < math.exp(-delta_s / max(t, 1e-10)):
                    current_nz = new_nz
                    if new_nz < best_local_nz:
                        best_local_nz = new_nz
                        best_local_e = list(exponents)
                        best_local_c = list(coefficients)
                else:
                    exponents[i], exponents[j] = exponents[j], exponents[i]

        return best_local_e, best_local_c, best_local_nz

    # Generate diverse initial candidates
    candidates = []

    # Arithmetic progression with various steps and offsets
    for base in range(0, 20):
        for step in [1, 2, 3, 4, 5]:
            exps = [base + i * step for i in range(12)]
            if max(exps) > 40:
                continue
            for mask in range(64):
                signs = [1 if (mask >> (i % 6)) & 1 else -1 for i in range(12)]
                candidates.append((exps, signs))

    # Two-group structure: 6 terms in group A, 6 in group B
    for gap in range(15, 41):
        for step_a in [1, 2, 3]:
            for step_b in [1, 2, 3]:
                exps_a = [i * step_a for i in range(6)]
                exps_b = [gap + i * step_b for i in range(6)]
                exps = sorted(exps_a + exps_b)
                if len(set(exps)) != 12 or max(exps) > 40:
                    continue
                for mask in range(32):
                    signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
                    candidates.append((exps, signs))

    # Geometric-like structures
    for base_set in [[0,1,2,3,4,5], [0,1,2,4,5,7], [0,2,4,6,8,10], [0,1,3,5,7,9], [0,2,4,6,8,12], [0,1,3,4,6,8]]:
        for mult in [1, 2, 3]:
            exps = [e * mult for e in base_set]
            if max(exps) > 40 or len(set(exps)) != 6:
                continue
            offset = max(exps) + 1
            exps2 = exps + [e + offset for e in base_set]
            exps2 = sorted(exps2)
            if len(set(exps2)) != 12 or max(exps2) > 40:
                continue
            for mask in range(32):
                signs = [1 if (mask >> (i % 5)) & 1 else -1 for i in range(12)]
                candidates.append((exps2, signs))

    # Random candidates
    rng = random.Random(42)
    for _ in range(3000):
        exps = sorted(rng.sample(range(0, 41), 12))
        coeffs = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        candidates.append((exps, coeffs))

    # Evaluate candidates quickly, sort by score
    scored = []
    for exps, coeffs in candidates:
        if not elapsed_ok():
            break
        if len(exps) != 12 or len(set(exps)) != 12:
            continue
        if max(exps) > 40:
            continue
        if any(c == 0 or c < -64 or c > 64 for c in coeffs):
            continue
        nz = square_nonzero_sparse(exps, coeffs)
        scored.append((nz, exps, coeffs))

    scored.sort(key=lambda x: x[0])

    # Run SA on top candidates
    for nz, exps, coeffs in scored[:200]:
        if not elapsed_ok():
            break
        new_e, new_c, new_nz = simulated_annealing(exps, coeffs, max_iters=4000)
        consider(new_e, new_c)

    # Local polish on the current best
    if best_exponents is not None and elapsed_ok():
        for _ in range(8):
            if not elapsed_ok():
                break
            new_e, new_c, new_nz = local_polish(best_exponents, best_coefficients, max_iters=3000)
            consider(new_e, new_c)
            # Also try SA from the polished result
            new_e2, new_c2, new_nz2 = simulated_annealing(new_e, new_c, max_iters=5000)
            consider(new_e2, new_c2)

    # Final random restarts
    if elapsed_ok():
        for _ in range(500):
            if not elapsed_ok():
                break
            exps = sorted(rng.sample(range(0, 41), 12))
            coeffs = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            new_e, new_c, new_nz = simulated_annealing(exps, coeffs, max_iters=2000)
            consider(new_e, new_c)

    # Build final polynomial
    if best_exponents is None:
        return [1] * 12
    poly = make_poly(best_exponents, best_coefficients)
    if is_valid(poly):
        return poly
    return [1] * 12
# EVOLVE-BLOCK-END
