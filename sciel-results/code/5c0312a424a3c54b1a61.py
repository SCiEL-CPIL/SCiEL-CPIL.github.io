"""Structural redesign: fast convolution + simulated annealing on exponent structure."""
import time
import math
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s kill

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

    def square_dict(exponents, coefficients):
        """Return dict of exponent->coeff for P^2 (for detailed analysis)."""
        sq = defaultdict(int)
        n = len(exponents)
        for i in range(n):
            ci = coefficients[i]
            ei = exponents[i]
            for j in range(i, n):
                e = ei + exponents[j]
                if i == j:
                    sq[e] += ci * ci
                else:
                    sq[e] += 2 * ci * coefficients[j]
        return {e: c for e, c in sq.items() if c != 0}

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
        nz = square_nonzero_fast(exponents, coefficients)
        if nz < best_score:
            best_score = nz
            best_exponents = list(exponents)
            best_coefficients = list(coefficients)

    # Simulated annealing with structural moves (no spectral gradient to save time)
    def simulated_annealing(init_exponents, init_coefficients, max_iters=2000, t0=1.0, t_end=0.01):
        exponents = list(init_exponents)
        coefficients = list(init_coefficients)
        current_nz = square_nonzero_fast(exponents, coefficients)
        best_local_nz = current_nz
        best_local_e = list(exponents)
        best_local_c = list(coefficients)

        rng = random.Random(hash(tuple(exponents)) % (2**31))

        for it in range(max_iters):
            if not elapsed_ok():
                break
            t = t0 * (t_end / t0) ** (it / max_iters)

            # Choose a move type
            move_type = rng.random()
            i = rng.randint(0, 11)

            if move_type < 0.5:
                # Exponent move: pick new exponent
                delta = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -8, 8, -12, 12])
                new_e = exponents[i] + delta
                if new_e < 0 or new_e > 40:
                    continue
                others = set(exponents[:i] + exponents[i+1:])
                if new_e in others:
                    continue
                exponents[i] = new_e
                new_nz = square_nonzero_fast(exponents, coefficients)
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
                # Coefficient move (random search in small range)
                old_c = coefficients[i]
                # Try small integer changes
                new_c = old_c + rng.choice([-2, -1, 1, 2])
                if new_c == 0 or new_c < -64 or new_c > 64:
                    continue
                coefficients[i] = new_c
                new_nz = square_nonzero_fast(exponents, coefficients)
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
                # Swap two exponents (structural)
                j = rng.randint(0, 11)
                if j == i:
                    continue
                exponents[i], exponents[j] = exponents[j], exponents[i]
                new_nz = square_nonzero_fast(exponents, coefficients)
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

    # Final polish: local descent on coefficients with exact evaluation
    def polish_coefficients(exponents, max_iters=1000):
        """Refine coefficients using local descent with exact square evaluation."""
        # Initialize with small integers
        coefficients = [1 if i % 2 == 0 else -1 for i in range(12)]
        current_nz = square_nonzero_fast(exponents, coefficients)
        best_nz = current_nz
        best_c = list(coefficients)
        no_improve = 0
        rng = random.Random(hash(tuple(exponents)) % (2**31))

        for it in range(max_iters):
            if not elapsed_ok():
                break
            if no_improve > 100:
                break

            i = rng.randint(0, 11)
            old_c = coefficients[i]
            # Try neighboring values
            neighbors = [c for c in range(max(-64, old_c - 4), min(65, old_c + 5)) if c != 0 and c != old_c]
            if not neighbors:
                neighbors = [1, -1, 2, -2]
            new_c = rng.choice(neighbors)
            coefficients[i] = new_c
            new_nz = square_nonzero_fast(exponents, coefficients)

            if new_nz < current_nz:
                current_nz = new_nz
                no_improve = 0
                if new_nz < best_nz:
                    best_nz = new_nz
                    best_c = list(coefficients)
            else:
                coefficients[i] = old_c
                no_improve += 1

        return best_c, best_nz

    # Generate diverse initial candidates (reduced count)
    candidates = []

    # Arithmetic progression bases with various signs
    for base in range(0, 15):
        for step in [1, 2, 3]:
            exps = [base + i * step for i in range(12)]
            if max(exps) > 40:
                continue
            for mask in range(16):
                signs = [1 if (mask >> (i % 4)) & 1 else -1 for i in range(12)]
                candidates.append((exps, signs))

    # Two-group structure: 6 terms in group A, 6 in group B, far apart
    for gap in range(20, 41):
        for step_a in [1, 2]:
            for step_b in [1, 2]:
                exps_a = [i * step_a for i in range(6)]
                exps_b = [gap + i * step_b for i in range(6)]
                exps = sorted(exps_a + exps_b)
                if len(set(exps)) != 12 or max(exps) > 40:
                    continue
                for mask in range(8):
                    signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
                    candidates.append((exps, signs))

    # Random
    rng = random.Random(42)
    for _ in range(500):
        exps = sorted(rng.sample(range(0, 41), 12))
        coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
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
        nz = square_nonzero_fast(exps, coeffs)
        scored.append((nz, exps, coeffs))

    scored.sort(key=lambda x: x[0])

    # Run SA on top candidates
    for nz, exps, coeffs in scored[:50]:
        if not elapsed_ok():
            break
        new_e, new_c, new_nz = simulated_annealing(exps, coeffs, max_iters=1500)
        consider(new_e, new_c)

    # More SA iterations on the current best
    if best_exponents is not None and elapsed_ok():
        for _ in range(3):
            if not elapsed_ok():
                break
            new_e, new_c, new_nz = simulated_annealing(best_exponents, best_coefficients, max_iters=3000)
            consider(new_e, new_c)

    # Final random restarts
    if elapsed_ok():
        for _ in range(100):
            if not elapsed_ok():
                break
            exps = sorted(rng.sample(range(0, 41), 12))
            coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
            new_e, new_c, new_nz = simulated_annealing(exps, coeffs, max_iters=1000)
            consider(new_e, new_c)

    # Polish pass on the best exponent structure found
    if best_exponents is not None and elapsed_ok():
        polished_c, polished_nz = polish_coefficients(best_exponents, max_iters=2000)
        consider(best_exponents, polished_c)

    # Build final polynomial
    if best_exponents is None:
        return [1] * 12
    poly = make_poly(best_exponents, best_coefficients)
    if is_valid(poly):
        return poly
    return [1] * 12
# EVOLVE-BLOCK-END
