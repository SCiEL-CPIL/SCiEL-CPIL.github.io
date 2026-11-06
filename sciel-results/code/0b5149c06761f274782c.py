"""Combined: sparse evaluation + pair-sum structural search + targeted polish."""
import time
import math
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80

    def elapsed_ok():
        return time.time() < deadline

    def square_nonzero_sparse(exponents, coefficients):
        """Sparse square: count nonzero terms."""
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

    def square_dict_sparse(exponents, coefficients):
        """Return dict of exponent->coeff for P^2."""
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
        nz = square_nonzero_sparse(exponents, coefficients)
        if nz < best_score:
            best_score = nz
            best_exponents = list(exponents)
            best_coefficients = list(coefficients)

    def pair_sum_structure_search(max_iters=3000):
        """Search for 6 pairs of exponents with shared sum targets."""
        rng = random.Random(12345)
        local_best_nz = 10**9
        local_best_e = None
        local_best_c = None

        for it in range(max_iters):
            if not elapsed_ok():
                break
            # Generate 6 target sums
            targets = sorted(rng.sample(range(0, 41), 6))
            # For each target, pick 2 exponents that sum to it
            exps = []
            valid = True
            used = set()
            for t in targets:
                found = False
                for a in range(0, t + 1):
                    b = t - a
                    if a < 0 or b < 0 or a > 40 or b > 40:
                        continue
                    if a == b:
                        continue
                    if a in used or b in used:
                        continue
                    exps.extend([a, b])
                    used.add(a)
                    used.add(b)
                    found = True
                    break
                if not found:
                    valid = False
                    break
            if not valid or len(set(exps)) != 12:
                continue

            # Try several coefficient patterns
            for coeff_pattern in range(16):
                coeffs = []
                for i in range(12):
                    if (coeff_pattern >> (i % 4)) & 1:
                        coeffs.append(rng.choice([1, 2, 3, 4, -1, -2, -3, -4]))
                    else:
                        coeffs.append(rng.choice([1, 2, 3, 4, -1, -2, -3, -4]))

                nz = square_nonzero_sparse(exps, coeffs)
                if nz < local_best_nz:
                    local_best_nz = nz
                    local_best_e = list(exps)
                    local_best_c = list(coeffs)
                    consider(exps, coeffs)

        return local_best_e, local_best_c, local_best_nz

    def targeted_polish(exponents, coefficients, max_iters=5000):
        """Focused local search on a good candidate."""
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

            if move_type < 0.7:
                # Exponent move
                delta = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, -6, 6, -8, 8])
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
                # Coefficient move
                old_c = coeffs[i]
                new_c = rng.choice([-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8])
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

    def simulated_annealing(init_exponents, init_coefficients, max_iters=4000, t0=1.0, t_end=0.01):
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

            if move_type < 0.55:
                # Exponent move
                delta = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -8, 8, -10, 10])
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
            elif move_type < 0.85:
                # Coefficient move
                old_c = coefficients[i]
                new_c = rng.choice([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16])
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

    # Phase 1: Pair-sum structural search
    if elapsed_ok():
        pe, pc, pnz = pair_sum_structure_search(max_iters=5000)
        if pe is not None:
            consider(pe, pc)

    # Phase 2: Generate diverse candidates
    candidates = []

    # Arithmetic progressions
    for base in range(0, 15):
        for step in [1, 2, 3, 4]:
            exps = [base + i * step for i in range(12)]
            if max(exps) > 40:
                continue
            for mask in range(32):
                signs = [1 if (mask >> (i % 5)) & 1 else -1 for i in range(12)]
                mags = [rng.choice([1, 2, 3]) for rng in [random.Random(i)] for i in range(12)]
                coeffs = [signs[i] * mags[i] for i in range(12)]
                candidates.append((exps, coeffs))

    # Two-group structure
    for gap in range(15, 41):
        for step_a in [1, 2, 3]:
            for step_b in [1, 2, 3]:
                exps_a = [i * step_a for i in range(6)]
                exps_b = [gap + i * step_b for i in range(6)]
                exps = sorted(exps_a + exps_b)
                if len(set(exps)) != 12 or max(exps) > 40:
                    continue
                for mask in range(16):
                    signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
                    mags = [random.Random(gap*step_a*step_b+i).choice([1,2,3]) for i in range(12)]
                    coeffs = [signs[i] * mags[i] for i in range(12)]
                    candidates.append((exps, coeffs))

    # Random candidates
    rng = random.Random(42)
    for _ in range(2000):
        exps = sorted(rng.sample(range(0, 41), 12))
        coeffs = [rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
        candidates.append((exps, coeffs))

    # Evaluate and sort
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

    # Phase 3: SA on top candidates
    for nz, exps, coeffs in scored[:100]:
        if not elapsed_ok():
            break
        new_e, new_c, new_nz = simulated_annealing(exps, coeffs, max_iters=3000)
        consider(new_e, new_c)

    # Phase 4: Targeted polish on best
    if best_exponents is not None and elapsed_ok():
        for _ in range(5):
            if not elapsed_ok():
                break
            new_e, new_c, new_nz = targeted_polish(best_exponents, best_coefficients, max_iters=4000)
            consider(new_e, new_c)

    # Phase 5: Final random restarts
    if elapsed_ok():
        rng2 = random.Random(99)
        for _ in range(300):
            if not elapsed_ok():
                break
            exps = sorted(rng2.sample(range(0, 41), 12))
            coeffs = [rng2.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            new_e, new_c, new_nz = simulated_annealing(exps, coeffs, max_iters=1500)
            consider(new_e, new_c)

    # Build final polynomial
    if best_exponents is None:
        return [1] * 12
    poly = make_poly(best_exponents, best_coefficients)
    if is_valid(poly):
        return poly
    return [1] * 12
# EVOLVE-BLOCK-END
