import time
import numpy as np
import random
from itertools import combinations, product

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_score_fast(exps, coeffs):
        """
        Fast computation of nonzero terms in P(x)^2.
        exps: list of 12 exponents (sorted)
        coeffs: list of 12 coefficients
        Returns count of nonzero terms, or inf if invalid.
        """
        if not exps or not coeffs:
            return float('inf')

        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')

        # Build dense array
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c

        q = np.convolve(p, p)
        count = int(np.count_nonzero(q))
        return count

    def _update_best(exps, coeffs, score):
        """Update the global best candidate if improved."""
        nonlocal best_score, best_candidate
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def search_sign_patterns(exps, max_patterns=5000):
        """
        For a fixed set of exponents, search over sign patterns and coefficient magnitudes.
        Returns best score and corresponding coeffs.
        """
        nonlocal best_score, best_candidate

        # Start with all +1
        best_local_score = compute_score_fast(exps, [1] * 12)
        _update_best(exps, [1] * 12, best_local_score)

        if best_local_score <= 12:
            return

        # Try flipping signs one at a time (local search on signs)
        current_coeffs = [1] * 12
        current_score = best_local_score

        # Iterative improvement: try flipping each sign
        improved = True
        while improved:
            improved = False
            for i in range(12):
                new_coeffs = current_coeffs[:]
                new_coeffs[i] = -new_coeffs[i]
                new_score = compute_score_fast(exps, new_coeffs)
                if new_score < current_score:
                    current_coeffs = new_coeffs
                    current_score = new_score
                    improved = True
                    _update_best(exps, current_coeffs, current_score)
                    if current_score <= 12:
                        return

        # Random sign flips with restarts
        random.seed(hash(tuple(exps)) % (2**32))
        for restart in range(max_patterns):
            if time.time() > deadline - 5:
                break

            # Start from current best or random
            if restart == 0:
                coeffs = current_coeffs[:]
            else:
                coeffs = [random.choice([-1, 1]) for _ in range(12)]

            score = compute_score_fast(exps, coeffs)
            _update_best(exps, coeffs, score)

            if score <= 12:
                return

            # Local improvement
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, 11)
                new_coeffs = coeffs[:]
                new_coeffs[i] = -new_coeffs[i]
                new_score = compute_score_fast(exps, new_coeffs)
                if new_score < score:
                    coeffs = new_coeffs
                    score = new_score
                    _update_best(exps, coeffs, score)
                    if score <= 12:
                        return

    def search_exponent_sets():
        """
        Search over different exponent sets, using structured generation.
        """
        nonlocal best_score, best_candidate

        # Strategy: try various structured exponent sets
        # 1. Consecutive blocks
        for start in range(0, 30):
            if time.time() > deadline - 10:
                return
            exps = [start + i for i in range(12)]
            if max(exps) <= 40:
                search_sign_patterns(exps, max_patterns=200)

        # 2. Evenly spaced
        for gap in [2, 3, 4, 5]:
            for start in range(0, 41 - gap * 11):
                if time.time() > deadline - 10:
                    return
                exps = [start + i * gap for i in range(12)]
                if max(exps) <= 40:
                    search_sign_patterns(exps, max_patterns=100)

        # 3. Random sparse sets
        random.seed(42)
        for _ in range(50):
            if time.time() > deadline - 10:
                return
            exps = sorted(random.sample(range(0, 41), 12))
            search_sign_patterns(exps, max_patterns=100)

        # 4. Clustered sets
        random.seed(123)
        for _ in range(30):
            if time.time() > deadline - 10:
                return
            # Two clusters of 6
            c1 = random.randint(0, 15)
            c2 = random.randint(20, 35)
            exps = []
            for i in range(6):
                e = c1 + random.randint(-2, 2)
                if 0 <= e <= 40 and e not in exps:
                    exps.append(e)
            for i in range(6):
                e = c2 + random.randint(-2, 2)
                if 0 <= e <= 40 and e not in exps:
                    exps.append(e)
            while len(exps) < 12:
                e = random.randint(0, 40)
                if e not in exps:
                    exps.append(e)
            exps = sorted(exps[:12])
            search_sign_patterns(exps, max_patterns=100)

    def full_local_search(exps, coeffs, max_iters=2000):
        """
        Full local search on both exponents and coefficients.
        """
        nonlocal best_score, best_candidate

        # Ensure sorted
        idx = np.argsort(exps)
        exps = [exps[i] for i in idx]
        coeffs = [coeffs[i] for i in idx]

        current_score = compute_score_fast(exps, coeffs)
        _update_best(exps, coeffs, current_score)

        if current_score <= 12:
            return

        T = 3.0
        T_min = 0.05
        cooling = 0.997

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            new_exps = exps[:]
            new_coeffs = coeffs[:]
            move_type = random.random()

            if move_type < 0.3:
                # Flip sign
                i = random.randint(0, 11)
                new_coeffs[i] = -new_coeffs[i]
            elif move_type < 0.5:
                # Shift exponent
                i = random.randint(0, 11)
                delta = random.choice([-2, -1, 1, 2])
                new_e = new_exps[i] + delta
                if 0 <= new_e <= 40 and new_e not in new_exps:
                    new_exps[i] = new_e
                else:
                    continue
            elif move_type < 0.65:
                # Change coefficient magnitude
                i = random.randint(0, 11)
                new_c = random.choice([-3, -2, -1, 1, 2, 3])
                if new_c != new_coeffs[i]:
                    new_coeffs[i] = new_c
                else:
                    continue
            elif move_type < 0.8:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
                new_coeffs[i], new_coeffs[j] = new_coeffs[j], new_coeffs[i]
            else:
                # Jump
                i = random.randint(0, 11)
                new_e = random.randint(0, 40)
                if new_e not in new_exps:
                    new_exps[i] = new_e
                else:
                    continue

            nidx = np.argsort(new_exps)
            new_exps = [new_exps[i] for i in nidx]
            new_coeffs = [new_coeffs[i] for i in nidx]

            new_score = compute_score_fast(new_exps, new_coeffs)

            if new_score < current_score:
                exps = new_exps
                coeffs = new_coeffs
                current_score = new_score
                _update_best(exps, coeffs, current_score)
                if current_score <= 12:
                    break
            else:
                delta_score = new_score - current_score
                if T > T_min:
                    accept_prob = np.exp(-delta_score / T)
                    if random.random() < accept_prob:
                        exps = new_exps
                        coeffs = new_coeffs
                        current_score = new_score

            T = max(T * cooling, T_min)

    # --- Main search ---

    # Phase 1: Structured exponent set search with sign optimization
    search_exponent_sets()

    if best_score <= 12:
        return best_candidate

    # Phase 2: Full local search from best candidate
    if time.time() < deadline - 15:
        best_exps = []
        best_coeffs = []
        for i, c in enumerate(best_candidate):
            if c != 0:
                best_exps.append(i)
                best_coeffs.append(c)

        if len(best_exps) == 12:
            full_local_search(best_exps, best_coeffs, max_iters=3000)

    # Phase 3: Random restarts
    if time.time() < deadline - 10 and best_score > 12:
        random.seed(789)
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            full_local_search(exps, coeffs, max_iters=1000)
            if best_score <= 12:
                break

    return best_candidate

# EVOLVE-BLOCK-END
