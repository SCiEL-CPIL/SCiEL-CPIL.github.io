import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    best_score = float('inf')
    best_poly = [1] * 12  # baseline

    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if not is_valid(coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Try a few known simple polynomials first
    for n in range(1, 42):
        coeffs = [0] * n
        coeffs[-1] = 1
        # Add 11 more random terms
        for _ in range(11):
            pos = random.randint(0, n - 1)
            if coeffs[pos] == 0:
                coeffs[pos] = random.choice([-1, 1])
        # Ensure exactly 12 nonzero
        nz = [i for i, c in enumerate(coeffs) if c != 0]
        if len(nz) < 12:
            for i in range(n):
                if len(nz) >= 12:
                    break
                if i not in nz:
                    coeffs[i] = random.choice([-1, 1])
                    nz.append(i)
        elif len(nz) > 12:
            keep = random.sample(nz, 12)
            new_coeffs = [0] * n
            for i in keep:
                new_coeffs[i] = coeffs[i]
            coeffs = new_coeffs
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        if len(coeffs) > 0 and coeffs[-1] != 0:
            try_poly(coeffs)

    # Core strategy: support-compression search
    # 1. Find a small support S where |supp(P^2)| is minimized
    # 2. Greedily expand to 12 terms without increasing |supp(P^2)|

    def find_best_support(max_degree=40):
        """Find a support of size 12 that minimizes the square's nonzero count."""
        best_sq_count = float('inf')
        best_support = None
        best_coeffs = None

        # Try various support patterns
        # Pattern 1: Consecutive blocks
        for start in range(0, max_degree - 11):
            support = list(range(start, start + 12))
            coeffs = [0] * (max_degree + 1)
            for i in support:
                coeffs[i] = 1
            sq = poly_square(coeffs)
            cnt = count_nonzero(sq)
            if cnt < best_sq_count:
                best_sq_count = cnt
                best_support = support
                best_coeffs = coeffs

        # Pattern 2: Arithmetic progressions
        for a in range(0, max_degree):
            for d in range(1, 10):
                support = [a + i * d for i in range(12)]
                if max(support) > max_degree:
                    continue
                coeffs = [0] * (max_degree + 1)
                for i in support:
                    coeffs[i] = 1
                sq = poly_square(coeffs)
                cnt = count_nonzero(sq)
                if cnt < best_sq_count:
                    best_sq_count = cnt
                    best_support = support
                    best_coeffs = coeffs

        # Pattern 3: Random supports with local search
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            support = sorted(random.sample(range(max_degree + 1), 12))
            coeffs = [0] * (max_degree + 1)
            for i in support:
                coeffs[i] = 1
            sq = poly_square(coeffs)
            cnt = count_nonzero(sq)
            if cnt < best_sq_count:
                best_sq_count = cnt
                best_support = support
                best_coeffs = coeffs

            # Local search: swap one term
            if cnt < best_sq_count + 2:
                for _ in range(50):
                    if time.time() > deadline - 10:
                        break
                    i = random.randint(0, 11)
                    new_pos = random.randint(0, max_degree)
                    if new_pos in support:
                        continue
                    new_support = support[:]
                    new_support[i] = new_pos
                    new_support.sort()
                    new_coeffs = [0] * (max_degree + 1)
                    for j in new_support:
                        new_coeffs[j] = 1
                    new_sq = poly_square(new_coeffs)
                    new_cnt = count_nonzero(new_sq)
                    if new_cnt < cnt:
                        cnt = new_cnt
                        support = new_support
                        coeffs = new_coeffs
                        if cnt < best_sq_count:
                            best_sq_count = cnt
                            best_support = new_support
                            best_coeffs = new_coeffs

        return best_sq_count, best_support, best_coeffs

    # Run support compression search
    for restart in range(20):
        if time.time() > deadline - 5:
            break

        # Find a good starting support
        sq_count, support, coeffs = find_best_support(40)

        if support is None:
            continue

        # Greedily try to add terms (change coefficients) without increasing square count
        current_sq_count = sq_count
        current_coeffs = coeffs[:]
        current_support = support[:]

        # Try flipping signs
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            i = random.randint(0, 11)
            pos = current_support[i]
            current_coeffs[pos] = -current_coeffs[pos]
            sq = poly_square(current_coeffs)
            cnt = count_nonzero(sq)
            if cnt <= current_sq_count:
                current_sq_count = cnt
            else:
                current_coeffs[pos] = -current_coeffs[pos]  # revert

        # Try moving terms to new positions
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            i = random.randint(0, 11)
            old_pos = current_support[i]
            new_pos = random.randint(0, 40)
            if new_pos == old_pos:
                continue
            # Check if new_pos is already in support
            if new_pos in current_support:
                continue

            # Try the move
            val = current_coeffs[old_pos]
            current_coeffs[old_pos] = 0
            if new_pos >= len(current_coeffs):
                current_coeffs.extend([0] * (new_pos - len(current_coeffs) + 1))
            current_coeffs[new_pos] = val

            sq = poly_square(current_coeffs)
            cnt = count_nonzero(sq)

            if cnt <= current_sq_count:
                current_sq_count = cnt
                current_support[i] = new_pos
                current_support.sort()
            else:
                # Revert
                current_coeffs[new_pos] = 0
                current_coeffs[old_pos] = val
                current_support[i] = old_pos
                current_support.sort()

        # Evaluate final candidate
        while current_coeffs and current_coeffs[-1] == 0:
            current_coeffs.pop()
        if len(current_coeffs) > 0 and current_coeffs[-1] != 0:
            try_poly(current_coeffs)

        # If we found a very good score, do more targeted search
        if best_score <= 14:
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                perturbed = list(best_poly)
                # Randomly perturb one coefficient
                nz = [i for i, c in enumerate(perturbed) if c != 0]
                if nz:
                    idx = random.choice(nz)
                    perturbed[idx] = random.choice([-1, 0, 1])
                # Fix to 12 nonzero
                nz2 = [i for i, c in enumerate(perturbed) if c != 0]
                if len(nz2) > 12:
                    keep = random.sample(nz2, 12)
                    new_perturbed = [0] * len(perturbed)
                    for i in keep:
                        new_perturbed[i] = perturbed[i]
                    perturbed = new_perturbed
                elif len(nz2) < 12:
                    for i in range(len(perturbed)):
                        if len(nz2) >= 12:
                            break
                        if i not in nz2:
                            perturbed[i] = random.choice([-1, 1])
                            nz2.append(i)
                while perturbed and perturbed[-1] == 0:
                    perturbed.pop()
                if len(perturbed) > 0 and perturbed[-1] != 0:
                    try_poly(perturbed)

    # Final local search around best
    if time.time() < deadline - 2:
        for _ in range(300):
            if time.time() > deadline - 2:
                break
            perturbed = list(best_poly)
            # Single flip
            nz = [i for i, c in enumerate(perturbed) if c != 0]
            if nz:
                idx = random.choice(nz)
                perturbed[idx] = -perturbed[idx]
            try_poly(perturbed)

    return best_poly
