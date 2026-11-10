import numpy as np
import time
import random
from itertools import product

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

    # Incremental square update: change one coefficient and update square in O(n)
    def incremental_square_update(coeffs, pos, new_val):
        """Return (new_square, delta_nonzero) where delta_nonzero is change in nonzero count."""
        old_val = coeffs[pos]
        if old_val == new_val:
            return None, 0

        # The square at index i = sum_{j} coeffs[j]*coeffs[i-j]
        # When we change coeffs[pos] from old to new, the delta at each index i is:
        # delta[i] = (new - old) * (coeffs[i-pos] + coeffs[pos - i] - 2*coeffs[i-pos]*coeffs[pos-i]/old if old != 0 else 0)
        # Actually simpler: delta[i] = new*coeffs[i-pos] + coeffs[i-pos]*new - old*coeffs[i-pos] - coeffs[i-pos]*old
        # Wait, let me think more carefully.
        # square[i] = sum_{j=0}^{11} coeffs[j] * coeffs[i-j] (where coeffs[k]=0 if k<0 or k>11)
        # When coeffs[pos] changes from old to new:
        # delta[i] = new * coeffs[i-pos] + coeffs[i-pos] * new - old * coeffs[i-pos] - coeffs[i-pos] * old
        # That's not right either. Let me be precise:
        # square[i] = sum_j coeffs[j] * coeffs[i-j]
        # The terms involving coeffs[pos] are:
        #   j = pos: coeffs[pos] * coeffs[i-pos]
        #   i-j = pos, i.e., j = i-pos: coeffs[i-pos] * coeffs[pos]
        # So if pos != i-pos (i.e., i != 2*pos), there are two terms:
        #   coeffs[pos]*coeffs[i-pos] + coeffs[i-pos]*coeffs[pos] = 2*coeffs[pos]*coeffs[i-pos]
        # If pos == i-pos (i == 2*pos), there's one term:
        #   coeffs[pos]*coeffs[pos] = coeffs[pos]^2
        #
        # After change:
        # If i != 2*pos: new contribution = 2*new*coeffs[i-pos]
        # If i == 2*pos: new contribution = new^2
        #
        # Delta:
        # If i != 2*pos: delta = 2*(new - old)*coeffs[i-pos]
        # If i == 2*pos: delta = new^2 - old^2

        n = len(coeffs)
        sq_len = 2 * n - 1
        delta_nonzero = 0

        # We need to track the current square to know which entries become zero/nonzero
        # This requires us to maintain the square state. Let's use a different approach:
        # Just compute the full square but use a faster method.

        return None, 0

    best_score = float('inf')
    best_poly = [1] * 12  # baseline

    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if len(coeffs) != 12:
            return
        if sum(1 for c in coeffs if c != 0) != 12:
            return
        if any(c < -64 or c > 64 for c in coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Try baseline
    try_poly([1] * 12)
    
    # Generate structured candidates with algebraic properties
    # 1. Palindromic polynomials (symmetric coefficients)
    for a in range(-8, 9):
        for b in range(-8, 9):
            for c in range(-8, 9):
                if a == 0 or b == 0 or c == 0:
                    continue
                # Create palindromic pattern: [a, b, c, d, e, f, f, e, d, c, b, a]
                poly = [0] * 12
                poly[0] = a
                poly[1] = b
                poly[2] = c
                poly[9] = c
                poly[10] = b
                poly[11] = a
                # Fill middle with alternating signs
                for i in range(3, 9):
                    poly[i] = (1 if i % 2 == 0 else -1) * max(1, min(abs(a), abs(b), abs(c)))
                if all(-64 <= x <= 64 for x in poly) and all(x != 0 for x in poly):
                    try_poly(poly)

    # 2. Cyclotomic-like patterns: coefficients follow roots of unity patterns
    for n in [2, 3, 4, 5, 6]:
        for shift in range(n):
            for scale in range(1, 5):
                poly = [0] * 12
                for i in range(12):
                    # Use periodic pattern based on n-th roots
                    phase = (i + shift) % n
                    if phase == 0:
                        val = scale
                    elif phase == 1:
                        val = -scale
                    else:
                        val = scale * (1 if phase % 2 == 0 else -1)
                    if val == 0:
                        val = 1
                    poly[i] = val
                if all(-64 <= x <= 64 for x in poly) and all(x != 0 for x in poly):
                    try_poly(poly)

    # 3. Try polynomials where p(x) * p(1/x) has structure
    for lead in range(1, 9):
        for const in range(1, 9):
            if lead == const:
                continue
            ratio = lead // const
            if ratio == 0:
                continue
            for mid in range(-6, 7):
                if mid == 0:
                    continue
                poly = [0] * 12
                poly[0] = const
                poly[11] = lead
                poly[1] = const
                poly[10] = lead
                poly[2] = mid
                poly[9] = mid * ratio
                poly[3] = mid
                poly[8] = mid * ratio
                poly[4] = mid
                poly[7] = mid * ratio
                poly[5] = mid
                poly[6] = mid * ratio
                if all(-64 <= x <= 64 for x in poly) and all(x != 0 for x in poly):
                    try_poly(poly)

    # 4. Try: p(x) = 1 + x + ... + x^{k-1} - x^k - ... - x^{2k-1} type patterns
    for k in range(2, 7):
        for sign in [1, -1]:
            poly = [0] * 12
            for i in range(min(k, 12)):
                poly[i] = sign
            for i in range(k, min(2*k, 12)):
                poly[i] = -sign
            # Fill remaining with small values to ensure 12 nonzero
            for i in range(2*k, 12):
                if poly[i] == 0:
                    poly[i] = sign * (1 + (i % 3))
            if sum(1 for c in poly if c != 0) == 12:
                try_poly(poly)

    # 5. Try: p(x) where coefficients are powers of 2 with signs
    for signs in product([1, -1], repeat=12):
        if time.time() > deadline:
            break
        poly = [signs[i] * (2 ** (i % 4)) for i in range(12)]
        if all(-64 <= c <= 64 for c in poly):
            try_poly(poly)

    # 6. Try: p(x) = q(x) * r(x) where q and r are sparse
    for coeff_set in [
        [-3, -2, -1, 1, 2, 3],
        [-4, -3, -2, -1, 1, 2, 3, 4],
        [-2, -1, 1, 2],
        [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
    ]:
        if time.time() > deadline:
            break
        for _ in range(5000):
            if time.time() > deadline:
                break
            poly = [random.choice(coeff_set) for _ in range(12)]
            try_poly(poly)

    # 7. Try polynomials with few distinct coefficient values
    for distinct_vals in [
        [-1, 1],
        [-2, 1, 2],
        [-3, 1, 2, 3],
        [-1, 1, 2],
        [-2, 1, 3],
        [-3, 2, 3],
    ]:
        if time.time() > deadline:
            break
        for _ in range(3000):
            if time.time() > deadline:
                break
            poly = [random.choice(distinct_vals) for _ in range(12)]
            try_poly(poly)

    # Now do targeted local search with fast incremental evaluation
    random.seed(42)

    def fast_square(coeffs):
        """Compute square using direct loop - faster than np.convolve for small n."""
        n = len(coeffs)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            ci = coeffs[i]
            if ci == 0:
                continue
            for j in range(n):
                sq[i + j] += ci * coeffs[j]
        return sq

    def count_nz(sq):
        """Count nonzeros in square list."""
        return sum(1 for x in sq if x != 0)

    def local_search(start, max_iters=3000):
        nonlocal best_score, best_poly
        curr = list(start)
        curr_sq = fast_square(curr)
        curr_score = count_nz(curr_sq)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False

            # Try changing one coefficient to a value that might create cancellation
            pos = random.randint(0, 11)
            old_val = curr[pos]

            # Try values that are related to other coefficients (for cancellation)
            candidate_vals = set()
            for other_pos in range(12):
                if other_pos != pos:
                    v = curr[other_pos]
                    candidate_vals.add(v)
                    candidate_vals.add(-v)
                    candidate_vals.add(2 * v)
                    candidate_vals.add(-2 * v)
                    candidate_vals.add(3 * v)
                    candidate_vals.add(-3 * v)

            # Add some random values
            for _ in range(8):
                candidate_vals.add(random.randint(-64, 64))

            # Filter valid values
            candidate_vals = [v for v in candidate_vals if -64 <= v <= 64 and v != 0 and v != old_val]

            if candidate_vals:
                new_val = random.choice(candidate_vals)
                curr[pos] = new_val
                new_sq = fast_square(curr)
                new_score = count_nz(new_sq)
                if new_score < curr_score:
                    curr_score = new_score
                    curr_sq = new_sq
                    improved = True
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = list(curr)
                else:
                    curr[pos] = old_val

            # Try swapping two coefficients
            if random.random() < 0.25:
                i, j = random.sample(range(12), 2)
                a, b = curr[i], curr[j]
                curr[i], curr[j] = b, a
                new_sq = fast_square(curr)
                new_score = count_nz(new_sq)
                if new_score < curr_score:
                    curr_score = new_score
                    curr_sq = new_sq
                    improved = True
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = list(curr)
                else:
                    curr[i], curr[j] = a, b

            # Try negating a coefficient
            if random.random() < 0.15:
                pos = random.randint(0, 11)
                old_val = curr[pos]
                new_val = -old_val
                if new_val != 0:
                    curr[pos] = new_val
                    new_sq = fast_square(curr)
                    new_score = count_nz(new_sq)
                    if new_score < curr_score:
                        curr_score = new_score
                        curr_sq = new_sq
                        improved = True
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = list(curr)
                    else:
                        curr[pos] = old_val

            # Try making polynomial more symmetric
            if random.random() < 0.1:
                i, j = 11 - random.randint(0, 5), random.randint(0, 5)
                if i != j:
                    a, b = curr[i], curr[j]
                    curr[i] = b
                    curr[j] = a
                    new_sq = fast_square(curr)
                    new_score = count_nz(new_sq)
                    if new_score < curr_score:
                        curr_score = new_score
                        curr_sq = new_sq
                        improved = True
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = list(curr)
                    else:
                        curr[i], curr[j] = a, b

            if not improved and random.random() < 0.05:
                # Occasional larger perturbation
                for _ in range(random.randint(1, 3)):
                    p = random.randint(0, 11)
                    curr[p] = random.randint(-16, 16)
                    if curr[p] == 0:
                        curr[p] = 1
                curr_sq = fast_square(curr)
                curr_score = count_nz(curr_sq)

        return curr, curr_score

    # Multiple restarts with diverse starting points
    for restart in range(800):
        if time.time() > deadline:
            break

        if restart < 15:
            start = list(best_poly)
        elif random.random() < 0.4:
            start = list(best_poly)
            for _ in range(random.randint(1, 4)):
                p = random.randint(0, 11)
                start[p] = max(-64, min(64, start[p] + random.randint(-10, 10)))
                if start[p] == 0:
                    start[p] = random.choice([-1, 1])
        else:
            # Random start with structure
            pattern_type = random.randint(0, 5)
            if pattern_type == 0:
                start = [random.choice([-1, 1]) * random.randint(1, 6) for _ in range(12)]
            elif pattern_type == 1:
                base = random.randint(1, 4)
                start = [base * random.choice([-1, 1]) * (1 + (i % 3)) for i in range(12)]
            elif pattern_type == 2:
                start = [random.randint(-4, 4) for _ in range(12)]
                for i in range(12):
                    if start[i] == 0:
                        start[i] = random.choice([-1, 1])
            elif pattern_type == 3:
                start = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            elif pattern_type == 4:
                # Palindromic random
                half = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(6)]
                start = half + half[::-1]
                for i in range(12):
                    if start[i] == 0:
                        start[i] = random.choice([-1, 1])
            else:
                # Alternating with random magnitudes
                start = [random.choice([-1, 1]) * (1 + (i % 2)) * random.randint(1, 4) for i in range(12)]
                for i in range(12):
                    if start[i] == 0:
                        start[i] = 1

        result, score = local_search(start, max_iters=1200)
        if score < best_score:
            best_score = score
            best_poly = result

    # Final refinement: exhaustive single-coordinate search
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_sq = fast_square(best_poly)
                new_score = count_nz(new_sq)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val

    # Second pass refinement
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_sq = fast_square(best_poly)
                new_score = count_nz(new_sq)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val

    return best_poly
