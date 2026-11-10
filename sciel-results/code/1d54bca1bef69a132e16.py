import numpy as np
import time
import random
from itertools import combinations, product

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
        if coeffs[11] == 0:  # nonzero final coefficient
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Try baseline
    try_poly([1] * 12)

    # ============================================================
    # PHASE 1: Structured pattern generation
    # ============================================================

    # Try polynomials with symmetry (palindromic, anti-palindromic)
    for sign in [1, -1]:
        for i in range(6):
            for j in range(i, 6):
                # Build palindromic: c[i] = sign * c[11-i]
                pass

    # Generate palindromic polynomials: c[i] = c[11-i]
    for a in range(-4, 5):
        for b in range(-4, 5):
            for c in range(-4, 5):
                for d in range(-4, 5):
                    for e in range(-4, 5):
                        for f in range(-4, 5):
                            if a == 0 or b == 0 or c == 0 or d == 0 or e == 0 or f == 0:
                                continue
                            poly = [a, b, c, d, e, f, f, e, d, c, b, a]
                            if all(-64 <= x <= 64 for x in poly):
                                try_poly(poly)

    # Anti-palindromic: c[i] = -c[11-i]
    for a in range(-4, 5):
        for b in range(-4, 5):
            for c in range(-4, 5):
                for d in range(-4, 5):
                    for e in range(-4, 5):
                        for f in range(-4, 5):
                            if a == 0 or b == 0 or c == 0 or d == 0 or e == 0 or f == 0:
                                continue
                            poly = [a, b, c, d, e, f, -f, -e, -d, -c, -b, -a]
                            if all(-64 <= x <= 64 for x in poly):
                                try_poly(poly)

    # Periodic patterns
    for period in [2, 3, 4, 6]:
        for base in range(-3, 4):
            for amp in range(1, 4):
                for offset in range(period):
                    pattern = []
                    for i in range(12):
                        val = base + amp * ((i + offset) % period)
                        if val == 0:
                            val = 1
                        pattern.append(val)
                    if all(-64 <= c <= 64 for c in pattern):
                        try_poly(pattern)

    # Sign patterns with small magnitudes
    for mag_set in [[1, 1, 1], [1, 2, 1], [1, 1, 2], [2, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 1]]:
        for signs in product([1, -1], repeat=12):
            if time.time() > deadline:
                break
            poly = [signs[i] * mag_set[i % 3] for i in range(12)]
            if all(-64 <= c <= 64 for c in poly):
                try_poly(poly)

    # ============================================================
    # PHASE 2: Constraint-based search
    # Target specific zero patterns in the square
    # ============================================================

    # The square has 23 coefficients. We want to zero out as many as possible.
    # Square coefficient at degree k: sum_{i+j=k} c[i]*c[j]
    # For k=0: c[0]^2 (always nonzero if c[0]!=0)
    # For k=22: c[11]^2 (always nonzero if c[11]!=0)
    # So we can potentially zero degrees 1 through 21 (21 coefficients)

    # Try to zero specific subsets of middle coefficients
    # Start with trying to zero consecutive middle coefficients

    def solve_constraints(target_zeros, start_poly=None):
        """Try to find coefficients where square has zeros at target_zeros degrees."""
        if start_poly is None:
            start_poly = [1] * 12

        curr = list(start_poly)
        curr_score = evaluate(curr)

        # Iteratively adjust to zero out target coefficients
        for iteration in range(50):
            if time.time() > deadline:
                break

            # Find which target zeros are still nonzero
            sq = poly_square(curr)
            still_nonzero = [z for z in target_zeros if abs(sq[z]) > 0]

            if not still_nonzero:
                break

            # Try to fix the first nonzero target
            target_deg = still_nonzero[0]
            old_val = None
            improved = False

            # Try changing each coefficient
            for pos in range(12):
                if time.time() > deadline:
                    break
                old_val = curr[pos]
                # Try values that might help cancel
                # The coefficient at degree target_deg involves pairs (i,j) with i+j=target_deg
                # Changing c[pos] affects all degrees involving pos
                for new_val in range(-16, 17):
                    if new_val == old_val or new_val == 0:
                        continue
                    if not (-64 <= new_val <= 64):
                        continue
                    curr[pos] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        break
                    else:
                        curr[pos] = old_val

                if improved:
                    break
                curr[pos] = old_val

            if not improved:
                # Try random perturbation
                pos = random.randint(0, 11)
                curr[pos] = random.choice([-1, 1, -2, 2, -3, 3])
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                else:
                    curr[pos] = random.randint(-8, 8)
                    if curr[pos] == 0:
                        curr[pos] = 1
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score

        return curr, curr_score

    # Try to zero out various patterns
    zero_patterns = [
        list(range(1, 5)),      # zero degrees 1-4
        list(range(1, 6)),      # zero degrees 1-5
        list(range(2, 6)),      # zero degrees 2-5
        list(range(1, 8)),      # zero degrees 1-7
        list(range(1, 12)),     # zero degrees 1-11
        list(range(1, 16)),     # zero degrees 1-15
        list(range(1, 22)),     # zero degrees 1-21
        list(range(2, 10)),     # zero degrees 2-9
        list(range(3, 12)),     # zero degrees 3-11
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    ]

    for pattern in zero_patterns:
        if time.time() > deadline:
            break
        for restart in range(50):
            if time.time() > deadline:
                break
            if restart == 0:
                start = list(best_poly)
            elif restart < 10:
                start = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
                start[11] = random.choice([-3, -2, -1, 1, 2, 3])
            else:
                start = [random.randint(-8, 8) for _ in range(12)]
                for i in range(12):
                    if start[i] == 0:
                        start[i] = random.choice([-1, 1])

            result, score = solve_constraints(pattern, start)
            if score < best_score:
                best_score = score
                best_poly = result

    # ============================================================
    # PHASE 3: Smart local search with incremental updates
    # ============================================================

    random.seed(42)

    def local_search(start, max_iters=5000):
        curr = list(start)
        curr_score = evaluate(curr)
        curr_sq = poly_square(curr)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False

            # Strategy 1: Change one coefficient, recompute square
            pos = random.randint(0, 11)
            old_val = curr[pos]

            # Generate candidate values focused on creating cancellations
            candidate_vals = set()

            # Values that relate to existing coefficients
            for other in range(12):
                if other != pos:
                    candidate_vals.add(curr[other])
                    candidate_vals.add(-curr[other])
                    candidate_vals.add(2 * curr[other])
                    candidate_vals.add(-2 * curr[other])
                    candidate_vals.add(3 * curr[other])
                    candidate_vals.add(-3 * curr[other])

            # Small values
            for v in range(-16, 17):
                if v != 0:
                    candidate_vals.add(v)

            # Filter
            candidate_vals = [v for v in candidate_vals if -64 <= v <= 64 and v != 0 and v != old_val]

            if candidate_vals:
                # Try a subset (random sample if too many)
                if len(candidate_vals) > 30:
                    candidate_vals = random.sample(candidate_vals, 30)

                for new_val in candidate_vals:
                    curr[pos] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        try_poly(curr)
                        break
                    else:
                        curr[pos] = old_val

            # Strategy 2: Swap two coefficients
            if random.random() < 0.15:
                i, j = random.sample(range(12), 2)
                a, b = curr[i], curr[j]
                curr[i], curr[j] = b, a
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    try_poly(curr)
                else:
                    curr[i], curr[j] = a, b

            # Strategy 3: Negate a coefficient
            if random.random() < 0.1:
                pos = random.randint(0, 11)
                old_val = curr[pos]
                new_val = -old_val
                if new_val != 0:
                    curr[pos] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        try_poly(curr)
                    else:
                        curr[pos] = old_val

            # Strategy 4: Add/subtract small value to two coefficients
            if random.random() < 0.1:
                i, j = random.sample(range(12), 2)
                delta = random.choice([-2, -1, 1, 2])
                old_i, old_j = curr[i], curr[j]
                new_i = old_i + delta
                new_j = old_j - delta
                if -64 <= new_i <= 64 and -64 <= new_j <= 64 and new_i != 0 and new_j != 0:
                    curr[i], curr[j] = new_i, new_j
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        try_poly(curr)
                    else:
                        curr[i], curr[j] = old_i, old_j

            # Strategy 5: Scale a coefficient
            if random.random() < 0.05:
                pos = random.randint(0, 11)
                old_val = curr[pos]
                new_val = old_val * 2
                if -64 <= new_val <= 64 and new_val != 0:
                    curr[pos] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        try_poly(curr)
                    else:
                        curr[pos] = old_val

            if not improved and random.random() < 0.02:
                # Larger perturbation
                for _ in range(random.randint(1, 3)):
                    p = random.randint(0, 11)
                    curr[p] = random.randint(-16, 16)
                    if curr[p] == 0:
                        curr[p] = random.choice([-1, 1])

        return curr, curr_score

    # Multiple restarts
    for restart in range(200):
        if time.time() > deadline:
            break

        if restart < 20:
            start = list(best_poly)
        elif random.random() < 0.4:
            start = list(best_poly)
            for _ in range(random.randint(1, 3)):
                p = random.randint(0, 11)
                start[p] = max(-64, min(64, start[p] + random.randint(-10, 10)))
                if start[p] == 0:
                    start[p] = random.choice([-1, 1])
        else:
            # Structured random start
            ptype = random.randint(0, 4)
            if ptype == 0:
                # Palindromic
                half = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(6)]
                start = half + list(reversed(half))
            elif ptype == 1:
                # Anti-palindromic
                half = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(6)]
                start = half + [-x for x in reversed(half)]
            elif ptype == 2:
                # Periodic
                period = random.choice([2, 3, 4, 6])
                base_pat = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(period)]
                start = [base_pat[i % period] for i in range(12)]
            elif ptype == 3:
                # Small random
                start = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
            else:
                # Mixed
                start = [random.randint(-6, 6) for _ in range(12)]
                for i in range(12):
                    if start[i] == 0:
                        start[i] = random.choice([-1, 1])

        result, score = local_search(start, max_iters=3000)
        if score < best_score:
            best_score = score
            best_poly = result

    # ============================================================
    # PHASE 4: Final exhaustive refinement
    # ============================================================

    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-32, 33):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val

    # Second pass with wider range
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val

    return best_poly
