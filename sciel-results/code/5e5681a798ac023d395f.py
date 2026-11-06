import time
import random


def _square(coeffs):
    """Compute exact integer square of polynomial given as coefficient list (ascending degree)."""
    n = len(coeffs)
    result = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            cj = coeffs[j]
            if cj == 0:
                continue
            result[i + j] += ci * cj
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _nonzero_count(coeffs):
    return sum(1 for c in coeffs if c != 0)


def _is_valid(coeffs):
    """Check all constraints."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    if _nonzero_count(coeffs) != 12:
        return False
    return True


def _score(coeffs):
    """Lower is better: number of nonzero terms in the square."""
    return _nonzero_count(_square(coeffs))


def _trim(poly):
    """Return a trimmed copy of poly (drop trailing zeros, keep at least length 1)."""
    p = poly[:]
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def _local_improve(poly, best_score, deadline):
    """Local improvement: sign flips and magnitude changes. Returns (poly, score)."""
    poly = poly[:]
    local_best = _score(poly)
    if local_best >= best_score:
        return poly, local_best

    candidates = [-64, -8, -4, -3, -2, -1, 1, 2, 3, 4, 8, 64]

    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        nz_indices = [i for i, c in enumerate(poly) if c != 0]

        # 1. Flip sign of each term
        for idx in nz_indices:
            old = poly[idx]
            poly[idx] = -old
            s = _score(poly)
            if s < local_best:
                local_best = s
                improved = True
            else:
                poly[idx] = old

        # 2. Change coefficient magnitude
        for idx in nz_indices:
            old = poly[idx]
            for new_val in candidates:
                if new_val == old:
                    continue
                poly[idx] = new_val
                s = _score(poly)
                if s < local_best:
                    local_best = s
                    improved = True
                else:
                    poly[idx] = old

        if not improved:
            break

    return poly, local_best


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)

    best_poly = None
    best_score = float('inf')

    def update_best(poly):
        nonlocal best_poly, best_score
        if _is_valid(poly):
            s = _score(poly)
            if s < best_score:
                best_score = s
                best_poly = poly[:]

    random.seed(42)

    # --- Structured candidates ---
    # Family A: consecutive with various sign patterns
    for sign_pattern in range(64):
        coeffs = [1 if (sign_pattern >> i) & 1 else -1 for i in range(12)]
        update_best(coeffs)

    # Family B: geometric / mixed exponent sets
    exponent_sets = [
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
    ]
    for exps in exponent_sets:
        if len(exps) != 12 or exps[-1] >= 41:
            continue
        coeffs = [0] * (exps[-1] + 1)
        for e in exps:
            coeffs[e] = 1
        update_best(coeffs)

    # Family C: symmetric centered patterns
    for center in range(5, 35):
        for d in [1, 2, 3]:
            poly = [0] * 41
            for i in range(12):
                deg = center - 5 + i * d
                if 0 <= deg < 41:
                    poly[deg] = 1
            trimmed = _trim(poly)
            if _is_valid(trimmed):
                update_best(trimmed)

    # --- Exhaustive sign-pattern enumeration over structured exponent sets ---
    for exps in exponent_sets:
        if len(exps) != 12 or exps[-1] >= 41:
            continue
        for sign_mask in range(4096):  # 2^12
            if time.time() > deadline:
                return _trim(best_poly) if best_poly is not None else [1] * 12
            coeffs = [0] * (exps[-1] + 1)
            for i, e in enumerate(exps):
                coeffs[e] = 1 if (sign_mask >> i) & 1 else -1
            if _is_valid(coeffs):
                s = _score(coeffs)
                if s < best_score:
                    best_score = s
                    best_poly = coeffs[:]

    # --- High-throughput random search ---
    coeff_choices = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    n_tries = 0
    while time.time() < deadline:
        n_tries += 1

        # Choose 12 distinct exponents from 0..40
        exponents = random.sample(range(41), 12)
        exponents.sort()
        coeffs = [0] * (exponents[-1] + 1)
        for e in exponents:
            coeffs[e] = random.choice(coeff_choices)

        # Ensure the last exponent's coefficient is nonzero
        if coeffs[exponents[-1]] == 0:
            coeffs[exponents[-1]] = random.choice([-2, -1, 1, 2, 3, -3, 4, -4])

        trimmed = _trim(coeffs)
        if not _is_valid(trimmed):
            continue

        s = _score(trimmed)
        if s < best_score:
            best_score = s
            best_poly = trimmed[:]

            # Local improvement on the new best
            if time.time() < deadline - 5:
                improved, improved_s = _local_improve(trimmed, best_score, deadline)
                if _is_valid(improved) and improved_s < best_score:
                    best_score = improved_s
                    best_poly = improved[:]

        # Occasionally do local improvement on near-best candidates
        elif s <= best_score + 2 and n_tries % 50 == 0:
            if time.time() < deadline - 5:
                improved, improved_s = _local_improve(trimmed, best_score, deadline)
                if _is_valid(improved) and improved_s < best_score:
                    best_score = improved_s
                    best_poly = improved[:]

    if best_poly is not None:
        return _trim(best_poly)

    # Fallback
    return [1] * 12
