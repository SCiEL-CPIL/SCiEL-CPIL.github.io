import time
import random
from itertools import combinations

_DEADLINE = time.time() + 80  # safety margin before 90s kill

def _square_nonzero_count(coeffs):
    """Given coefficient list (ascending degree), return number of nonzero terms in exact square."""
    n = len(coeffs)
    # Square via convolution
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ai = coeffs[i]
        if ai == 0:
            continue
        for j in range(n):
            aj = coeffs[j]
            if aj == 0:
                continue
            sq[i + j] += ai * aj
    return sum(1 for x in sq if x != 0)

def _valid_candidate(coeffs):
    """Check constraints: length<=41, nonzero final, entries in [-64,64], exactly 12 nonzero."""
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    nonzero = 0
    for c in coeffs:
        if c != 0:
            nonzero += 1
            if c < -64 or c > 64:
                return False
    if nonzero != 12:
        return False
    return True

def _make_candidate(exponents, coefficients):
    """Build coefficient list from exponents and coefficients."""
    if not exponents:
        return None
    max_deg = max(exponents)
    if max_deg >= 41:
        return None
    coeffs = [0] * (max_deg + 1)
    for e, c in zip(exponents, coefficients):
        if e < 0 or e > 40:
            return None
        coeffs[e] = c
    if coeffs[-1] == 0:
        return None
    return coeffs

def _random_exponents_12(max_deg=40):
    """Pick 12 distinct exponents in [0, max_deg]."""
    return sorted(random.sample(range(0, max_deg + 1), 12))

def _random_coefficients_12():
    """Pick 12 nonzero coefficients in [-64,64]."""
    return [random.choice([-1, 1]) * random.randint(1, 8) for _ in range(12)]

def construct():
    t0 = time.time()
    best_count = 78  # worst case: 12^2 = 144 self-pairs but max distinct is 12*13/2=78
    best_coeffs = None
    # Fallback: simple 12-term polynomial
    fallback = [0] * 12
    for i in range(12):
        fallback[i] = 1
    if _valid_candidate(fallback):
        best_coeffs = fallback
        best_count = _square_nonzero_count(fallback)

    rng = random.Random(12345)
    # Strategy: random exponent sets + small coefficients, evaluate square sparsity
    attempts = 0
    while time.time() - t0 < 75:  # leave 5s safety margin
        attempts += 1
        exponents = _random_exponents_12()
        # Use small coefficients to encourage cancellation
        coefficients = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        coeffs = _make_candidate(exponents, coefficients)
        if coeffs is None:
            continue
        if not _valid_candidate(coeffs):
            continue
        cnt = _square_nonzero_count(coeffs)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = coeffs
            # If we found something very good, do local search around it
            if best_count <= 30:
                # Local perturbation
                for _ in range(200):
                    if time.time() - t0 > 75:
                        break
                    e2 = list(exponents)
                    c2 = list(coefficients)
                    # Perturb one exponent
                    idx = rng.randint(0, 11)
                    if rng.random() < 0.5:
                        # change coefficient
                        c2[idx] = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                    else:
                        # shift an exponent
                        shift = rng.choice([-2, -1, 1, 2])
                        new_e = e2[idx] + shift
                        if 0 <= new_e <= 40 and new_e not in e2:
                            e2[idx] = new_e
                            e2.sort()
                    coeffs2 = _make_candidate(e2, c2)
                    if coeffs2 is None or not _valid_candidate(coeffs2):
                        continue
                    cnt2 = _square_nonzero_count(coeffs2)
                    if cnt2 < best_count:
                        best_count = cnt2
                        best_coeffs = coeffs2
                        exponents = e2
                        coefficients = c2
        # Also try structured: palindromic or symmetric exponents
        if attempts % 10 == 0:
            # Symmetric exponent set around center
            center = rng.randint(10, 20)
            offsets = sorted(rng.sample(range(1, 25), 6))
            exponents_sym = sorted([center - o for o in offsets] + [center + o for o in offsets])
            if len(set(exponents_sym)) == 12 and 0 <= min(exponents_sym) and max(exponents_sym) <= 40:
                # Assign coefficients symmetrically
                coeffs_sym = [0] * (max(exponents_sym) + 1)
                for i in range(6):
                    c = rng.choice([-2, -1, 1, 2])
                    coeffs_sym[exponents_sym[i]] = c
                    coeffs_sym[exponents_sym[i + 6]] = c
                if _valid_candidate(coeffs_sym):
                    cnt = _square_nonzero_count(coeffs_sym)
                    if cnt < best_count:
                        best_count = cnt
                        best_coeffs = coeffs_sym

    # Ensure we return something valid
    if best_coeffs is None or not _valid_candidate(best_coeffs):
        best_coeffs = [1] * 12
    return best_coeffs

# EVOLVE-BLOCK-END
