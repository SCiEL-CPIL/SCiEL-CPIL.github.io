import time
import math
import random
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 110.0  # safety margin before 150s wall-clock limit


def _compat(assign, m, r):
    """Check if residue r for modulus m is compatible with all already-placed moduli."""
    for m2, r2 in assign.items():
        g = math.gcd(m, m2)
        if r % g == r2 % g:
            return False
    return True


def _flexibility_score(assign, m, r):
    """
    Score how 'flexible' placing residue r on modulus m is.
    Lower flexibility = worse (blocks more future options).
    We count how many residues are still available for each unplaced modulus.
    """
    # For speed, we just count the number of compatible residues for a few key future moduli
    # A simpler heuristic: fewer valid residues remaining for small future moduli is worse
    score = 0
    # Check a subset of future moduli (those not yet placed and not too many)
    for m2 in range(20, 201):
        if m2 in assign or m2 == m:
            continue
        # Count how many residues of m2 are compatible with the current assignment PLUS this new placement
        # This is expensive, so we approximate: check compatibility of r with m2's residues
        g = math.gcd(m, m2)
        # The residue r % g blocks exactly one residue class mod g for m2
        # So m2 has (m2 - m2//g) valid residues after placing m:r, minus any already blocked
        # Actually, let's just do a quick check: is there at least one valid residue for m2?
        # For a better score, count valid residues for m2 given current assign + (m,r)
        valid_count = 0
        for r2 in range(m2):
            if _compat_with_extra(assign, m, r, m2, r2):
                valid_count += 1
        # Prefer placements that leave more options: penalty if valid_count is low
        if valid_count == 0:
            score -= 1000  # very bad, but shouldn't happen if _compat passed
        else:
            # Normalize: smaller m2 with fewer options is more critical
            score -= 10.0 / (valid_count + 1) * (200.0 / m2)
    return score


def _compat_with_extra(assign, m_new, r_new, m2, r2):
    """Check if r2 for m2 is compatible with assign plus the extra (m_new, r_new)."""
    # Check against the new placement
    g = math.gcd(m_new, m2)
    if r_new % g == r2 % g:
        return False
    # Check against existing
    for m3, r3 in assign.items():
        g2 = math.gcd(m2, m3)
        if r2 % g2 == r3 % g2:
            return False
    return True


def _pick_best_residue(assign, m):
    """Pick the residue for m that maximizes flexibility (minimizes blocking)."""
    best_r = None
    best_score = None
    for r in range(m):
        if _compat(assign, m, r):
            s = _flexibility_score(assign, m, r)
            if best_score is None or s > best_score:
                best_score = s
                best_r = r
    return best_r


def _greedy_construct(order):
    """
    Greedy construction: place moduli in the given order, picking best residue each time.
    Returns list of [m, r] pairs.
    """
    assign = {}
    for m in order:
        r = _pick_best_residue(assign, m)
        if r is not None:
            assign[m] = r
    return [[m, assign[m]] for m in sorted(assign.keys())]


def _score(classes):
    """Compute exact rational score, return as float."""
    if not classes:
        return 0.0
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)


def _validate(classes):
    """Verify pairwise disjointness."""
    n = len(classes)
    for i in range(n):
        m1, r1 = classes[i]
        for j in range(i + 1, n):
            m2, r2 = classes[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                return False
    return True


def construct():
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return a simple valid fallback
        return [[20 * j, j - 1] for j in range(1, 11)]

    # All moduli in [20, 200]
    all_moduli = list(range(20, 201))

    # Strategy 1: Greedy in descending order of modulus (larger moduli first, easier to place)
    # Strategy 2: Greedy in ascending order
    # Strategy 3: Randomized orders with multiple restarts

    best_classes = None
    best_score_val = 0.0

    # Deterministic attempts first
    orders = [
        sorted(all_moduli, reverse=True),  # descending
        sorted(all_moduli),                # ascending
    ]

    for order in orders:
        elapsed = time.time() - _START_TIME
        if elapsed > _TIME_LIMIT - 5:
            break
        classes = _greedy_construct(order)
        if classes and _validate(classes):
            s = _score(classes)
            if s > best_score_val:
                best_score_val = s
                best_classes = classes

    # Randomized restarts
    rng = random.Random(42)
    max_restarts = 200
    for restart in range(max_restarts):
        elapsed = time.time() - _START_TIME
        if elapsed > _TIME_LIMIT - 10:
            break
        # Random shuffle
        order = all_moduli[:]
        rng.shuffle(order)
        classes = _greedy_construct(order)
        if classes and _validate(classes):
            s = _score(classes)
            if s > best_score_val:
                best_score_val = s
                best_classes = classes

    if best_classes is not None:
        return best_classes

    # Fallback: simple valid construction
    return [[20 * j, j - 1] for j in range(1, 11)]
