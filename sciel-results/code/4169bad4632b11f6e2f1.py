"""
Sparse squares of twelve-term integer polynomials.

Goal: find a 12-term integer polynomial P (coefficients in [-64,64], degree <= 40)
whose EXACT integer square P^2 has the minimum number of nonzero terms.

Key insight: if P = Q^2 where Q is a sparse polynomial, then P^2 = Q^4.
So we want Q as sparse as possible to make Q^4 have few terms.

If Q has m terms, Q^2 has at most m^2 terms (before cancellation), and Q^4 has at most (m^2)^2 terms
but with more cancellation. Actually Q^4 = (Q^2)^2, so it's the square of a polynomial with up to m^2 terms.

Better approach: search for P directly. We know P must have exactly 12 nonzero terms.
We want to minimize the number of nonzero terms in P^2.

A systematic approach:
1. Try structured families where P = sum c_i * x^{e_i} with specific exponent patterns.
2. For each family, optimize the coefficients c_i to maximize cancellations in P^2.
3. Use random search over coefficient assignments.

The number of terms in P^2 for P = sum_{i=0}^{11} c_i x^{e_i} is the number of distinct values
of e_i + e_j (for i <= j) such that the sum of products is nonzero.

We can try small integer coefficients (like ±1, ±2) and specific exponent patterns
that are known to produce cancellations (e.g., geometric progressions, arithmetic progressions).

Key known constructions:
- If P = (1 + x + x^2 + ... + x^{k-1})^2 type structures...
- If P has terms at positions forming an arithmetic progression with specific coefficients.

Let's try:
1. P with exponents in arithmetic progression: e_i = i*d
2. P with exponents in geometric-like patterns
3. Random searches with local optimization
"""

import time
import random
from itertools import product

def _count_nonzero_in_square(coeffs):
    """Given coefficient list (ascending degree), return count of nonzero terms in square."""
    n = len(coeffs)
    if n == 0:
        return 0
    # Find nonzero positions
    nz = [(i, coeffs[i]) for i in range(n) if coeffs[i] != 0]
    if not nz:
        return 0
    # Compute square coefficients
    # Max degree is 2*(n-1)
    square = [0] * (2 * n - 1)
    for i, ci in nz:
        for j, cj in nz:
            square[i + j] += ci * cj
    # Count nonzero
    return sum(1 for v in square if v != 0)

def _validate(coeffs):
    """Check if coeffs is a valid candidate."""
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        # Trim trailing zeros
        while len(coeffs) > 0 and coeffs[-1] == 0:
            coeffs = coeffs[:-1]
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    # Check entries in [-64, 64]
    for c in coeffs:
        if not isinstance(c, int) or c < -64 or c > 64:
            return False
    # Check exactly 12 nonzero terms
    nz_count = sum(1 for c in coeffs if c != 0)
    if nz_count != 12:
        return False
    return True

def _to_output(coeffs):
    """Convert to proper output format (trim trailing zeros, ensure int)."""
    while len(coeffs) > 0 and coeffs[-1] == 0:
        coeffs = coeffs[:-1]
    return [int(c) for c in coeffs]

def _build_poly(exponents, coefficients):
    """Build coefficient list from exponents and coefficients."""
    if not exponents:
        return []
    max_deg = max(exponents)
    poly = [0] * (max_deg + 1)
    for exp, coeff in zip(exponents, coefficients):
        if 0 <= exp <= max_deg:
            poly[exp] = coeff
    return poly

def _try_pattern(exponents, coeff_values):
    """Try all assignments of coeff_values to exponents, return best score and coeffs."""
    best_score = 999999
    best_coeffs = None
    n = len(exponents)
    if n == 0:
        return best_score, best_coeffs
    
    # Try all combinations if small, otherwise sample
    if n <= 8:
        for combo in product(coeff_values, repeat=n):
            poly = _build_poly(exponents, list(combo))
            if not _validate(poly):
                continue
            s = _count_nonzero_in_square(poly)
            if s < best_score:
                best_score = s
                best_coeffs = poly[:]
    else:
        # Random sampling
        for _ in range(2000):
            combo = [random.choice(coeff_values) for _ in range(n)]
            poly = _build_poly(exponents, list(combo))
            if not _validate(poly):
                continue
            s = _count_nonzero_in_square(poly)
            if s < best_score:
                best_score = s
                best_coeffs = poly[:]
    
    return best_score, best_coeffs

def construct():
    deadline = time.time() + 85  # Leave 5s margin for 90s limit
    
    best_score = 999999
    best_coeffs = None
    
    # Strategy 1: Try structured exponent patterns
    # Pattern A: Arithmetic progression
    for d in range(1, 10):
        if time.time() > deadline - 10:
            break
        exponents = [i * d for i in range(12)]
        if max(exponents) > 40:
            continue
        coeff_values = [1, -1]
        s, c = _try_pattern(exponents, coeff_values)
        if s < best_score:
            best_score = s
            best_coeffs = c
    
    # Pattern B: Symmetric exponents (palindromic)
    for d in range(1, 5):
        if time.time() > deadline - 10:
            break
        # e_0=0, e_1=d, e_2=2d, ..., e_11=11d but symmetric
        exponents = [i * d for i in range(12)]
        if max(exponents) > 40:
            continue
        coeff_values = [1, -1]
        s, c = _try_pattern(exponents, coeff_values)
        if s < best_score:
            best_score = s
            best_coeffs = c
    
    # Pattern C: Grouped exponents
    # 6 terms at even positions, 6 at odd
    if time.time() < deadline - 10:
        exponents = [2*i for i in range(6)] + [2*i+1 for i in range(6)]
        if max(exponents) <= 40:
            coeff_values = [1, -1]
            s, c = _try_pattern(exponents, coeff_values)
            if s < best_score:
                best_score = s
                best_coeffs = c
    
    # Pattern D: Sparse with gaps
    gap_patterns = [
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
    ]
    for exps in gap_patterns:
        if time.time() > deadline - 10:
            break
        if max(exps) > 40:
            continue
        coeff_values = [1, -1]
        s, c = _try_pattern(exps, coeff_values)
        if s < best_score:
            best_score = s
            best_coeffs = c
    
    # Strategy 2: Random search with local improvement
    def random_poly():
        # Generate random 12-term polynomial
        exponents = random.sample(range(41), 12)
        exponents.sort()
        coefficients = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        return _build_poly(exponents, coefficients)
    
    def local_search(coeffs, max_iters=200):
        if not _validate(coeffs):
            return coeffs, 999999
        current_score = _count_nonzero_in_square(coeffs)
        n = len(coeffs)
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try flipping a coefficient sign
            for i in range(n):
                if coeffs[i] == 0:
                    continue
                trial = coeffs[:]
                trial[i] = -trial[i]
                if not _validate(trial):
                    continue
                s = _count_nonzero_in_square(trial)
                if s < current_score:
                    current = trial
                    current_score = s
                    improved = True
                    break
            # Try changing coefficient value
            if not improved:
                for i in range(n):
                    if coeffs[i] == 0:
                        continue
                    for new_val in [-2, -1, 1, 2]:
                        if new_val == coeffs[i]:
                            continue
                        trial = coeffs[:]
                        trial[i] = new_val
                        if not _validate(trial):
                            continue
                        s = _count_nonzero_in_square(trial)
                        if s < current_score:
                            current = trial
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
            # Try moving a term
            if not improved:
                for i in range(n):
                    if coeffs[i] == 0:
                        continue
                    for new_exp in range(41):
                        if coeffs[new_exp] != 0:
                            continue
                        trial = coeffs[:]
                        trial[new_exp] = trial[i]
                        trial[i] = 0
                        # Trim
                        while len(trial) > 0 and trial[-1] == 0:
                            trial = trial[:-1]
                        if not _validate(trial):
                            continue
                        s = _count_nonzero_in_square(trial)
                        if s < current_score:
                            current = trial
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
            coeffs = current
        return coeffs, current_score
    
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        coeffs = random_poly()
        coeffs, score = local_search(coeffs, max_iters=150)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs[:]
    
    # Strategy 3: Try specific known good patterns
    # P = (1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^10 - x^11)
    # This has 12 terms
    if time.time() < deadline - 3:
        for pattern in [
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],  # alternating
            [1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1],  # block
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],  # double alternation
            [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],  # triple pattern
        ]:
            coeffs = pattern[:]
            if _validate(coeffs):
                s = _count_nonzero_in_square(coeffs)
                if s < best_score:
                    best_score = s
                    best_coeffs = coeffs[:]
    
    # Return best found
    if best_coeffs is not None and _validate(best_coeffs):
        return _to_output(best_coeffs)
    
    # Fallback: simple 12-term polynomial
    return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
