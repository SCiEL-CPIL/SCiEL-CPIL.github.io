# EVOLVE-BLOCK-START
import time
import random
import math

def _square_count(coeffs):
    """coeffs: list of (coef, exponent). Return number of nonzero terms in the square."""
    res = {}
    n = len(coeffs)
    for i in range(n):
        c1, e1 = coeffs[i]
        for j in range(i, n):
            e_sum = e1 + coeffs[j][1]
            val = c1 * coeffs[j][0]
            if i != j:
                val *= 2
            res[e_sum] = res.get(e_sum, 0) + val
    return sum(1 for v in res.values() if v != 0)

def _to_list(coeffs):
    """Convert [(coef, exp), ...] to a coefficient list in ascending degree, length<=41,
    nonzero final coefficient, exactly 12 nonzero entries, entries in [-64,64].
    Returns None if invalid."""
    if len(coeffs) != 12:
        return None
    for c, e in coeffs:
        if c == 0 or e < 0 or e > 40 or abs(c) > 64:
            return None
    max_deg = max(e for _, e in coeffs)
    if max_deg > 40:
        return None
    lst = [0] * (max_deg + 1)
    for c, e in coeffs:
        lst[e] = c
    # Trim trailing zeros (shouldn't be any since max_deg is from a nonzero term)
    while len(lst) > 1 and lst[-1] == 0:
        lst.pop()
    if len(lst) > 41:
        return None
    if lst[-1] == 0:
        return None
    if sum(1 for x in lst if x != 0) != 12:
        return None
    return lst

def _make_polynomial(exps, coefs):
    return list(zip(coefs, exps))

def _try_identity(name, exps, coefs):
    """Build a 12-term polynomial from a structured identity and return (score, coeffs)."""
    if len(exps) != 12 or len(coefs) != 12:
        return None
    coeffs = _make_polynomial(exps, coefs)
    sc = _square_count(coeffs)
    return (sc, coeffs)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    best_score = 1000
    best_coeffs = None

    def consider(sc, coeffs):
        nonlocal best_score, best_coeffs
        if sc < best_score:
            best_score = sc
            best_coeffs = coeffs

    # ---------- Phase 1: Known algebraic identities ----------
    # Cyclotomic and related polynomials with sparse squares.
    # We need exactly 12 nonzero terms. We can take products of known sparse polynomials.

    # Identity 1: (1+x+x^2)(1+x^3+x^6)(1+x^9) ... let's think differently.
    # Actually, let's use the fact that if P(x) = product of cyclotomic polynomials,
    # P(x)^2 can be very sparse.

    # Let's try: P(x) = (1+x)(1+x^2)(1+x^4)(1+x^8) ... that's 4 terms.
    # We need 12 terms.

    # Strategy: Use polynomials of the form sum of few terms with specific exponents
    # that are known to have sparse autocorrelation.

    # Known: (1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11)
    # has square with 23 nonzero terms. Not great.

    # Better: Use sign patterns. (1 - x + x^2 - x^3 + ... ) 
    # Let's try alternating signs on consecutive exponents.

    # Try: exponents = 0..11, alternating signs
    for base_exps in [list(range(12))]:
        for sign_pattern in [[1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                             [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                             [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]]:
            sc = _square_count(list(zip(sign_pattern, base_exps)))
            consider(sc, list(zip(sign_pattern, base_exps)))

    # Try: exponents with gaps, e.g., 0,1,2,4,5,6,8,9,10,12,13,14
    gap_exps = [0,1,2,4,5,6,8,9,10,12,13,14]
    for signs in [[1]*12, [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]]:
        sc = _square_count(list(zip(signs, gap_exps)))
        consider(sc, list(zip(signs, gap_exps)))

    # ---------- Phase 2: Local search from good starting points ----------
    def local_search(start_coeffs, max_iters=5000):
        """Perturb coefficients and exponents to reduce square count."""
        nonlocal best_score, best_coeffs
        coeffs = [list(c) for c in start_coeffs]  # mutable
        current_score = _square_count(coeffs)
        consider(current_score, [tuple(c) for c in coeffs])
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            # Pick a random term to modify
            idx = random.randint(0, 11)
            c, e = coeffs[idx]
            
            # Try modifying coefficient
            new_c = c + random.choice([-2, -1, 1, 2])
            if abs(new_c) <= 64 and new_c != 0:
                old = (c, e)
                coeffs[idx] = (new_c, e)
                new_score = _square_count(coeffs)
                if new_score < current_score:
                    current_score = new_score
                    consider(new_score, [tuple(x) for x in coeffs])
                else:
                    coeffs[idx] = old
            
            # Try modifying exponent
            new_e = e + random.choice([-3, -2, -1, 1, 2, 3])
            if 0 <= new_e <= 40:
                old = (c, e)
                coeffs[idx] = (c, new_e)
                new_score = _square_count(coeffs)
                if new_score < current_score:
                    current_score = new_score
                    consider(new_score, [tuple(x) for x in coeffs])
                else:
                    coeffs[idx] = old

    # Start local search from several good initial points
    if best_coeffs is not None:
        local_search(best_coeffs, 3000)

    # More starting points
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
        # Random 12-term polynomial
        exps = sorted(random.sample(range(0, 41), 12))
        coefs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        coeffs = list(zip(coefs, exps))
        sc = _square_count(coeffs)
        consider(sc, coeffs)
        if sc < 15:
            local_search(coeffs, 2000)

    # ---------- Phase 3: Try specific known good structures ----------
    # Fibonacci polynomial identities
    # (1 + x)^12 has 13 terms but coefficients are large.
    # Let's try: exponents that are multiples of 3: 0,3,6,9,12,15,18,21,24,27,30,33
    for exps in [list(range(0, 36, 3)), list(range(0, 24, 2)), [0,1,3,4,6,7,9,10,12,13,15,16]]:
        for signs in [[1]*12, [1,-1]*6, [1,1,-1,-1]*3]:
            if len(exps) == 12 and len(signs) == 12:
                sc = _square_count(list(zip(signs, exps)))
                consider(sc, list(zip(signs, exps)))
                if sc < 15 and time.time() < deadline - 10.0:
                    local_search(list(zip(signs, exps)), 2000)

    # ---------- Phase 4: Final local search refinement ----------
    if best_coeffs is not None and time.time() < deadline - 5.0:
        local_search(best_coeffs, 3000)

    # ---------- Build result ----------
    if best_coeffs is None:
        # Fallback: 12 consecutive terms with alternating signs
        return [1 if i % 2 == 0 else -1 for i in range(12)]

    result = _to_list(best_coeffs)
    if result is None:
        # Fallback
        return [1 if i % 2 == 0 else -1 for i in range(12)]

    return result

# EVOLVE-BLOCK-END
