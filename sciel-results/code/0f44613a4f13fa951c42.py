"""Rewritten for the suite task: modular collision + exact meet-in-the-middle polish."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _local_refine(coefs, deadline, iters=200000):
    """Greedy 1- and 2-coordinate moves minimizing |sum c_i/i|, keeping sum nonzero."""
    n = len(coefs)
    c = list(coefs)
    s = Fraction(0, 1)
    for i, v in enumerate(c, start=1):
        if v:
            s += Fraction(v, i)
    if s == 0:
        return None
    best_abs = abs(s.numerator) * s.denominator  # proxy not used; use Fraction compare
    best_abs = abs(s)
    t0 = time.time()
    it = 0
    improved = True
    while improved and it < iters and time.time() < deadline:
        improved = False
        it += 1
        # 1-coordinate moves
        for i in range(n):
            if s == 0:
                return None
            v = c[i]
            for nv in (-1, 0, 1):
                if nv == v:
                    continue
                s2 = s + Fraction(nv - v, i + 1)
                if s2 == 0:
                    continue
                if abs(s2) < best_abs:
                    c[i] = nv
                    s = s2
                    best_abs = abs(s2)
                    improved = True
        if improved:
            continue
        # 2-coordinate moves (only if time remains)
        if time.time() > deadline:
            break
        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = c[i], c[j]
                base = s
                for ni in (-1, 0, 1):
                    if ni == vi:
                        continue
                    for nj in (-1, 0, 1):
                        if nj == vj:
                            continue
                        s2 = base + Fraction(ni - vi, i + 1) + Fraction(nj - vj, j + 1)
                        if s2 == 0:
                            continue
                        if abs(s2) < best_abs:
                            c[i] = ni
                            c[j] = nj
                            s = s2
                            best_abs = abs(s2)
                            improved = True
        # occasional reset guard
        if it % 40000 == 0 and time.time() > deadline:
            break
    if s == 0:
        return None
    return c

def _collision_vector(n=30, P=2113515701):
    """Find a nonzero coefficient vector in {-1,0,1}^n with small exact value via modular collision."""
    # Weights w_i = L / i mod P, with L = lcm(1..n) mod P.
    from math import gcd
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
    Lm = L % P
    w = [0] * n
    for i in range(1, n + 1):
        w[i - 1] = (Lm * pow(i, P - 2, P)) % P  # L/i mod P (i invertible mod P)
    # Meet-in-the-middle collision: split into two halves.
    half = n // 2
    left = w[:half]
    right = w[half:]
    # Enumerate all sign vectors of the right half (2^15 = 32768), map residue -> (res, vector)
    # Use residues in {-w,0,w} per coordinate.
    table = {}
    m = len(right)
    for mask in range(1 << m):
        res = 0
        vec = [0] * m
        for k in range(m):
            bit = (mask >> k) & 1
            if bit:
                res = (res + right[k]) % P
                vec[k] = 1
        if res not in table:
            table[res] = vec
    # Now enumerate left sign vectors and look for a collision with a right vector
    # such that the combined vector is nonzero and we prefer small exact value.
    best = None
    best_key = None
    m2 = len(left)
    for mask in range(1 << m2):
        res = 0
        vec = [0] * m2
        for k in range(m2):
            if (mask >> k) & 1:
                res = (res + left[k]) % P
                vec[k] = 1
        target = (-res) % P
        rvec = table.get(target)
        if rvec is None:
            continue
        full = vec + rvec
        if all(x == 0 for x in full):
            continue
        # compute exact value
        s = Fraction(0, 1)
        for i, v in enumerate(full, start=1):
            if v:
                s += Fraction(v, i)
        if s == 0:
            continue
        key = abs(s)
        if best_key is None or key < best_key:
            best_key = key
            best = full
        # early exit if we already have something very small
        if best_key == Fraction(1, 10**12):
            break
    if best is None:
        # fallback: single-term difference
        best = [0] * n
        best[0] = 1
        best[1] = -1
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 80.0
    n = 30
    # Start from a modular collision candidate.
    try:
        base = _collision_vector(n, 2113515701)
    except Exception:
        base = [0] * (n - 2) + [1, -1]
    # Polish with local refinement under a time budget.
    refined = _local_refine(base, deadline)
    if refined is None:
        # ensure a valid nonzero fallback
        refined = [0] * (n - 2) + [1, -1]
    # final validity: exactly 30 entries in {-1,0,1}, sum nonzero
    if len(refined) != n:
        refined = (refined + [0] * n)[:n]
    for i in range(n):
        if refined[i] not in (-1, 0, 1):
            refined[i] = 0
    s = sum(Fraction(v, i + 1) for i, v in enumerate(refined))
    if s == 0:
        refined = [0] * (n - 2) + [1, -1]
    return refined
# EVOLVE-BLOCK-END
