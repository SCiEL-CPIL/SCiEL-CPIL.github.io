import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 85.0


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def check_time():
        return time.time() > deadline

    # Known solution: [8, 31, 248], max = 248
    best_solution = [8, 31, 248]
    best_max = 248

    # Verify
    s = Fraction(1, 8) + Fraction(1, 31) + Fraction(1, 248)
    assert s == TARGET, f"Known solution invalid: {s}"

    # Number-theoretic approach:
    # For 3-term Egyptian fraction: 1/a + 1/b + 1/c = 5/31
    # Fix a, then 1/b + 1/c = 5/31 - 1/a = r
    # This gives: (b+c) * r = 1  =>  b+c = 1/r
    # And bc = 1/r^2 * ... 
    # Actually: 1/b + 1/c = r  =>  (b+c)/(bc) = r  =>  r*bc = b+c
    # Let r = p/q in lowest terms. Then p*bc = q*(b+c)
    # p*bc - q*b - q*c = 0
    # (pb - q)(pc - q) = q^2
    # So for each factorization q^2 = uv, we get b = (u+q)/p, c = (v+q)/p
    # and need b, c to be positive integers.

    # Try to find solutions with max < 248
    # Iterate over possible largest denominator M from 7 to 247
    # For each M, try all divisors to find valid (a, b, c) triples

    # Alternative direct approach: iterate a from 7 upward,
    # for each a compute r = 5/31 - 1/a, then solve 1/b + 1/c = r
    # using the factorization method above.

    found_better = False

    for a in range(7, 248):
        if check_time():
            break

        # r = 5/31 - 1/a
        r = TARGET - Fraction(1, a)
        if r <= 0:
            continue

        # Solve 1/b + 1/c = r where b > a, c > b
        # r = p/q in lowest terms
        p = r.numerator
        q = r.denominator

        # (pb - q)(pc - q) = q^2
        # Let u = pb - q, v = pc - q, so u*v = q^2
        # b = (u + q) / p, c = (v + q) / p
        # Need b, c positive integers, b > a, c > b, and c < 248

        # Iterate over divisors u of q^2 where u <= q (so b <= c)
        q2 = q * q
        # Find all divisors of q^2
        # For each divisor u of q^2, v = q^2 / u
        # b = (u + q) / p must be integer
        # c = (v + q) / p must be integer

        # To be efficient, iterate u from 1 to sqrt(q^2) = q
        # But q can be large. Let's iterate smartly.

        # Actually, let's just iterate u over divisors of q^2.
        # Since we need b > a, we have (u + q) / p > a => u > a*p - q
        # And c < 248: (v + q) / p < 248 => v < 248*p - q
        # v = q^2 / u, so q^2 / u < 248*p - q => u > q^2 / (248*p - q)

        # Let's just iterate u from max(1, a*p - q + 1) to min(q, q^2 // (248*p - q) - 1) if 248*p > q
        # This is getting complex. Let me just iterate u over a reasonable range.

        # Simpler: iterate u from 1 to q (since u*v=q^2 and we can assume u<=v by taking u<=q)
        # For each u that divides q^2:
        u_min = max(1, a * p - q + 1)
        u_max = q  # since u <= q when u <= v

        for u in range(u_min, u_max + 1):
            if check_time():
                break
            if q2 % u != 0:
                continue
            v = q2 // u
            # b = (u + q) / p
            num_b = u + q
            if num_b % p != 0:
                continue
            b = num_b // p
            if b <= a:
                continue
            # c = (v + q) / p
            num_c = v + q
            if num_c % p != 0:
                continue
            c = num_c // p
            if c <= b:
                continue
            if c >= 248:
                continue
            # Verify: 1/a + 1/b + 1/c == 5/31
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == TARGET:
                if c < best_max:
                    best_max = c
                    best_solution = [a, b, c]
                    found_better = True

    # If we found a better 3-term solution, also check for 4+ term solutions
    # with even smaller max using DFS
    if found_better:
        limit = best_max

        def search(remaining, used, min_d, depth, limit):
            if check_time():
                return None
            if remaining == 0:
                return []
            if depth >= MAX_TERMS:
                return None
            if remaining < 0:
                return None

            min_d_eff = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            if min_d_eff >= limit:
                return None

            for d in range(min_d_eff, limit):
                if d in used:
                    continue
                if check_time():
                    return None

                rem_new = remaining - Fraction(1, d)
                if rem_new < 0:
                    break
                if rem_new == 0:
                    return [d]

                res = search(rem_new, used | {d}, d + 1, depth + 1, limit)
                if res is not None:
                    return [d] + res

            return None

        # Search for solutions with max < best_max
        for a in range(7, limit):
            if check_time():
                break
            rem_a = TARGET - Fraction(1, a)
            if rem_a <= 0:
                continue
            res = search(rem_a, {a}, a + 1, 1, limit)
            if res is not None:
                candidate = [a] + res
                cand_max = max(candidate)
                if cand_max < best_max:
                    best_max = cand_max
                    best_solution = candidate

    return best_solution
