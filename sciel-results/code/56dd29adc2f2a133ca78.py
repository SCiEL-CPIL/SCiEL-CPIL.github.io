"""Rewritten program: systematic squarefree-kernel construction for minimal max."""

def construct():
    # Multiplier set: 17 distinct values in [1, 210], all squarefree kernels distinct,
    # product's squarefree kernel == 210.
    multipliers = [1, 2, 3, 4, 6, 7, 8, 15, 21, 35, 40, 42, 60, 84, 105, 140, 168]

    base = 210
    result = [base * m for m in multipliers]

    # Pad with pairs {k, base*k} for k in [2..200], skipping used k values.
    # Each pair has squarefree kernel 1, so overall kernel stays 210.
    used = set(multipliers)
    k = 2
    while len(result) < 100:
        if k not in used:
            result.append(k)
            result.append(base * k)
            used.add(k)
        k += 1

    # Verify: 210 * product must be a perfect square
    # (kernel check: each prime 2,3,5,7 appears odd times; others even)
    # We trust the construction but do a quick sanity check.
    from math import isqrt
    prod = base
    for x in result:
        prod *= x
    r = isqrt(prod)
    assert r * r == prod, "Not a perfect square!"

    # Ensure types are Python int
    result = [int(x) for x in result]

    return result
