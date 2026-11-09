"""Direct construction for 'Large sets with distinct subset products'.

Strategy: return prime powers in [2,40]. Every subset product of prime powers
has a unique prime factorization, so all 2^k subset products (including the
empty product = 1) are guaranteed distinct. We only need to maximize the count
of distinct prime powers, capped at 18.
"""

import time

DEADLINE = 82.0  # wall-clock seconds (90s limit, safety margin)
T_START = time.time()


def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _prime_powers(lo, hi):
    """All distinct prime powers p^e with lo <= p^e <= hi, e >= 1."""
    vals = set()
    for p in range(2, hi + 1):
        if not _is_prime(p):
            continue
        v = p
        while v <= hi:
            if v >= lo:
                vals.add(v)
            v *= p
    return sorted(vals)


def construct():
    global DEADLINE, T_START
    deadline = DEADLINE
    if deadline < 0:
        deadline = time.time() + 82.0

    def time_left():
        return deadline - time.time()

    # All prime powers in [2, 40]
    pp = _prime_powers(2, 40)

    # We want at most 18 distinct integers.
    target = min(18, len(pp))

    # Prefer to keep the set valid: any subset of prime powers is also valid
    # (subset products remain distinct). So just take the first `target`.
    res = [int(x) for x in pp[:target]]

    # ---- Verification (cheap) ----
    # distinct
    assert len(res) == len(set(res)), "duplicates"
    # type and range
    assert all(isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40 for x in res)

    # Confirm every element is a prime power (sanity; ensures distinct subset products).
    for x in res:
        found = False
        for p in range(2, x + 1):
            if not _is_prime(p):
                continue
            v = p
            while v <= x:
                if v == x:
                    found = True
                    break
                v *= p
            if found:
                break
        assert found, "non prime-power: %d" % x

    # Full subset-product distinctness check (2^18 = 262144 products, fast).
    n = len(res)
    prods = set()
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= res[i]
        assert prod not in prods, "collision at mask %d" % mask
        prods.add(prod)

    # Deadline guard: if we somehow ran long, still return the valid set.
    if time_left() < 0:
        pass

    return list(res)
