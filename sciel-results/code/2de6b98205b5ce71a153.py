def construct():
    """
    Construct a set of distinct integers in [2, 40] with distinct subset products.
    Uses prime-power family decomposition: each number is assigned to a prime family,
    and within each family, exponents are chosen so all subset sums are distinct.
    Cross-family independence is guaranteed by distinct primes.
    Returns a list of ints (the selected numbers).
    """
    import math
    from itertools import combinations

    def subset_sums_distinct(exponents):
        """Check if all subset sums of exponents are distinct."""
        sums = set()
        for mask in range(1 << len(exponents)):
            s = 0
            for i in range(len(exponents)):
                if mask & (1 << i):
                    s += exponents[i]
            if s in sums:
                return False
            sums.add(s)
        return True

    def max_independent_exponents(max_exp):
        """Find the largest subset of {1..max_exp} with distinct subset sums."""
        for size in range(max_exp, 0, -1):
            for combo in combinations(range(1, max_exp + 1), size):
                if subset_sums_distinct(combo):
                    return list(combo)
        return [1]

    # Determine max exponent for each prime in [2, 40]
    # p^e <= 40 => e <= floor(log(40)/log(p))
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    result = []
    for p in primes:
        max_e = 1
        while p ** (max_e + 1) <= 40:
            max_e += 1
        exponents = max_independent_exponents(max_e)
        for e in exponents:
            result.append(p ** e)

    # Final verification: check all constraints
    # 1. All in [2, 40]
    assert all(2 <= x <= 40 for x in result)
    # 2. All distinct
    assert len(result) == len(set(result))
    # 3. All subset products distinct
    products = set()
    k = len(result)
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= result[i]
        if prod in products:
            raise ValueError("Duplicate subset product found")
        products.add(prod)

    return [int(x) for x in result]
