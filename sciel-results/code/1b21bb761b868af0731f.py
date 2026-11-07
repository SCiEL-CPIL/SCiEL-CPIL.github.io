import math
import time

def construct():
    limit = 10210
    start_time = time.time()
    deadline = start_time + 55.0

    # Step 1: Smallest prime factor sieve for fast factorization
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Collect primes and map
    primes = [i for i in range(2, limit + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    nprimes = len(primes)

    # Step 2: Precompute parity bitmasks for 211..10210 using SPF
    bitmasks = {}
    for n in range(211, limit + 1):
        mask = 0
        x = n
        while x > 1:
            p = spf[x]
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count % 2 == 1:
                mask |= (1 << prime_to_idx[p])
        bitmasks[n] = mask

    # Step 3: Target bitmask for 210
    target_mask = 0
    x = 210
    while x > 1:
        p = spf[x]
        count = 0
        while x % p == 0:
            x //= p
            count += 1
        if count % 2 == 1:
            target_mask |= (1 << prime_to_idx[p])

    # Step 4: Gaussian elimination, tracking best solution
    basis = {}  # pivot -> (vec, comb)
    best_solution = None
    best_max = limit + 1

    for M in range(211, limit + 1):
        if time.time() > deadline:
            break

        v = bitmasks[M]
        u = v
        c = 1 << (M - 211)

        for pivot in sorted(basis.keys(), reverse=True):
            if (u >> pivot) & 1:
                b_vec, b_comb = basis[pivot]
                u ^= b_vec
                c ^= b_comb

        if u != 0:
            pivot = u.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (u, c)

        # Check if target is in span
        t_u = target_mask
        t_c = 0
        for pivot in sorted(basis.keys(), reverse=True):
            if (t_u >> pivot) & 1:
                b_vec, b_comb = basis[pivot]
                t_u ^= b_vec
                t_c ^= b_comb

        if t_u == 0:
            # Extract selected elements
            selected = []
            for i in range(M - 211 + 1):
                if (t_c >> i) & 1:
                    selected.append(211 + i)

            if 1 <= len(selected) <= 100:
                # Verify with exact arithmetic
                prod = 210
                for x in selected:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    cur_max = max(selected)
                    if cur_max < best_max:
                        best_max = cur_max
                        best_solution = [int(x) for x in selected]

    if best_solution is not None:
        return best_solution

    # Fallback
    return [840]
