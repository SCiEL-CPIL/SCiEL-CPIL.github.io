import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 210
    max_val = 10210

    # Sieve for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    def get_parity_mask(x):
        """Return bitmask where bit i is set if prime primes[i] appears to odd power in x."""
        mask = 0
        temp = x
        for idx, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count ^= 1
                if count:
                    mask |= (1 << idx)
        if temp > 1:
            # temp is a prime factor
            if temp in prime_index:
                mask |= (1 << prime_index[temp])
        return mask

    # Precompute masks for all numbers up to max_val
    masks = [0] * (max_val + 1)
    for i in range(211, max_val + 1):
        masks[i] = get_parity_mask(i)

    # Parity mask of 210: 2^1 * 3^1 * 5^1 * 7^1
    # Indices: 2->0, 3->1, 5->2, 7->3
    target = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

    min_M = None
    basis = {}  # key: pivot bit index, value: (vector, history_mask)
    
    # Iterate M from 211 upwards
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break

        v = masks[M]
        h = 1 << (M - 211)  # History bit for candidate M

        # Insert v into basis
        tv = v
        th = h
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                bv, bh = basis[p]
                tv ^= bv
                th ^= bh
            else:
                basis[p] = (tv, th)
                break

        # Check if target is in span
        tv = target
        th_sol = 0
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                bv, bh = basis[p]
                tv ^= bv
                th_sol ^= bh
            else:
                break

        if tv == 0:
            min_M = M
            # We have a solution represented by th_sol
            # Extract the solution
            sol_mask = th_sol
            sol = []
            hh = sol_mask
            while hh:
                lsb = hh & -hh
                idx = lsb.bit_length() - 1
                sol.append(211 + idx)
                hh ^= lsb
            
            # Check if this solution is valid (size <= 100, product square)
            if len(sol) <= 100:
                prod = 210
                for x in sol:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return [int(x) for x in sol]
            
            # If the automatic solution is too large or invalid (shouldn't happen if math is right),
            # try to find a smaller solution manually.
            # First, check for size 1
            for i in range(211, min_M + 1):
                if masks[i] == target:
                    return [int(i)]
            
            # Check for size 2
            for i in range(211, min_M + 1):
                mi = masks[i]
                for j in range(i + 1, min_M + 1):
                    if mi ^ masks[j] == target:
                        return [int(i), int(j)]
            
            # Check for size 3
            for i in range(211, min_M + 1):
                mi = masks[i]
                for j in range(i + 1, min_M + 1):
                    mij = mi ^ masks[j]
                    for k in range(j + 1, min_M + 1):
                        if mij ^ masks[k] == target:
                            return [int(i), int(j), int(k)]
            
            # If we get here, return the basis solution as fallback (it must be valid)
            if len(sol) <= 100:
                return [int(x) for x in sol]

    # Fallback if no M found (should not happen)
    # 840 is a known solution: 210 * 840 = 210 * 4 * 210 = 420^2
    return [840]
