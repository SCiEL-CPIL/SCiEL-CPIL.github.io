import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    N = 210
    MAX_X = 10210
    MAX_K = 100

    # Sieve of Eratosthenes
    limit = MAX_X
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Precompute square-free kernel vectors for x in (210, 10210]
    vecs = [0] * (limit + 1)
    for x in range(211, limit + 1):
        v = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt % 2 == 1:
                    v |= (1 << prime_to_idx[p])
        if temp > 1:
            v |= (1 << prime_to_idx[temp])
        vecs[x] = v

    # Target: square-free part of 210 = 2*3*5*7
    target_vec = (1 << prime_to_idx[2]) | (1 << prime_to_idx[3]) | (1 << prime_to_idx[5]) | (1 << prime_to_idx[7])

    # Incremental basis with history
    basis = {}  # pivot_bit -> (vec, history_int)

    for Y in range(211, limit + 1):
        if time.time() > deadline:
            break

        v = vecs[Y]
        hist = 1 << Y

        # Reduce by existing basis
        while v:
            hsb = v.bit_length() - 1
            if hsb in basis:
                b_vec, b_hist = basis[hsb]
                v ^= b_vec
                hist ^= b_hist
            else:
                break

        # Add to basis if independent
        if v != 0:
            hsb = v.bit_length() - 1
            basis[hsb] = (v, hist)

        # Check if target is in span
        v_t = target_vec
        hist_t = 0
        while v_t:
            hsb = v_t.bit_length() - 1
            if hsb in basis:
                b_vec, b_hist = basis[hsb]
                v_t ^= b_vec
                hist_t ^= b_hist
            else:
                break

        if v_t == 0 and hist_t != 0:
            # Extract selected numbers
            selected = []
            ht = hist_t
            while ht:
                lsb = (ht & -ht).bit_length() - 1
                selected.append(lsb)
                ht ^= (1 << lsb)

            if len(selected) <= MAX_K:
                # Verify
                prod = 210
                for x in selected:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return list(selected)

    return [840]
