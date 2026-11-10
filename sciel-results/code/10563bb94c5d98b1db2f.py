<<<<<<< SEARCH
    if best_len <= 2:
        return [int(primes[0]), int(primes[1])]

    # Reconstruct chain
    chain = [int(primes[best_j]), int(primes[best_i])]
    ci = best_i
    cj = best_j
    clen = best_len

    while clen > 2:
        k = int(amax[ci, cj])
        if k < 0:
            break
        chain.append(int(primes[k]))
        cj = ci
        ci = k
        clen -= 1

    chain.reverse()

    # Verify
    prime_set = set(primes.tolist())
    valid = (len(chain) >= 2 and
             all(chain[i+1] > chain[i] for i in range(len(chain)-1)) and
             all(p in prime_set for p in chain) and
             all(chain[i+1] - chain[i] >= chain[i] - chain[i-1] for i in range(1, len(chain)-1)) and
             chain[-1] <= 10000)

    if valid:
        return chain
    else:
        # Fallback: greedy
        values = []
        for p in range(2, 10001):
            if all(p % d for d in range(2, isqrt(p) + 1)):
                if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                    values.append(p)
        return values
=======
    if best_len <= 2:
        return [int(primes[0]), int(primes[1])]

    # Reconstruct chain
    chain = [int(primes[best_j]), int(primes[best_i])]
    ci = best_i
    cj = best_j
    clen = best_len

    while clen > 2:
        k = int(amax[ci, cj])
        if k < 0:
            break
        chain.append(int(primes[k]))
        cj = ci
        ci = k
        clen -= 1

    chain.reverse()

    # Verify
    prime_set = set(primes.tolist())
    valid = (len(chain) >= 2 and
             all(chain[i+1] > chain[i] for i in range(len(chain)-1)) and
             all(p in prime_set for p in chain) and
             all(chain[i+1] - chain[i] >= chain[i] - chain[i-1] for i in range(1, len(chain)-1)) and
             chain[-1] <= 10000)

    if not valid:
        # Fallback: greedy
        values = []
        for p in range(2, 10001):
            if all(p % d for d in range(2, isqrt(p) + 1)):
                if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                    values.append(p)
        return values

    # If time remains, try to extend the chain from the end
    elapsed = time.time() - t0
    if elapsed < deadline - 2:
        # Try extending from the last element
        last_prime = chain[-1]
        last_gap = chain[-1] - chain[-2] if len(chain) >= 2 else 0
        extended = list(chain)
        for p in primes[primes > last_prime]:
            if p - last_prime >= last_gap:
                extended.append(int(p))
                last_prime = int(p)
                last_gap = int(p) - extended[-2]
            else:
                break
        if len(extended) > len(chain):
            # Verify extended chain
            if (all(extended[i+1] > extended[i] for i in range(len(extended)-1)) and
                all(p in prime_set for p in extended) and
                all(extended[i+1] - extended[i] >= extended[i] - extended[i-1] for i in range(1, len(extended)-1)) and
                extended[-1] <= 10000):
                chain = extended

    return chain
>>>>>>> REPLACE
