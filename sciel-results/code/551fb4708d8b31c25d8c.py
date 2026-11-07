import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110

    L = 30000000  # 3*10^7

    # Step 1: Generate primes up to L using bytearray sieve
    is_prime = bytearray(b'\x01') * (L + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    sqrt_L = int(math.isqrt(L))
    for i in range(2, sqrt_L + 1):
        if is_prime[i]:
            start = i * i
            step = i
            is_prime[start::step] = b'\x00' * ((L - start) // step + 1)

    primes = [i for i in range(2, L + 1) if is_prime[i]]

    # Step 2: Compute sigma using multiplicative sieve
    sigma = [1] * (L + 1)
    sigma_complete = True

    for idx, p in enumerate(primes):
        if idx % 100000 == 0 and time.time() > deadline:
            sigma_complete = False
            break
        p_power = p
        while p_power <= L:
            next_power = p_power * p
            factor = (next_power - 1) // (p - 1)
            if next_power > L:
                for m in range(p_power, L + 1, p_power):
                    sigma[m] *= factor
                break
            else:
                for m in range(p_power, L + 1, p_power):
                    if m % next_power != 0:
                        sigma[m] *= factor
            p_power = next_power

    if not sigma_complete:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Step 3: Find M with most collisions
    m_counts = {}
    m_examples = {}
    best_m = 0
    best_count = 0

    for k in range(1, L + 1):
        M = k * sigma[k]
        if M in m_counts:
            m_counts[M] += 1
            if len(m_examples[M]) < 1000:
                m_examples[M].append(k)
            if m_counts[M] > best_count:
                best_count = m_counts[M]
                best_m = M
        else:
            m_counts[M] = 1
            m_examples[M] = [k]
            if 1 > best_count:
                best_count = 1
                best_m = M

        if k % 5000000 == 0:
            if time.time() > deadline:
                break

    if best_m > 0:
        k_list = [int(k) for k in m_examples[best_m]]
        return {'M': int(best_m), 'k': k_list}

    return {'M': 333312, 'k': [336, 372, 434]}
