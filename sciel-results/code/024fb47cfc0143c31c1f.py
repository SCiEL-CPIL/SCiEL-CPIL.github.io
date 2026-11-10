"""Segmented-sieve approach for distinct totient runs."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Block size: compute phi for 1..B
    B = 5_000_000

    # Sieve primes up to sqrt(B)
    limit = int(math.isqrt(B))
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    if limit >= 1:
        is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = bytearray(len(is_prime[start::step]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    # Initialize phi array: phi[n] = n for n in 1..B
    phi = list(range(B + 1))

    # For each prime p, apply phi[n] -= phi[n]//p for multiples of p
    for p in primes:
        for m in range(p, B + 1, p):
            phi[m] -= phi[m] // p

    # Sliding window scan for longest run of distinct phi values
    best_len = 1
    best_start = 1
    last_pos = {}  # phi value -> last index where it appeared
    left = 1
    for x in range(1, B + 1):
        v = phi[x]
        if v in last_pos and last_pos[v] >= left:
            # Move left past the previous occurrence
            left = last_pos[v] + 1
        last_pos[v] = x
        cur_len = x - left + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = left
            if best_len >= 200:
                break

    return {'start': best_start, 'length': best_len}
# EVOLVE-BLOCK-END
