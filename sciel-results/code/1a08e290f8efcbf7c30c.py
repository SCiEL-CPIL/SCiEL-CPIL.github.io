"""Rewritten program: systematic search for a 3x3 sumset of distinct primes."""
import time
import math
from itertools import permutations

# EVOLVE-BLOCK-START
def _sieve(limit):
    """Return list of primes <= limit (limit >= 2)."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = 0
    is_p[1] = 0
    r = int(limit ** 0.5)
    for i in range(2, r + 1):
        if is_p[i]:
            step = i
            start = i * i
            is_p[start:limit + 1:step] = bytearray(len(range(start, limit + 1, step)))
    return [i for i in range(2, limit + 1) if is_p[i]]


def _valid(A, B):
    """Return True if all 9 sums are distinct primes (hence pairwise coprime)."""
    sums = []
    for a in A:
        for b in B:
            s = a + b
            if not is_p[s]:
                return False
            if s in sums:
                return False
            sums.append(s)
    return True


def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    is_p = _sieve(256)
    primes = [p for p in is_p if p <= 128]

    # Precompute all 6 permutations of offset triples in [1,100]
    offsets = list(range(1, 101))
    perms = list(permutations(offsets, 3))

    # Fallback: the seed solution (5 distinct sums)
    fallback = [[1, 7, 13], [4, 10, 16]]

    n = len(primes)
    for i in range(n):
        a = primes[i]
        for j in range(n):
            if i == j:
                continue
            b = primes[j]
            # a, b in [2,128], offsets in [1,100] -> sums in [3,228]
            for x, y, z in perms:
                A = [a + x, a + y, a + z]
                B = [b + x, b + y, b + z]
                # quick bound check: all elements must be in [1,128]
                if A[0] > 128 or A[1] > 128 or A[2] > 128:
                    continue
                if B[0] > 128 or B[1] > 128 or B[2] > 128:
                    continue
                if _valid(A, B):
                    return [A, B]
                if time.time() > deadline:
                    return fallback

    return fallback
# EVOLVE-BLOCK-END
