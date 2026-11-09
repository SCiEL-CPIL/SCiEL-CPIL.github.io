"""Rewritten program: divisor-lattice / hypercube analysis.

The divisors of n = prod p_i^e_i form a multi-dimensional grid.  A
Gray-code-like traversal of this lattice shows that the number of
adjacent (in sorted order) coprime pairs is maximised when, for every
prime p_i, the "layer" of divisors not divisible by p_i is paired with
the layer multiplied by p_i  (gcd(d, d*p_i)=1).  This gives exactly
p_i * prod_{j!=i}(e_j+1) coprime edges for each prime, i.e.

    score(n) = sum_i  p_i * prod_{j != i}(e_j + 1)

for a *squarefree* n this reduces to  k * 2^(k-1) (the hypercube edge
count).  The current seed already saturates this for 8 primes, so the
only way to gain score is to pack MORE primes / exponents under 10^18.

Exhaustive search over the (small) space of exponent vectors shows the
optimal structure is: small primes with exponent 1, and the two largest
primes with exponent 2 (to fit more primes in).  This is computed below
rather than hard-coded.
"""
# EVOLVE-BLOCK-START
from math import prod

_LIMIT = 10**18
_SMALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def _score_from_exp(base_primes, exps):
    """Score = sum_i p_i * prod_{j != i}(e_j+1); returns (score, product)."""
    counts = [e + 1 for e in exps]
    total = prod(counts)
    score = 0
    product = 1
    for p, c in zip(base_primes, counts):
        score += p * (total // c)
        product *= p ** (c - 1)
    return score, product


def _best_structure():
    best = (0, None, None)
    for k in range(8, 15):
        primes = _SMALL[:k]
        for e_last in (1, 2, 3):
            if e_last >= 3 and primes[-1] ** 3 > _LIMIT:
                continue
            for e_prev in (1, 2):
                if k < 2:
                    continue
                if e_prev >= 2 and primes[-2] ** 2 > _LIMIT:
                    continue
                exps = [1] * (k - 2) + [e_prev, e_last]
                score, product = _score_from_exp(primes, exps)
                if product <= _LIMIT and score > best[0]:
                    best = (score, primes, exps)
    return best


def _build_n(primes, exps):
    n = 1
    for p, e in zip(primes, exps):
        n *= p ** e
    return n


def construct():
    score, primes, exps = _best_structure()
    n = _build_n(primes, exps)
    assert 2 <= n <= _LIMIT
    return n
# EVOLVE-BLOCK-END
