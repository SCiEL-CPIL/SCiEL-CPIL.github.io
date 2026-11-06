"""
Strategy: Deterministic construction using paired exponents and finite-abelian-group
structure to force massive cancellation in the square.

Key insight: If we pair terms so that each "effective monomial" has coefficient 2a_i,
then the square has terms (2a_i)(2a_j) x^{e_i+e_j} = 4 a_i a_j x^{e_i+e_j}.
If we choose exponents from a small cyclic group (e.g., Z_m) lifted to actual exponents
via a homomorphism, then many pairs (i,j) will have e_i+e_j = e_k+e_l mod m,
and if we map those to the SAME actual exponent, the coefficients add and may cancel.

Concretely: choose 6 exponents e_0..e_5 from a small set such that the multiset
{e_i + e_j : i <= j} has as few distinct values as possible. The minimum distinct
sums for 6 elements is achieved when they form an arithmetic progression or a
subgroup structure. For 6 elements in Z_6, if we take all 6 elements, the sums
cover all of Z_6 (6 distinct values). But we can do better with structure.

Actually, the best known: 6 elements in an abelian group, the number of distinct
pair sums (with repetition, i<=j) is at least 6. We can achieve exactly 6 if the
elements form a coset of a subgroup or are in arithmetic progression in Z_6.

Let's use exponents that are multiples of a base B, so e_i = B * g_i where g_i are
in Z_6. Then e_i + e_j = B*(g_i+g_j). If g_i+g_j mod 6 only takes 6 values, and
we map each residue class to a single actual exponent, we get at most 6 distinct
exponents in the square, plus we need the coefficients to cancel within each class.

But we have 12 terms, paired: term 2i and 2i+1 both have exponent e_i and coefficient
a_i, so the effective coefficient at e_i is 2a_i. The square coefficient at exponent
B*(g_i+g_j mod 6) is sum over all (i,j) with g_i+g_j ≡ r (mod 6) of 4*a_i*a_j.

We need to choose a_0..a_5 (integers in [-32,32] so that 2a_i is in [-64,64]) such
that for each residue r, the sum is zero or as small as possible.

This is a system of 6 equations (one per residue) in 15 unknowns (a_i*a_j for i<=j).
We can try to make all 6 sums zero. This is a quadratic system. Let's try a_i = 1
for all i: then sum for residue r = 4 * (number of pairs (i,j) with g_i+g_j ≡ r).
If the g_i are all distinct in Z_6, each residue appears exactly 6 times (since for
each i, there's exactly one j such that g_i+g_j ≡ r). So sum = 4*6 = 24 ≠ 0.

Instead, choose g_i to be a subgroup: {0,1,2,3,4,5} is the whole group. Try g_i =
{0,0,1,1,2,2} (duplicates). Then pairs: (0,0): sum 0, (0,1): sum 1, (0,2): sum 2,
(1,1): sum 2, (1,2): sum 3, (2,2): sum 4. Distinct sums: {0,1,2,3,4} = 5 values.
With a_i = a, b, a, b, c, c (matching the duplicate pattern), we can try to make
each sum zero.

Let's just brute-force small integer a_i and g_i patterns.
"""
import time
import random
from itertools import product

DEADLINE = 90.0
SAFETY_MARGIN = 8.0


def _square_nonzero_count(coefs):
    n = len(coefs)
    sq = [0] * (2 * n - 1)
    nz = [(i, c) for i, c in enumerate(coefs) if c != 0]
    for a_i, a_c in nz:
        for b_j, b_c in nz:
            sq[a_i + b_j] += a_c * b_c
    return sum(1 for x in sq if x != 0)


def _valid_candidate(coefs):
    if len(coefs) > 41:
        return False
    if not coefs:
        return False
    if coefs[-1] == 0:
        return False
    if any(c not in range(-64, 65) for c in coefs):
        return False
    if sum(1 for c in coefs if c != 0) != 12:
        return False
    return True


def _construct_from_pattern(g_pattern, a_values, base):
    """
    g_pattern: list of 6 integers (residues mod 6)
    a_values: list of 6 integers (coefficients for each pair)
    base: multiplier for actual exponents
    Returns coefficient list or None if invalid.
    """
    if len(g_pattern) != 6 or len(a_values) != 6:
        return None
    coefs = [0] * 41
    for i in range(6):
        exp = base * (g_pattern[i] % 6)
        # Two terms at this exponent, each with coefficient a_values[i]
        if exp >= 41:
            return None
        # Check that 2*a_values[i] is in range (it is, since we'll bound a_values)
        if coefs[exp] != 0:
            # Collision: two different g values map to same exponent
            # This can happen if base*g1 == base*g2, i.e., g1==g2 mod 6
            # But we already have duplicates, so this is expected
            # Actually, if g1 != g2 but g1%6 == g2%6, they're the same
            # So this only happens if same i appears twice, which it doesn't
            pass
        coefs[exp] += a_values[i]
        # Second term of the pair
        coefs[exp] += a_values[i]
    # Trim trailing zeros
    while len(coefs) > 1 and coefs[-1] == 0:
        coefs.pop()
    if len(coefs) > 41:
        return None
    if not _valid_candidate(coefs):
        return None
    return coefs


def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN

    best_coefs = None
    best_sq_nz = 10**9

    def update(cand):
        nonlocal best_coefs, best_sq_nz
        if cand is None or not _valid_candidate(cand):
            return
        nz = _square_nonzero_count(cand)
        if nz < best_sq_nz:
            best_sq_nz = nz
            best_coefs = list(cand)

    # Seed: trivial
    seed = [1] * 12
    update(seed)

    # Try structured patterns
    # g_patterns: 6-tuples of residues mod 6
    # We want the pair sums to have few distinct values
    rng = random.Random(42)

    # Precompute all 6-tuples of residues mod 6 (6^6 = 46656, manageable)
    # But filter to those with at most 8 distinct pair sums
    good_patterns = []
    for g in product(range(6), repeat=6):
        sums = set()
        for i in range(6):
            for j in range(i, 6):
                sums.add((g[i] + g[j]) % 6)
        if len(sums) <= 8:
            good_patterns.append(g)

    # Try each good pattern with small coefficient assignments
    # Coefficients a_i in [-3, -2, -1, 0, 1, 2, 3] (so 2a_i in [-6,6])
    # But we need exactly 12 nonzero terms, so no a_i can be 0
    # Actually, if a_i = 0, both terms at that exponent are zero, reducing term count
    # So we need all a_i != 0
    coeff_choices = [-3, -2, -1, 1, 2, 3]

    # Try base = 1, 2, 3 (so exponents are 0..5, 0..10, 0..15)
    for base in [1, 2, 3]:
        for g in good_patterns:
            # Try a subset of coefficient assignments
            for a in product(coeff_choices, repeat=6):
                if time.time() > deadline:
                    break
                coefs = _construct_from_pattern(g, a, base)
                update(coefs)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Also try random search as fallback
    if time.time() < deadline:
        for _ in range(200000):
            if time.time() > deadline:
                break
            exponents = sorted(rng.sample(range(0, 41), 12))
            coefs = [0] * 41
            for e in exponents:
                coefs[e] = rng.choice([-3, -2, -1, 1, 2, 3])
            update(coefs)

    if best_coefs is None or not _valid_candidate(best_coefs):
        best_coefs = [1] * 12

    return list(best_coefs)
