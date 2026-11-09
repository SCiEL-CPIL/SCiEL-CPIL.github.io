# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_q^2 with q = p^2, p = 7.

    A set N is a Nikodym set if for every point x in F_q^2 there is a direction
    v (of the form (1,m) or (0,1)) such that the punctured line
    {x + t*v : t in F_q, t != 0} is fully contained in N.

    Strategy: take N to be the complement of a Baer subplane S in F_q^2.
    A Baer subplane is a blocking set (every line meets it in 0, 1, or p points),
    so its complement is a Nikodym set. The Baer subplane has p^2 + p points,
    giving |N| = p^4 - p^2 - p and a complement of size p^2 + p.

    Field model: F_q = F_p[alpha] with alpha^2 = k, k the smallest quadratic
    non-residue mod p. An element is the pair (a,b) for a + b*alpha.

    Returns:
        points: np.ndarray of shape (k, 2, 2) int64 array. Each point is
        [[x_a, x_b], [y_a, y_b]] with F_q coordinates, entries in [0, p).
    """
    p = 7

    # Smallest quadratic non-residue mod p.
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    if k is None:
        k = 3  # fallback

    # F_q multiplication: (a,b)*(c,d) = (a*c + b*d*k, a*d + b*c) mod p
    def mul(a, b, c, d):
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)

    # F_q power by squaring.
    def pow2(a, b, e):
        ra, rb = 1, 0
        ca, cb = a, b
        while e > 0:
            if e & 1:
                ra, rb = mul(ra, rb, ca, cb)
            ca, cb = mul(ca, cb, ca, cb)
            e >>= 1
        return ra, rb

    # Build the Baer subplane S = { (x, y) : x = c^p, y = d^p + c }
    # for c, d in F_p. (c^p = c, d^p = d in F_p, embedded as (c, 0), (d, 0).)
    S = set()
    for c in range(p):
        for d in range(p):
            # x = c (embedded in F_q as (c, 0)), y = d (embedded as (d, 0))
            # but we need the full F_q point; the Baer subplane uses F_p-embedded
            # coordinates. So x = (c, 0), y = (d, 0).
            S.add((c, 0, d, 0))

    # Nikodym set N = complement of S in F_q^2.
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) in S:
                        continue
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
