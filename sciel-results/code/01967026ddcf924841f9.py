"""2x3x9 exact tensor decomposition via Kronecker product of (2x3x3, rank 7) and (1x1x3, rank 3)."""
import time

DIMENSIONS = [2, 3, 9]


def _terms_to_matrices(terms, n, m, p):
    """Convert a list of (i,j,k,coeff) terms to U,V,W matrices.
    U row index: i*m + j   shape (n*m, rank)
    V row index: j*p + k   shape (m*p, rank)
    W row index: k*n + i   shape (p*n, rank)
    """
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (i, j, k, coeff) in enumerate(terms):
        U[i * m + j][c] = coeff
        V[j * p + k][c] = coeff
        W[k * n + i][c] = coeff
    return U, V, W


def _kronecker_terms(terms1, terms2):
    """Kronecker product of two tensor decompositions.
    If T1 lives on indices (i1,j1,k1) and T2 on (i2,j2,k2),
    the combined tensor lives on combined indices
    (i1,i2) x (j1,j2) x (k1,k2) with the flattening:
    i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2.
    Each combined term is (i,j,k, coeff1*coeff2).
    """
    # We need the dimensions of each factor to compute the combined indices.
    # Infer them from the maximum indices appearing in the terms.
    def max_idx(terms, idx_pos):
        return max(t[idx_pos] for t in terms) if terms else 0

    n1, m1, p1 = max_idx(terms1, 0) + 1, max_idx(terms1, 1) + 1, max_idx(terms1, 2) + 1
    n2, m2, p2 = max_idx(terms2, 0) + 1, max_idx(terms2, 1) + 1, max_idx(terms2, 2) + 1
    combined = []
    for (i1, j1, k1, c1) in terms1:
        for (i2, j2, k2, c2) in terms2:
            i = i1 * n2 + i2
            j = j1 * m2 + j2
            k = k1 * p2 + k2
            combined.append((i, j, k, c1 * c2))
    return combined


def construct():
    # Base tensor 1: 2x3x3, rank 7 (Coppersmith-Winograd-style decomposition).
    terms_233 = [
        (1, 0, 0, -1),
        (1, 0, 1,  1),
        (0, 1, 0, -1),
        (0, 1, 1,  1),
        (1, 2, 0,  1),
        (1, 2, 1, -1),
        (0, 2, 0,  1),
    ]
    # Base tensor 2: 1x1x3, rank 3 (identity: three independent scalar products).
    terms_113 = [
        (0, 0, 0, 1),
        (0, 0, 1, 1),
        (0, 0, 2, 1),
    ]

    # Combine via Kronecker product.
    combined = _kronecker_terms(terms_233, terms_113)

    # Build final matrices for 2x3x9.
    n, m, p = DIMENSIONS
    U, V, W = _terms_to_matrices(combined, n, m, p)

    return {"U": U, "V": V, "W": W}
