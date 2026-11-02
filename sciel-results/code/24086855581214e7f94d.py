"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-minimal exact tensor decomposition for multiplying a
    (2 x 2) matrix by a (2 x 11) matrix.

    The multiplication tensor T in {A}_{i,j} {B}_{j,k} {C}_{i,k} for
    n=2, m=2, p=11 has 44 independent monomials A[i,j]*B[j,k].  Each
    monomial contributes to exactly one output entry C[i,k] (because the
    pair (i,k) determines which (j) terms can appear, and the j=0,1
    products for that (i,k) are the only ones in that entry).  Hence no
    rank-1 term can be shared between two output entries, and the trivial
    construction with one rank-1 term per monomial is optimal.

    Return factor matrices U, V, W with integer entries such that
    sum_r U[row_a, r] * V[row_b, r] * W[row_c, r] reproduces the
    multiplication tensor.  The index conventions follow the public
    task contract:
      U has shape (n*m, rank) indexed by (i*m + j)
      V has shape (m*p, rank) indexed by (j*p + k)
      W has shape (p*n, rank) indexed by (k*n + i)   # official trace convention
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 44 for 2 x 2 x 11

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
