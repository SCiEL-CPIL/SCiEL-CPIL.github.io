"""Matrix multiplication 4,4,5 - Strassen block decomposition + rational rank reduction."""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def elapsed():
        return time.time() - deadline

    # Index mappings
    # U[i*m+j] for A[i,j], m=4
    A11 = {(0, 0): 0, (0, 1): 1, (1, 0): 4, (1, 1): 5}
    A12 = {(0, 0): 2, (0, 1): 3, (1, 0): 6, (1, 1): 7}
    A21 = {(0, 0): 8, (0, 1): 9, (1, 0): 12, (1, 1): 13}
    A22 = {(0, 0): 10, (0, 1): 11, (1, 0): 14, (1, 1): 15}

    # V[j*p+k] for B[j,k], p=5
    B11 = {(0, 0): 0, (0, 1): 1, (1, 0): 5, (1, 1): 6}
    B12 = {(0, 0): 2, (0, 1): 3, (1, 0): 7, (1, 1): 8}
    B13 = {(0, 0): 4, (1, 0): 9}
    B21 = {(0, 0): 10, (0, 1): 11, (1, 0): 15, (1, 1): 16}
    B22 = {(0, 0): 12, (0, 1): 13, (1, 0): 17, (1, 1): 18}
    B23 = {(0, 0): 14, (1, 0): 19}

    # W[k*n+i] for C[i,k], n=4
    C11 = {(0, 0): 0, (0, 1): 4, (1, 0): 1, (1, 1): 5}
    C12 = {(0, 0): 8, (0, 1): 12, (1, 0): 9, (1, 1): 13}
    C13 = {(0, 0): 16, (1, 0): 17}
    C21 = {(0, 0): 2, (0, 1): 6, (1, 0): 3, (1, 1): 7}
    C22 = {(0, 0): 10, (0, 1): 14, (1, 0): 11, (1, 1): 15}
    C23 = {(0, 0): 18, (1, 0): 19}

    terms = []

    def add_strassen_2x2(A_block, B_block, C_block):
        """Add Strassen terms for 2x2*2x2=2x2."""
        x11, x12 = A_block[(0, 0)], A_block[(0, 1)]
        x21, x22 = A_block[(1, 0)], A_block[(1, 1)]
        y11, y12 = B_block[(0, 0)], B_block[(0, 1)]
        y21, y22 = B_block[(1, 0)], B_block[(1, 1)]
        z11, z12 = C_block[(0, 0)], C_block[(0, 1)]
        z21, z22 = C_block[(1, 0)], C_block[(1, 1)]

        # M1 = (x11+x22)*(y11+y22) -> contributes to z11, z22
        terms.append(({x11: 1, x22: 1}, {y11: 1, y22: 1}, {z11: 1, z22: 1}))
        # M2 = (x21+x22)*y11 -> contributes to z21, z22
        terms.append(({x21: 1, x22: 1}, {y11: 1}, {z21: 1, z22: -1}))
        # M3 = x11*(y12-y22) -> contributes to z12, z22
        terms.append(({x11: 1}, {y12: 1, y22: -1}, {z12: 1, z22: 1}))
        # M4 = x22*(y21-y11) -> contributes to z11, z21
        terms.append(({x22: 1}, {y21: 1, y11: -1}, {z11: 1, z21: 1}))
        # M5 = (x11+x12)*y22 -> contributes to z12, z11
        terms.append(({x11: 1, x12: 1}, {y22: 1}, {z12: 1, z11: -1}))
        # M6 = (x21-x11)*(y11+y12) -> contributes to z22
        terms.append(({x21: 1, x11: -1}, {y11: 1, y12: 1}, {z22: 1}))
        # M7 = (x12-x22)*(y21+y22) -> contributes to z11, z22
        terms.append(({x12: 1, x22: -1}, {y21: 1, y22: 1}, {z11: 1, z22: 1}))

    def add_schoolbook_2x1(A_block, B_block, C_block):
        """Add schoolbook terms for 2x2*2x1=2x1."""
        for i in range(2):
            for j in range(2):
                terms.append(({A_block[(i, j)]: 1}, {B_block[(j, 0)]: 1}, {C_block[(i, 0)]: 1}))

    # Build decomposition
    # C11 = A11*B11 + A12*B21
    add_strassen_2x2(A11, B11, C11)
    add_strassen_2x2(A12, B21, C11)

    # C12 = A11*B12 + A12*B22
    add_strassen_2x2(A11, B12, C12)
    add_strassen_2x2(A12, B22, C12)

    # C13 = A11*B13 + A12*B23
    add_schoolbook_2x1(A11, B13, C13)
    add_schoolbook_2x1(A12, B23, C13)

    # C21 = A21*B11 + A22*B21
    add_strassen_2x2(A21, B11, C21)
    add_strassen_2x2(A22, B21, C21)

    # C22 = A21*B12 + A22*B22
    add_strassen_2x2(A21, B12, C22)
    add_strassen_2x2(A22, B22, C22)

    # C23 = A21*B13 + A22*B23
    add_schoolbook_2x1(A21, B13, C23)
    add_schoolbook_2x1(A22, B23, C23)

    rank = len(terms)

    # Build U, V, W as lists of Fractions
    U = [[Fraction(0)] * rank for _ in range(nm)]
    V = [[Fraction(0)] * rank for _ in range(mp)]
    W = [[Fraction(0)] * rank for _ in range(pn)]

    for c, (U_t, V_t, W_t) in enumerate(terms):
        for row, coeff in U_t.items():
            U[row][c] = Fraction(coeff)
        for row, coeff in V_t.items():
            V[row][c] = Fraction(coeff)
        for row, coeff in W_t.items():
            W[row][c] = Fraction(coeff)

    # Verify correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c] for c in range(rank))
                if s != 1:
                    # Fall back to schoolbook
                    rank_sb = n * m * p
                    U = [[Fraction(0)] * rank_sb for _ in range(nm)]
                    V = [[Fraction(0)] * rank_sb for _ in range(mp)]
                    W = [[Fraction(0)] * rank_sb for _ in range(pn)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][col] = Fraction(1)
                                V[jj * p + kk][col] = Fraction(1)
                                W[kk * n + ii][col] = Fraction(1)
                                col += 1
                    return {
                        "U": [[float(x) for x in row] for row in U],
                        "V": [[float(x) for x in row] for row in V],
                        "W": [[float(x) for x in row] for row in W],
                    }

    # Rational Gaussian elimination to find minimal rank
    # Each term c corresponds to a vector in R^(nm+mp+pn)
    # Build the list of term vectors
    dim = nm + mp + pn
    term_vectors = []
    for c in range(rank):
        vec = [Fraction(0)] * dim
        for i in range(nm):
            vec[i] = U[i][c]
        for j in range(mp):
            vec[nm + j] = V[j][c]
        for k in range(pn):
            vec[nm + mp + k] = W[k][c]
        term_vectors.append(vec)

    # Gaussian elimination to find a basis
    basis = []
    for tv in term_vectors:
        v = tv[:]
        for b in basis:
            # Find pivot in b
            pivot = None
            for idx in range(dim):
                if b[idx] != 0:
                    pivot = idx
                    break
            if pivot is None:
                continue
            if v[pivot] != 0:
                factor = v[pivot] / b[pivot]
                for idx in range(dim):
                    v[idx] -= factor * b[idx]
        # Check if v is zero
        if all(x == 0 for x in v):
            continue
        basis.append(v)

    final_rank = len(basis)

    # If no reduction, return the original
    if final_rank == rank:
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W],
        }

    # Build new U, V, W from the basis vectors
    U_new = [[Fraction(0)] * final_rank for _ in range(nm)]
    V_new = [[Fraction(0)] * final_rank for _ in range(mp)]
    W_new = [[Fraction(0)] * final_rank for _ in range(pn)]

    for c, bv in enumerate(basis):
        for i in range(nm):
            U_new[i][c] = bv[i]
        for j in range(mp):
            V_new[j][c] = bv[nm + j]
        for k in range(pn):
            W_new[k][c] = bv[nm + mp + k]

    # Verify the reduced decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_new[i * m + j][c] * V_new[j * p + k][c] * W_new[k * n + i][c]
                        for c in range(final_rank))
                if s != 1:
                    # Fall back to original (unreduced)
                    return {
                        "U": [[float(x) for x in row] for row in U],
                        "V": [[float(x) for x in row] for row in V],
                        "W": [[float(x) for x in row] for row in W],
                    }

    return {
        "U": [[float(x) for x in row] for row in U_new],
        "V": [[float(x) for x in row] for row in V_new],
        "W": [[float(x) for x in row] for row in W_new],
    }
# EVOLVE-BLOCK-END
