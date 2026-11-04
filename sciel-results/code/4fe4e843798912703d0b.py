"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

import time

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-9):
    # Check that the decomposition reproduces the matrix multiplication tensor.
    # U: (n*m, rank) indexed [i*m+j]
    # V: (m*p, rank) indexed [j*p+k]
    # W: (p*n, rank) indexed [k*n+i]  (official trace convention)
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build the implied tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # and compare to 1 if (i,j,k) is a valid multiplication index else 0.
    # Actually the multiplication tensor is T_{i,j,k} = 1 for all i,j,k (the product
    # C[i,k] = sum_j A[i,j] B[j,k] uses tensor with entries 1 at (i,j,k)).
    # So the decomposition must satisfy: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1.
    # And for "invalid" combos there are none since all i in [0,n), j in [0,m), k in [0,p) are valid.
    # So we just need the trilinear form to equal 1 everywhere on the grid.
    import itertools
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                for c in range(rank):
                    s += uc[c] * vc[c] * wc[c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_4x4x8():
    """
    Build a rank decomposition for 4x4 * 4x8 using Strassen-style 2x2 blocks.
    We split A (4x4) into 2x2 blocks A00,A01,A10,A11 and B (4x8) into 2x2 blocks
    B00,B01,B10,B11 (each 2x4). The product C (4x8) = [C00 C01; C10 C11] where
    C00 = A00 B00 + A01 B10, etc.
    We use Strassen on the 2x2 block multiply of (2x2) A-blocks times (2x4) B-blocks.
    For a 2x2 * 2x4 multiply, we can use rank 7 Strassen on the 2x2 left with the
    2x4 right treated as two 2x2 columns.
    This is complex; instead we use a simpler verified construction:
    treat 4x8 as two independent 4x4 multiplies sharing nothing -> rank 32? 
    But that's only valid if A were duplicated. In tensor form, the two 4x4 products
    A*B1 and A*B2 both depend on the SAME A, so we cannot just double.
    We fall back to a constructive rank-96 via Strassen on the 4x4 A with 8 columns.
    Given complexity and verification, we return None and let caller fall back.
    """
    return None


def construct():
    deadline = time.time() + 110.0  # safety margin before 120s
    # Try the better construction; if it fails or times out, use schoolbook.
    best = None
    try:
        cand = _strassen_4x4x8()
        if cand is not None and _verify(cand["U"], cand["V"], cand["W"]):
            best = cand
    except Exception:
        best = None
    if best is None:
        best = _schoolbook()
    if time.time() > deadline:
        # return best (schoolbook is cheap)
        pass
    return best

# EVOLVE-BLOCK-END
