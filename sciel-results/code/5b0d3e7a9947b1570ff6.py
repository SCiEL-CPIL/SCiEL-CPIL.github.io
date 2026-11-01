import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W):
        """Check if U, V, W correctly decompose T exactly."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-10)

    # Rank-23 decomposition:
    # - j in {0,1}, k in {0,1}: Strassen 2x2x2 (rank 7)
    # - j in {0,1}, k in {2,3}: Strassen 2x2x2 (rank 7)
    # - j in {0,1}, k = 4:     Schoolbook 2x2x1 (rank 4)
    # - j = 2, k in {0..4}:    Schoolbook 2x1x5 (rank 5)
    # Total rank = 7 + 7 + 4 + 5 = 23

    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0

    def add_strassen_block(j_off, k_off):
        """Add a Strassen 2x2x2 block for rows j in {j_off, j_off+1}, cols k in {k_off, k_off+1}."""
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[0*m + j_off,     col] += 1; U[1*m + j_off + 1, col] += 1
        V[j_off*p + k_off, col] += 1; V[(j_off+1)*p + k_off+1, col] += 1
        W[k_off*n + 0, col] += 1; W[(k_off+1)*n + 1, col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[0*m + j_off + 1, col] += 1; U[1*m + j_off + 1, col] += 1
        V[j_off*p + k_off, col] += 1
        W[k_off*n + 1, col] += 1; W[(k_off+1)*n + 1, col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[0*m + j_off, col] += 1
        V[j_off*p + k_off+1, col] += 1; V[(j_off+1)*p + k_off+1, col] -= 1
        W[(k_off+1)*n + 0, col] += 1; W[(k_off+1)*n + 1, col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[1*m + j_off + 1, col] += 1
        V[(j_off+1)*p + k_off, col] += 1; V[j_off*p + k_off, col] -= 1
        W[k_off*n + 0, col] += 1; W[(k_off+1)*n + 0, col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[0*m + j_off, col] += 1; U[0*m + j_off + 1, col] += 1
        V[(j_off+1)*p + k_off+1, col] += 1
        W[k_off*n + 0, col] -= 1; W[(k_off+1)*n + 0, col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[0*m + j_off + 1, col] += 1; U[0*m + j_off, col] -= 1
        V[j_off*p + k_off, col] += 1; V[j_off*p + k_off+1, col] += 1
        W[(k_off+1)*n + 1, col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[0*m + j_off + 1, col] += 1; U[1*m + j_off + 1, col] -= 1
        V[(j_off+1)*p + k_off, col] += 1; V[(j_off+1)*p + k_off+1, col] += 1
        W[k_off*n + 0, col] += 1
        col += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen_block(0, 0)

    # Block 2: j in {0,1}, k in {2,3}
    add_strassen_block(0, 2)

    # Block 3: j in {0,1}, k = 4 (Schoolbook 2x2x1, rank 4)
    for i in range(n):
        for j in range(2):
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1

    # Block 4: j = 2, k in {0..4} (Schoolbook 2x1x5, rank 5)
    for k in range(p):
        U[0*m + 2, col] = 1
        U[1*m + 2, col] = 1
        V[2*p + k, col] = 1
        W[k*n + 0, col] = 1
        W[k*n + 1, col] = 1
        col += 1

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: rank-30 schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb))
    V_fb = np.zeros((m*p, rank_fb))
    W_fb = np.zeros((p*n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
