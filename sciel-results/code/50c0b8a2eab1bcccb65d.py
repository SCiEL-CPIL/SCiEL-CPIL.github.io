"""Matrix multiplication 2,3,5 — rank-28 decomposition via two Strassen 2x2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]
START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # Safety margin

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-28 ansatz.
    
    Strategy: Use Strassen's 7-term decomposition for two disjoint 2x2x2 blocks.
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    The remaining 14 terms are handled by schoolbook multiplication.
    Total rank: 7 + 7 + 14 = 28.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    
    # Helper to add a Strassen block
    # j_offset, k_offset
    def add_strassen_block(j_off, k_off):
        nonlocal c
        # m1 = (a00+a11)*(b00+b11), contributes to c00(+1), c11(+1)
        U[0*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m2 = (a10+a11)*b00, contributes to c10(+1), c11(-1)
        U[1*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        
        # m3 = a00*(b01-b11), contributes to c01(+1), c11(+1)
        U[0*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m4 = a11*(b10-b00), contributes to c00(+1), c10(+1)
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1; V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1; W[k_off*n + 1][c] = 1
        c += 1
        
        # m5 = (a00+a01)*b11, contributes to c00(-1), c01(+1)
        U[0*m + j_off][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        
        # m6 = (a10-a00)*(b00+b01), contributes to c11(+1)
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1; V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m7 = (a01-a11)*(b10+b11), contributes to c00(+1)
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen_block(0, 0)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen_block(0, 2)
    
    # Remaining 14 schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    # Verify with multiple random matrices for robustness
    np.random.seed(42)
    verified = True
    for trial in range(5):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        C_calc = np.zeros((n, p))
        for r in range(rank):
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n)
            L_A = np.sum(U_r * A)
            L_B = np.sum(V_r * B)
            C_calc += (L_A * L_B) * W_r.T
        if not np.allclose(C_calc, C_true, atol=1e-9):
            verified = False
            break
    
    if not verified:
        # Fallback to schoolbook
        rank = 30
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        assert c == rank

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
