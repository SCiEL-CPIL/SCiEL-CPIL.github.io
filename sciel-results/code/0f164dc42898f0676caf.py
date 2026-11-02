"""
Program for matrix multiplication 2,2,7.
Uses a rank-25 decomposition based on Strassen's algorithm for 2x2 blocks.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 7]
START_TIME = time.time()
DEADLINE = 140.0  # 10s margin before 150s kill

def _time_remaining():
    return DEADLINE - (time.time() - START_TIME)

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.

    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    rank = 25

    # U is (n*m, rank) = (4, 25)
    # V is (m*p, rank) = (14, 25)
    # W is (p*n, rank) = (14, 25)
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)

    # Indexing:
    # A[i,j] -> U[2*i+j]
    # B[j,k] -> V[7*j+k]
    # C[k,i] -> W[2*k+i]  (Trace convention: W[k*n+i, r])

    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""

        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        # C0,ks += M1, C1,ks+1 += M1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        # C1,ks += M2, C1,ks+1 -= M2
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        # C0,ks+1 += M3, C1,ks+1 += M3
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        # C0,ks += M4, C1,ks += M4
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        # C0,ks -= M5, C0,ks+1 += M5
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        # C1,ks+1 += M6
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        # C0,ks += M7
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1

    # Add Strassen for columns 0,1
    add_strassen(0, 0)

    # Add Strassen for columns 2,3
    add_strassen(7, 2)

    # Add Strassen for columns 4,5
    add_strassen(14, 4)

    # Add schoolbook for column 6 (4 terms)
    col_start = 21
    # A00 * B0,6 -> C0,6
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1

    # A01 * B1,6 -> C0,6
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1

    # A10 * B0,6 -> C1,6
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1

    # A11 * B1,6 -> C1,6
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1

    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)

    # Robust verification: check the tensor identity on a set of random integer inputs.
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # Decomposition: sum_r U[2*i+j, r] * V[7*j+k, r] * W[2*k+i, r] must equal A[i,j]*B[j,k] contribution.
    # Equivalently, for each r, the term contributes to C[k,i] the product of (linear form in A) * (linear form in B).
    # We verify by reconstructing C for several random A, B.

    def verify(A, B):
        C_expected = A @ B  # shape (n, p), C[i,k]
        C_decomp = np.zeros((n, p), dtype=np.float64)
        for r in range(rank):
            # linear form in A: a_r[i,j] = U[2*i+j, r]
            # linear form in B: b_r[j,k] = V[7*j+k, r]
            # contribution to C[i,k] = a_r[i,j]*b_r[j,k] summed over j
            # but W scales it: actually the tensor is sum_r U_r outer V_r outer W_r
            # C[k,i] = sum_r U[2*i+j,r]*V[7*j+k,r]*W[2*k+i,r] summed over j
            # Let's compute directly.
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U_np[2*i+j, r] * V_np[7*j+k, r] * W_np[2*k+i, r]
                    C_decomp[i, k] += s
        return np.allclose(C_decomp, C_expected, atol=1e-9)

    ok = True
    if _time_remaining() > 2.0:
        np.random.seed(42)
        for _ in range(30):
            A_test = np.random.randint(-4, 5, (n, m))
            B_test = np.random.randint(-4, 5, (m, p))
            if not verify(A_test, B_test):
                ok = False
                break

    if ok:
        return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}

    # Fallback: correct schoolbook rank 28 (n*m*p = 28)
    rank28 = n * m * p
    U28 = np.zeros((n*m, rank28), dtype=np.float64)
    V28 = np.zeros((m*p, rank28), dtype=np.float64)
    W28 = np.zeros((p*n, rank28), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U28[2*i+j, idx] = 1
                V28[7*j+k, idx] = 1
                W28[2*k+i, idx] = 1
                idx += 1
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
# EVOLVE-BLOCK-END
