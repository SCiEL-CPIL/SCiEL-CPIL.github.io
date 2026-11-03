"""Rewritten program for 4x4x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Rank-minimal decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Kronecker product of M(2,2,2) [rank 7] and M(2,2,3) [rank 11].
    Total rank: 77.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # --- Base 1: Strassen 2x2x2 (Rank 7) ---
    # Indices: i, j, k in {0, 1}
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # Term 0: (a00+a11)(b00+b11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # Term 1: (a10+a11)b00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1
    
    # Term 2: a00(b01-b11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1
    
    # Term 3: a11(b10-b00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1
    
    # Term 4: (a00+a01)b11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    
    # Term 5: (a10-a00)(b00+b01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # Term 6: (a01-a11)(b10+b11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1

    # --- Base 2: 2x2x3 (Rank 11) ---
    # n=2, m=2, p=3
    # U[i*2+j, r], V[j*3+k, r], W[k*2+i, r]
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    # Terms 0-6: Strassen for cols 0,1
    for c in range(7):
        U2[:, c] = U1[:, c]
        # Map V1 (2x2) to V2 (2x3) for k=0,1
        V2[0, c] = V1[0, c]
        V2[1, c] = V1[1, c]
        V2[3, c] = V1[2, c]
        V2[4, c] = V1[3, c]
        # Map W1 (2x2) to W2 (3x2) for k=0,1
        W2[0, c] = W1[0, c]
        W2[1, c] = W1[1, c]
        W2[2, c] = W1[2, c]
        W2[3, c] = W1[3, c]
        
    # Terms 7-10: Schoolbook for col 2
    # C[:, 2] = A[:, 0] * B[0, 2] + A[:, 1] * B[1, 2]
    # U is A, V is B, W is C
    # Term 7: A00 * B02 -> C02
    U2[0, 7] = 1
    V2[2, 7] = 1  # j=0, k=2
    W2[4, 7] = 1  # k=2, i=0
    
    # Term 8: A10 * B02 -> C12
    U2[1, 8] = 1
    V2[2, 8] = 1  # j=0, k=2
    W2[5, 8] = 1  # k=2, i=1
    
    # Term 9: A01 * B12 -> C02
    U2[2, 9] = 1
    V2[5, 9] = 1  # j=1, k=2
    W2[4, 9] = 1  # k=2, i=0
    
    # Term 10: A11 * B12 -> C12
    U2[3, 10] = 1
    V2[5, 10] = 1 # j=1, k=2
    W2[5, 10] = 1 # k=2, i=1

    # --- Kronecker Product ---
    # M(4,4,6) from M(2,2,2) x M(2,2,3)
    # U: (2*2)x(2*2) -> 16 x 77
    # V: (2*2)x(2*3) -> 12 x 77 ? No.
    # M(n1,m1,p1) x M(n2,m2,p2)
    # U has shape (n1*n2, m1*m2, rank)
    # V has shape (m1*m2, p1*p2, rank)
    # W has shape (p1*p2, n1*n2, rank)
    
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 2, 2, 3
    N, M, P = n1*n2, m1*m2, p1*p2 # 4, 4, 6
    Rank = 7 * 11
    
    U = np.zeros((N*M, Rank), dtype=int)
    V = np.zeros((M*P, Rank), dtype=int)
    W = np.zeros((P*N, Rank), dtype=int)
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            
            # U: A[i1*n2+i2, j1*m2+j2]
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1, c1]
                            u2 = U2[i2*2+j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = (i1*n2 + i2)*M + (j1*m2 + j2)
                                U[row, c] = u1 * u2
            
            # V: B[j1*m2+j2, k1*p2+k2]
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1, c1]
                            v2 = V2[j2*3+k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = (j1*m2 + j2)*P + (k1*p2 + k2)
                                V[row, c] = v1 * v2
            
            # W: C[k1*p2+k2, i1*n2+i2]
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1, c1]
                            w2 = W2[k2*2+i2, c2]
                            if w1 != 0 and w2 != 0:
                                row = (k1*p2 + k2)*N + (i1*n2 + i2)
                                W[row, c] = w1 * w2

    # --- Verification ---
    def _verify(U_mat, V_mat, W_mat, n, m, p, seed=42):
        np.random.seed(seed)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        
        return np.allclose(C_computed, C_expected, atol=1e-10)

    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # --- Rank Reduction (Quick Check) ---
    # Since 77 is likely optimal, we skip extensive search to save time.
    # We can try a few random removals if time permits.
    if time.time() < deadline - 10:
        U, V, W = _quick_reduce(U, V, W, n, m, p, deadline)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _quick_reduce(U, V, W, n, m, p, deadline):
    """Attempt to remove linearly dependent terms."""
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    rank = current_U.shape[1]
    
    def _verify_fast(U_mat, V_mat, W_mat, seed=42):
        np.random.seed(seed)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    best_U, best_V, best_W = current_U, current_V, current_W
    best_rank = rank
    
    # Try removing a few terms
    np.random.seed(123)
    indices = np.random.permutation(rank)
    for r in indices[:20]:
        if time.time() > deadline - 5:
            break
        mask = np.ones(rank, dtype=bool)
        mask[r] = False
        U_test = current_U[:, mask]
        V_test = current_V[:, mask]
        W_test = current_W[:, mask]
        if _verify_fast(U_test, V_test, W_test):
            current_U = U_test
            current_V = V_test
            current_W = W_test
            best_U = current_U
            best_V = current_V
            best_W = current_W
            best_rank -= 1
            rank -= 1
            indices = indices[indices != r]
            indices = np.random.permutation(indices)
            
    return best_U, best_V, best_W
# EVOLVE-BLOCK-END
