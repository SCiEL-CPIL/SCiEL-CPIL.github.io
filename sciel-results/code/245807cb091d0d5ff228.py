"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Kronecker product (2x2x2) ⊗ (2x2x3)
    - 2x2x2: Strassen (rank 7)
    - 2x2x3: Strassen for cols 0-1 (rank 7) + schoolbook for col 2 (rank 4) = rank 11
    - Total initial rank: 7 * 11 = 77
    - Then attempt rank reduction via linear dependency detection
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    # ===== Build Strassen 2x2x2 base (rank 7) =====
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # M2 = (a10+a11)b00 -> C10 += 1, C11 += -1
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1
    
    # M3 = a00(b01-b11) -> C01 += 1, C11 += 1
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1
    
    # M4 = a11(b10-b00) -> C00 += 1, C10 += 1
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1
    
    # M5 = (a00+a01)b11 -> C00 += -1, C01 += 1
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    
    # M6 = (a10-a00)(b00+b01) -> C11 += 1
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11) -> C00 += 1
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1
    
    # ===== Build 2x2x3 base (rank 11) =====
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    # Terms 0-6: Strassen for columns 0,1
    for c in range(7):
        U2[:, c] = U1[:, c]
        V2[0, c] = V1[0, c]  # j=0,k=0
        V2[1, c] = V1[1, c]  # j=0,k=1
        V2[3, c] = V1[2, c]  # j=1,k=0
        V2[4, c] = V1[3, c]  # j=1,k=1
        W2[0, c] = W1[0, c]  # k=0,i=0
        W2[1, c] = W1[1, c]  # k=0,i=1
        W2[2, c] = W1[2, c]  # k=1,i=0
        W2[3, c] = W1[3, c]  # k=1,i=1
    
    # Terms 7-10: Schoolbook for column 2
    U2[0, 7] = 1
    V2[2, 7] = 1   # j=0, k=2
    W2[4, 7] = 1   # k=2, i=0
    
    U2[1, 8] = 1
    V2[5, 8] = 1   # j=1, k=2
    W2[4, 8] = 1   # k=2, i=0
    
    U2[2, 9] = 1
    V2[2, 9] = 1   # j=0, k=2
    W2[5, 9] = 1   # k=2, i=1
    
    U2[3, 10] = 1
    V2[5, 10] = 1  # j=1, k=2
    W2[5, 10] = 1  # k=2, i=1
    
    # ===== Kronecker product to get 4x4x6 (rank 77) =====
    rank = 7 * 11
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1 * 2 + j1, c1]
                            u2 = U2[i2 * 2 + j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1 * 8 + i2 * 4 + j1 * 2 + j2
                                U[row, c] = u1 * u2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1 * 2 + k1, c1]
                            v2 = V2[j2 * 3 + k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = j1 * 12 + j2 * 6 + k1 * 3 + k2
                                V[row, c] = v1 * v2
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1 * 2 + i1, c1]
                            w2 = W2[k2 * 2 + i2, c2]
                            if w1 != 0 and w2 != 0:
                                row = k1 * 12 + k2 * 4 + i1 * 2 + i2
                                W[row, c] = w1 * w2
    
    # ===== Verify =====
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
    
    # ===== Rank reduction via linear dependency detection =====
    if time.time() < deadline - 10:
        U, V, W = _reduce_rank(U, V, W, n, m, p, deadline)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _reduce_rank(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by finding linear dependencies among rank-1 terms.
    Uses vectorized verification for speed.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    rank = current_U.shape[1]
    
    # Pre-compute tensor components for fast verification
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T to equal the matrix multiplication tensor: T[i,j,k] = delta(i,j) for the appropriate entries
    
    def _compute_residual(U_mat, V_mat, W_mat):
        """Compute the residual ||T_decomposed - T_mm||_F"""
        n_, m_, p_ = n, m, p
        T_decomp = np.zeros((n_, m_, p_))
        for r in range(U_mat.shape[1]):
            u_vec = U_mat[:, r].reshape(n_, m_)
            v_vec = V_mat[:, r].reshape(m_, p_)
            w_vec = W_mat[:, r].reshape(p_, n_)
            T_decomp += np.einsum('ij,jk,ki->ijk', u_vec, v_vec, w_vec)
        
        # The matrix multiplication tensor: T_mm[i,j,k] = 1 if i==j else 0... 
        # Actually for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor is T[i,j,k] where output index is (i,k), input indices are (i,j) and (j,k)
        # T_decomp[i,j,k] should equal delta_{i,j} * something... 
        # Let me think again.
        # U has shape (n*m, rank), indexed by (i,j) where i is row of A, j is col of A = row of B
        # V has shape (m*p, rank), indexed by (j,k) where j is row of B, k is col of B
        # W has shape (p*n, rank), indexed by (k,i) where k is col of B, i is row of C
        # So T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This should equal 1 if the multiplication is correct, i.e., C[i,k] = sum_j A[i,j] B[j,k]
        # The target tensor is T_target[i,j,k] = 1 for all valid (i,j,k) in the multiplication?
        # No. The tensor for matrix multiplication is:
        # T(i,j; j,k) -> (i,k), so T_target[i,j,k] = 1 for all i,j,k (it's the identity in the middle index)
        # Wait, that's not right either.
        
        # Let me use the verification approach directly
        np.random.seed(42)
        A_test = np.random.randn(n_, m_)
        B_test = np.random.randn(m_, p_)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n_, p_))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n_, m_))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m_, p_))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p_, n_).T
        
        return np.linalg.norm(C_computed - C_expected)
    
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
    
    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    best_rank = rank
    
    # Multiple passes with different orderings
    for attempt in range(8):
        if time.time() > deadline - 5:
            break
        
        trial_U = current_U.copy()
        trial_V = current_V.copy()
        trial_W = current_W.copy()
        trial_rank = trial_U.shape[1]
        
        if attempt < 2:
            order = list(range(trial_rank))
        else:
            order = list(range(trial_rank))
            np.random.seed(123 + attempt)
            np.random.shuffle(order)
        
        removed = set()
        progress = True
        while progress and trial_U.shape[1] > 1:
            if time.time() > deadline - 5:
                break
            progress = False
            for r in order:
                if time.time() > deadline - 5:
                    break
                if r in removed:
                    continue
                if trial_U.shape[1] <= 1:
                    break
                
                mask = np.ones(trial_U.shape[1], dtype=bool)
                mask[r] = False
                
                U_test = trial_U[:, mask]
                V_test = trial_V[:, mask]
                W_test = trial_W[:, mask]
                
                if _verify_fast(U_test, V_test, W_test):
                    trial_U = U_test
                    trial_V = V_test
                    trial_W = W_test
                    removed.add(r)
                    trial_rank = trial_U.shape[1]
                    progress = True
                    if trial_rank < best_rank:
                        best_U = trial_U.copy()
                        best_V = trial_V.copy()
                        best_W = trial_W.copy()
                        best_rank = trial_rank
        
        if trial_rank < best_rank:
            best_U = trial_U
            best_V = trial_V
            best_W = trial_W
            best_rank = trial_rank
    
    return best_U, best_V, best_W
# EVOLVE-BLOCK-END
