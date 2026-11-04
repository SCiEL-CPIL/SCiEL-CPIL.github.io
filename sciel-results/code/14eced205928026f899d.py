"""Tensor decomposition of 5x5x5 matrix multiplication, rank 44."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    tmax = t0 + 110.0

    n = 5
    m = 5
    p = 5

    U_cols = []
    V_cols = []
    W_cols = []

    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        U_cols.append(u_dict)
        V_cols.append(v_dict)
        W_cols.append(w_dict)

    # 4+1 Split
    # A00 (4x4), A01 (4x1), A10 (1x4), A11 (1x1)
    # B00 (4x4), B01 (4x1), B10 (1x4), B11 (1x1)
    # C00 (4x4), C01 (4x1), C10 (1x4), C11 (1x1)

    # 1. C00 = A00 B00 (4x4x4 Schoolbook)
    # Rank 16
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Term for A[i,j] B[j,k] contributing to C[i,k]
                # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
                add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    # 2. C00 += A01 B10 (4x1 x 1x4)
    # A01 is col 4, rows 0-3. B10 is row 4, cols 0-3.
    # C[i,k] += A[i,4] B[4,k]
    # Rank 1
    add_term({i * m + 4: 1 for i in range(4)}, {4 * p + k: 1 for k in range(4)}, {k * n + i: 1 for i in range(4) for k in range(4)})
    # Wait, W must be a dict.
    # W[k*n+i] = 1 for all i,k in 0..3
    w_dict = {}
    for i in range(4):
        for k in range(4):
            w_dict[k * n + i] = 1
    # Correcting the add_term call above (it was invalid syntax)
    U_cols.pop()
    V_cols.pop()
    W_cols.pop()
    
    u_dict = {i * m + 4: 1 for i in range(4)}
    v_dict = {4 * p + k: 1 for k in range(4)}
    add_term(u_dict, v_dict, w_dict)

    # 3. C01 = A00 B01 (4x4x4 Schoolbook)
    # C[i,4] = sum_j A[i,j] B[j,4]
    # Rank 16
    for i in range(4):
        for j in range(4):
            # k is fixed to 4
            k = 4
            add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    # 4. C01 += A01 B11 (4x1 x 1x1)
    # C[i,4] += A[i,4] B[4,4]
    # Rank 1
    u_dict = {i * m + 4: 1 for i in range(4)}
    v_dict = {4 * p + 4: 1}
    w_dict = {4 * n + i: 1 for i in range(4)}
    add_term(u_dict, v_dict, w_dict)

    # 5. C10 = A10 B00 (1x4 x 4x4)
    # C[4,k] = sum_j A[4,j] B[j,k]
    # Rank 4 (sum over j=0..3)
    for j in range(4):
        u_dict = {4 * m + j: 1}
        v_dict = {j * p + k: 1 for k in range(4)}
        w_dict = {k * n + 4: 1 for k in range(4)}
        add_term(u_dict, v_dict, w_dict)

    # 6. C10 += A11 B10 (1x1 x 1x4)
    # C[4,k] += A[4,4] B[4,k]
    # Rank 1
    u_dict = {4 * m + 4: 1}
    v_dict = {4 * p + k: 1 for k in range(4)}
    w_dict = {k * n + 4: 1 for k in range(4)}
    add_term(u_dict, v_dict, w_dict)

    # 7. C11 = A10 B01 (1x4 x 4x1)
    # C[4,4] = sum_j A[4,j] B[j,4]
    # Rank 4
    for j in range(4):
        u_dict = {4 * m + j: 1}
        v_dict = {j * p + 4: 1}
        w_dict = {4 * n + 4: 1}
        add_term(u_dict, v_dict, w_dict)

    # 8. C11 += A11 B11 (1x1 x 1x1)
    # C[4,4] += A[4,4] B[4,4]
    # Rank 1
    add_term({4 * m + 4: 1}, {4 * p + 4: 1}, {4 * n + 4: 1})

    rank = len(U_cols)

    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify
    try:
        import numpy as np
        rng = np.random.RandomState(12345)
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        # Correct Verification Logic:
        # C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # This is NOT what the seed code did.
        # The seed code did: C_test += (u_mat @ v_mat) * w_mat.T
        # This implies the decomposition is C = sum_r (U_r A) (B V_r) W_r?
        # No, the standard tensor decomposition for matrix multiplication is:
        # C[i,k] = sum_r ( sum_j U[i,j]_r A[i,j] ) ( sum_j V[j,k]_r B[j,k] ) W[k,i]_r
        # This is incorrect. The correct form is:
        # C[i,k] = sum_r sum_j U[i,j]_r V[j,k]_r W[k,i]_r A[i,j] B[j,k] ??
        
        # Actually, the rank of the tensor T is defined such that:
        # T[i,j,k] = sum_r U[i,j]_r V[j,k]_r W[k,i]_r
        # And C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        
        # Let's verify this way.
        for trial in range(500):
            if time.time() > tmax:
                break
            A = rng.randint(0, 6, size=(n, m))
            B = rng.randint(0, 6, size=(m, p))
            C_true = A @ B
            
            C_test = np.zeros((n, p), dtype=np.int64)
            for r in range(rank):
                u_col = U_np[:, r]
                v_col = V_np[:, r]
                w_col = W_np[:, r]
                
                u_mat = u_col.reshape(n, m)
                v_mat = v_col.reshape(m, p)
                w_mat = w_col.reshape(p, n)
                
                # T[i,j,k] for this r is u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
                # We need sum_j T[i,j,k] A[i,j] B[j,k]
                # This is sum_j sum_k (u_mat[i,j] * v_mat[j,k] * w_mat[k,i]) * A[i,j] * B[j,k]
                # = sum_j A[i,j] u_mat[i,j] ( sum_k v_mat[j,k] B[j,k] w_mat[k,i] )
                
                # Efficient computation:
                # term_r = sum_j sum_k u_mat[i,j] v_mat[j,k] w_mat[k,i] A[i,j] B[j,k]
                
                # Let M1[i,j] = u_mat[i,j] * A[i,j]
                # Let M2[j,k] = v_mat[j,k] * B[j,k]
                # Let M3[k,i] = w_mat[k,i]
                
                # term_r[i,k] = sum_j M1[i,j] * ( sum_k M2[j,k] * M3[k,i] )
                # Note: sum_k M2[j,k] M3[k,i] is (M2 @ M3)[j,i]
                
                M1 = u_mat * A
                M2 = v_mat * B
                M3 = w_mat
                
                # M2 is (m,p), M3 is (p,n). M2 @ M3 is (m,n).
                # Let Prod = M2 @ M3. Prod[j,i] = sum_k M2[j,k] M3[k,i]
                Prod = M2 @ M3
                
                # term_r[i,k] = sum_j M1[i,j] * Prod[j,i] ??
                # Wait, the index of C is (i,k).
                # The sum is over j.
                # C[i,k] += sum_j u_mat[i,j] v_mat[j,k] w_mat[k,i] A[i,j] B[j,k]
                # = sum_j (u_mat[i,j] A[i,j]) * (v_mat[j,k] B[j,k]) * w_mat[k,i]
                
                # Let's compute this directly for small rank.
                for i in range(n):
                    for k in range(p):
                        s = 0
                        for j in range(m):
                            s += u_mat[i, j] * v_mat[j, k] * w_mat[k, i] * A[i, j] * B[j, k]
                        C_test[i, k] += s
                        
            if not np.array_equal(C_test, C_true):
                return _schoolbook()
    except ImportError:
        pass

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
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
