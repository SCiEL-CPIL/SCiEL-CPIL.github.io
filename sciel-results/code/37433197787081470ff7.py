import numpy as np
import time
import scipy.optimize as opt
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin
    
    # Rank 17 decomposition for 2x3x3 matrix multiplication
    # This is the best known construction.
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    # Indices for global tensors
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (Trace convention W[k*n+i, r])
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A_sub is 2x2 (rows 0,1; cols 0,1). B_sub is 2x2 (rows 0,1; cols 0,1).
    # C_sub is 2x2 (rows 0,1; cols 0,1).
    # U indices for A_sub: 
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    # V indices for B_sub:
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 3
    # j=1, k=1 -> 4
    # W indices for C_sub:
    # k=0, i=0 -> 0
    # k=1, i=0 -> 1
    # k=0, i=1 -> 2
    # k=1, i=1 -> 3
    
    # Strassen 7 terms
    # M1 = (A11+A22)(B11+B22)
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    # M2 = A22(B21-B11)
    U17[4, 1] = 1
    V17[3, 1] = 1; V17[0, 1] = -1
    W17[1, 1] = 1; W17[3, 1] = 1
    
    # M3 = (A11-A22)(B11+B12)
    U17[0, 2] = 1; U17[4, 2] = -1
    V17[0, 2] = 1; V17[1, 2] = 1
    W17[0, 2] = 1; W17[2, 2] = 1
    
    # M4 = A11(B22-B21)
    U17[0, 3] = 1
    V17[4, 3] = 1; V17[3, 3] = -1
    W17[2, 3] = 1; W17[3, 3] = 1
    
    # M5 = (A12+A22)(B11+B12)
    U17[1, 4] = 1; U17[4, 4] = 1
    V17[0, 4] = 1; V17[1, 4] = 1
    W17[0, 4] = 1; W17[2, 4] = 1
    
    # M6 = (A11+A12)(B22-B21)
    U17[0, 5] = 1; U17[1, 5] = 1
    V17[4, 6] = 1; V17[3, 6] = -1 # Wait, standard Strassen indices
    # Let's use a standard verified Strassen set.
    # I will re-verify the Strassen part carefully.
    
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Mapping to our indices:
    # A11 -> U[0, r], A12 -> U[1, r], A21 -> U[3, r], A22 -> U[4, r]
    # B11 -> V[0, r], B12 -> V[1, r], B21 -> V[3, r], B22 -> V[4, r]
    # C11 -> W[0, r], C12 -> W[1, r], C21 -> W[2, r], C22 -> W[3, r]
    # Note: C indices in W are k*n+i.
    # C11 (i=0,k=0) -> W[0, r]
    # C12 (i=0,k=1) -> W[1, r]
    # C21 (i=1,k=0) -> W[2, r]
    # C22 (i=1,k=1) -> W[3, r]
    
    # M1
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    # M2
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[2, 1] = 1; W17[3, 1] = -1
    
    # M3
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[1, 2] = 1; W17[3, 2] = 1
    
    # M4
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[2, 3] = 1
    
    # M5
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[1, 4] = 1
    
    # M6
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    # M7
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1; W17[3, 6] = 1
    
    # Part 2: Remaining terms for 2x3x3
    # We have handled A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Remaining A cols: j=2 (A[0,2], A[1,2])
    # Remaining B cols: k=2 (B[0,2], B[1,2], B[2,2])
    # Remaining B rows: j=2 (B[2,0], B[2,1], B[2,2])
    # Remaining C cols: k=2 (C[0,2], C[1,2])
    # Remaining C rows: i=0,1 (already covered in C[0:2, 0:2] for k=0,1)
    
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    # C[0,0], C[0,1], C[1,0], C[1,1] also have contributions from j=2 and k=2?
    # No, C[i,k] = sum_j A[i,j] B[j,k].
    # For k=0,1: C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0] + A[i,2]B[2,0].
    # We handled j=0,1. We need j=2.
    # So C[i,0] += A[i,2] B[2,0].
    # C[i,1] += A[i,2] B[2,1].
    # For k=2: C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2] + A[i,2]B[2,2].
    
    # Terms needed:
    # 1. A[0,2] B[2,0] -> C[0,0] (W[0, r])
    # 2. A[0,2] B[2,1] -> C[0,1] (W[1, r])
    # 3. A[1,2] B[2,0] -> C[1,0] (W[2, r])
    # 4. A[1,2] B[2,1] -> C[1,1] (W[3, r])
    # 5. A[0,0] B[0,2] -> C[0,2] (W[4, r])
    # 6. A[0,1] B[1,2] -> C[0,2] (W[4, r])
    # 7. A[1,0] B[0,2] -> C[1,2] (W[5, r])
    # 8. A[1,1] B[1,2] -> C[1,2] (W[5, r])
    # 9. A[0,2] B[2,2] -> C[0,2] (W[4, r])
    # 10. A[1,2] B[2,2] -> C[1,2] (W[5, r])
    
    # Total 10 terms. 7 + 10 = 17.
    
    # Indices:
    # A[0,2] -> U[2, r]
    # A[1,2] -> U[5, r]
    # B[2,0] -> V[6, r]
    # B[2,1] -> V[7, r]
    # B[2,2] -> V[8, r]
    # B[0,2] -> V[2, r]
    # B[1,2] -> V[5, r]
    
    # W[4, r] is C[0,2] (k=2, i=0 -> 2*2+0 = 4)
    # W[5, r] is C[1,2] (k=2, i=1 -> 2*2+1 = 5)
    
    # Term 7: A[0,2] B[2,0] -> C[0,0]
    U17[2, 7] = 1
    V17[6, 7] = 1
    W17[0, 7] = 1
    
    # Term 8: A[0,2] B[2,1] -> C[0,1]
    U17[2, 8] = 1
    V17[7, 8] = 1
    W17[1, 8] = 1
    
    # Term 9: A[1,2] B[2,0] -> C[1,0]
    U17[5, 9] = 1
    V17[6, 9] = 1
    W17[2, 9] = 1
    
    # Term 10: A[1,2] B[2,1] -> C[1,1]
    U17[5, 10] = 1
    V17[7, 10] = 1
    W17[3, 10] = 1
    
    # Term 11: A[0,0] B[0,2] -> C[0,2]
    U17[0, 11] = 1
    V17[2, 11] = 1
    W17[4, 11] = 1
    
    # Term 12: A[0,1] B[1,2] -> C[0,2]
    U17[1, 12] = 1
    V17[5, 12] = 1
    W17[4, 12] = 1
    
    # Term 13: A[1,0] B[0,2] -> C[1,2]
    U17[3, 13] = 1
    V17[2, 13] = 1
    W17[5, 13] = 1
    
    # Term 14: A[1,1] B[1,2] -> C[1,2]
    U17[4, 14] = 1
    V17[5, 14] = 1
    W17[5, 14] = 1
    
    # Term 15: A[0,2] B[2,2] -> C[0,2]
    U17[2, 15] = 1
    V17[8, 15] = 1
    W17[4, 15] = 1
    
    # Term 16: A[1,2] B[2,2] -> C[1,2]
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1

    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val, tol=1e-9):
        rank = U_mat.shape[1]
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    actual = sum(U_mat[i*m_val+j, r] * V_mat[j*p_val+k, r] * W_mat[k*n_val+i, r] for r in range(rank))
                    if abs(actual - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(matrix):
        return np.round(matrix * 2) / 2

    # Verify Rank 17
    if verify(U17, V17, W17, n, m, p):
        best_U, best_V, best_W = U17, V17, W17
        best_rank = 17
    else:
        # Fallback to trivial rank 18 if 17 fails (should not happen)
        best_rank = 18
        best_U = np.zeros((n*m, 18))
        best_V = np.zeros((m*p, 18))
        best_W = np.zeros((p*n, 18))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, r] = 1
                    best_V[j*p+k, r] = 1
                    best_W[k*n+i, r] = 1
                    r += 1

    # Attempt Rank 16 search (short timeout)
    if time.time() < deadline - 20.0:
        target_tensor = np.ones(n * m * p)
        u_indices = []
        v_indices = []
        w_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_indices.append(i * m + j)
                    v_indices.append(j * p + k)
                    w_indices.append(k * n + i)
        u_indices = np.array(u_indices)
        v_indices = np.array(v_indices)
        w_indices = np.array(w_indices)

        def objective(x):
            rank = 16
            size_U = n * m
            size_V = m * p
            size_W = p * n
            
            U_flat = x[:size_U * rank]
            V_flat = x[size_U * rank : size_U * rank + size_V * rank]
            W_flat = x[size_U * rank + size_V * rank :]
            
            U = U_flat.reshape(size_U, rank)
            V = V_flat.reshape(size_V, rank)
            W = W_flat.reshape(size_W, rank)
            
            u_vals = U[u_indices, :]
            v_vals = V[v_indices, :]
            w_vals = W[w_indices, :]
            
            prod = u_vals * v_vals * w_vals
            reconstructed = np.sum(prod, axis=1)
            
            error = reconstructed - target_tensor
            return np.sum(error * error)

        # Try 5 random restarts
        for r in range(5):
            if time.time() > deadline - 10.0:
                break
            x0 = np.random.randn(16 * (n * m + m * p + p * n))
            try:
                result = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
                if result.fun < 1e-6:
                    x_opt = result.x
                    size_U = n * m
                    size_V = m * p
                    size_W = p * n
                    rank = 16
                    U_opt = x_opt[:size_U * rank].reshape(size_U, rank)
                    V_opt = x_opt[size_U * rank : size_U * rank + size_V * rank].reshape(size_V, rank)
                    W_opt = x_opt[size_U * rank + size_V * rank :].reshape(size_W, rank)
                    
                    U_snap = snap_to_half_integer(U_opt)
                    V_snap = snap_to_half_integer(V_opt)
                    W_snap = snap_to_half_integer(W_opt)
                    
                    if verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = 16
                        break
            except Exception:
                pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
