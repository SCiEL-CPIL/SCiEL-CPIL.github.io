import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Strassen 2x2 coefficients
    # beta[s][I][J] for A, gamma[s][J][K] for B, delta[s][I][K] for C
    beta = np.zeros((7, 2, 2), dtype=float)
    gamma = np.zeros((7, 2, 2), dtype=float)
    delta = np.zeros((7, 2, 2), dtype=float)

    # s=0
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1

    # s=1
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1

    # s=2
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1

    # s=3
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1

    # s=4
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1

    # s=5
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1

    # s=6
    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1

    # Build 49-term decomposition using Kronecker products
    # r = 7*s + t
    U_full = np.zeros((16, 49), dtype=float)
    V_full = np.zeros((16, 49), dtype=float)
    W_full = np.zeros((16, 49), dtype=float)

    for s in range(7):
        for t in range(7):
            r = 7 * s + t

            # U[i*4+j, r]: i = 2*I + i', j = 2*J + j'
            # U_r = kron(beta[s], beta[t])
            U_full[:, r] = np.kron(beta[s], beta[t]).flatten()

            # V[j*4+k, r]: j = 2*J + j', k = 2*K + k'
            # V_r = kron(gamma[s], gamma[t])
            V_full[:, r] = np.kron(gamma[s], gamma[t]).flatten()

            # W[k*4+i, r]: k = 2*K + k', i = 2*I + i'
            # W_r[2*K+k', 2*I+i'] = delta[s][I][K] * delta[t][i'][k']
            # This is kron(delta[s].T, delta[t].T)
            W_full[:, r] = np.kron(delta[s].T, delta[t].T).flatten()

    if time.time() > deadline - 50.0:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Verify the 49-term decomposition
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_test = A_test @ B_test

    C_49 = np.zeros((4, 4))
    for r in range(49):
        u_col = U_full[:, r]
        v_col = V_full[:, r]
        w_col = W_full[:, r]
        # Reshape to 4x4 for clarity
        u_mat = u_col.reshape(4, 4)
        v_mat = v_col.reshape(4, 4)
        w_mat = w_col.reshape(4, 4)
        # Contribution to C[i,k] = sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # We can compute this efficiently
        # temp[i,j] = u_mat[i,j] * v_mat[j,k] ... but k is summed
        # Actually: C[i,k] += sum_j U[i,j] V[j,k] W[k,i]
        # Let's just do it directly for verification
        for i in range(4):
            for k in range(4):
                wk_i = w_mat[k, i]
                if wk_i == 0:
                    continue
                s_val = 0.0
                for j in range(4):
                    s_val += u_mat[i, j] * v_mat[j, k]
                C_49[i, k] += s_val * wk_i

    if not np.allclose(C_49, C_test, atol=1e-9):
        # If verification fails, return the raw 49 terms anyway (shouldn't happen)
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Now attempt to find a linear dependency among the 49 rank-1 tensors
    # The tensor T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We check if {T_r} are linearly dependent in R^(4*4*4) = R^64
    # Flatten each T_r to a 64-dim vector
    
    rank1_tensors = np.zeros((49, 64), dtype=float)
    for r in range(49):
        u_col = U_full[:, r]
        v_col = V_full[:, r]
        w_col = W_full[:, r]
        
        # Efficient construction of the tensor vector
        # Index mapping: idx = i*16 + j*4 + k
        # Value: U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
        
        for i in range(4):
            for j in range(4):
                u_val = U_full[i*4+j, r]
                if u_val == 0:
                    continue
                for k in range(4):
                    v_val = V_full[j*4+k, r]
                    if v_val == 0:
                        continue
                    w_val = W_full[k*4+i, r]
                    if w_val == 0:
                        continue
                    rank1_tensors[r, i*16 + j*4 + k] = u_val * v_val * w_val

    if time.time() > deadline - 20.0:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Check rank of the 49x64 matrix
    # We want to find alpha in R^49 such that sum_r alpha_r T_r = 0
    # This is rank1_tensors.T @ alpha = 0
    # rank1_tensors.T is 64x49
    # If rank < 49, there is a non-trivial null space
    
    # Use SVD of rank1_tensors (49x64)
    # rank1_tensors = U_svd @ diag(S) @ Vt
    # rank1_tensors.T = Vt.T @ diag(S) @ U_svd.T
    # Null space of rank1_tensors.T is the orthogonal complement of the row space of rank1_tensors
    # Rows of rank1_tensors span a subspace of R^64 of dimension `rank`
    # The null space of rank1_tensors.T (which maps R^49 -> R^64) consists of vectors alpha
    # such that alpha is orthogonal to the column space of rank1_tensors.T?
    # No. rank1_tensors.T @ alpha = 0.
    # Let M = rank1_tensors.T (64x49).
    # Null(M) = {alpha | M @ alpha = 0}.
    # M @ alpha = rank1_tensors.T @ alpha.
    # This is equivalent to finding the null space of the 64x49 matrix.
    # The dimension of the null space is 49 - rank(M).
    # rank(M) = rank(rank1_tensors).
    
    try:
        U_svd, S, Vt = np.linalg.svd(rank1_tensors, full_matrices=False)
    except Exception:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # S contains singular values.
    # The rank is the number of significant singular values.
    tol = 1e-10
    rank = np.sum(S > tol)
    
    if rank < 49:
        # There is a null space of dimension 49 - rank
        # We need alpha such that rank1_tensors.T @ alpha = 0
        # From SVD: rank1_tensors = U_svd @ diag(S) @ Vt
        # rank1_tensors.T = Vt.T @ diag(S) @ U_svd.T
        # Let alpha = U_svd @ y
        # rank1_tensors.T @ alpha = Vt.T @ diag(S) @ U_svd.T @ U_svd @ y = Vt.T @ diag(S) @ y
        # Since U_svd is 49x49 (square) and orthogonal, U_svd.T @ U_svd = I.
        # So we need Vt.T @ diag(S) @ y = 0.
        # Vt is 49x49? No, Vt is 49x49 if full_matrices=False for 49x64?
        # For M (m x n) with m=49, n=64:
        # U is m x m (49x49), S is min(m,n) (49), Vt is min(m,n) x n (49x64).
        # Wait, standard SVD: M = U S Vt.
        # U: 49x49, S: 49x49 (diagonal), Vt: 49x64.
        # M.T = V S.T U.T.
        # V is 64x49.
        # M.T @ alpha = V @ S.T @ U.T @ alpha = 0.
        # Let y = U.T @ alpha. Then alpha = U @ y.
        # V @ S.T @ y = 0.
        # S.T is 49x49 diagonal.
        # Let s = diag(S).
        # V is 64x49.
        # The equation is V @ (s * y) = 0.
        # This doesn't look right. Let's re-evaluate.
        
        # We want alpha such that sum_r alpha_r T_r = 0.
        # T_r is the r-th row of rank1_tensors (flattened).
        # So rank1_tensors.T @ alpha = 0.
        # rank1_tensors is 49x64.
        # Let's just compute the null space of rank1_tensors.T directly.
        
        # rank1_tensors.T is 64x49.
        # We can use np.linalg.null_space on rank1_tensors.T
        # Or simply, since we have SVD of rank1_tensors:
        # rank1_tensors = U_svd @ np.diag(S) @ Vt
        # rank1_tensors.T = Vt.T @ np.diag(S) @ U_svd.T
        # We want x such that rank1_tensors.T @ x = 0.
        # Vt.T @ np.diag(S) @ U_svd.T @ x = 0.
        # Let z = U_svd.T @ x.
        # Vt.T @ np.diag(S) @ z = 0.
        # Vt is 49x64. Vt.T is 64x49.
        # np.diag(S) is 49x49.
        # z is 49x1.
        # Vt.T @ (S * z) = 0.
        # This implies that the vector w = S * z is in the null space of Vt.T.
        # But Vt.T has full column rank if rank=49. If rank < 49, it doesn't.
        # Actually, the null space of rank1_tensors.T is the orthogonal complement of the column space of rank1_tensors.T.
        # The column space of rank1_tensors.T is the same as the row space of rank1_tensors.
        # The row space of rank1_tensors is spanned by the first `rank` rows of Vt? No.
        # Row space of M is spanned by rows of Vt? No, M = U S Vt. Rows of M are linear combos of rows of Vt.
        # Specifically, row i of M is u_i^T S Vt.
        # The non-zero singular vectors correspond to the first `rank` rows of Vt.
        # So the row space is spanned by the first `rank` rows of Vt.
        # The null space of M.T is the orthogonal complement of the column space of M.T.
        # Column space of M.T = Row space of M.
        # So Null(M.T) = (Row Space of M)^perp.
        # Since Row Space of M is spanned by v_1, ..., v_rank (rows of Vt),
        # Null(M.T) is spanned by v_{rank+1}, ..., v_49 (rows of Vt).
        
        # So we can take alpha to be any linear combination of the rows of Vt from index `rank` to 48.
        
        # Let's pick the last one: alpha = Vt[rank, :]
        
        alpha = Vt[rank, :]
        
        # Verify
        check = rank1_tensors.T @ alpha
        if np.max(np.abs(check)) < 1e-8:
            # Found a valid dependency
            # Remove the term with the largest absolute coefficient
            r_remove = np.argmax(np.abs(alpha))
            
            keep = [r for r in range(49) if r != r_remove]
            U_48 = U_full[:, keep]
            V_48 = V_full[:, keep]
            W_48 = W_full[:, keep]
            
            # Verify rank-48 decomposition
            C_48 = np.zeros((4, 4))
            for idx, r in enumerate(keep):
                u_mat = U_48[:, idx].reshape(4, 4)
                v_mat = V_48[:, idx].reshape(4, 4)
                w_mat = W_48[:, idx].reshape(4, 4)
                for i in range(4):
                    for k in range(4):
                        wk_i = w_mat[k, i]
                        if wk_i == 0:
                            continue
                        s_val = 0.0
                        for j in range(4):
                            s_val += u_mat[i, j] * v_mat[j, k]
                        C_48[i, k] += s_val * wk_i
            
            if np.allclose(C_48, C_test, atol=1e-9):
                return {"U": U_48.tolist(), "V": V_48.tolist(), "W": W_48.tolist()}
    
    # If no dependency found or verification failed, return rank 49
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
