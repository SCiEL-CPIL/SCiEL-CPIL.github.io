import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """Fast verification using vectorized numpy operations."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_mat = np.array(U, dtype=float)  # shape (n*m, rank)
    V_mat = np.array(V, dtype=float)  # shape (m*p, rank)
    W_mat = np.array(W, dtype=float)  # shape (p*n, rank)

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)

        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('jk,jkc->kc', B, V_reshaped)
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_reshaped)

        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True


def _build_strassen_2x2x2():
    """Standard Strassen decomposition for 2x2 * 2x2 -> 2x2, rank 7."""
    U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, 1]
    ], dtype=float)  # (4, 7)

    V = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)  # (4, 7)

    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)  # (4, 7)
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_block_decomposition(n, m, p):
    """
    Build a block-Strassen decomposition.
    Split A into n x (m/2) blocks of size n x m/2 (if m is even)
    Actually, let's split along m and p.
    
    For 2x4 * 4x8:
    Split A into 2 blocks of 2x2 (left, right)
    Split B into 2 blocks of 2x8 (top, bottom)
    C = A_left @ B_top + A_right @ B_bot
    
    Each 2x2 * 2x8 can be further decomposed into 4 * (2x2 * 2x2)
    Each 2x2 * 2x2 uses Strassen rank 7.
    Total rank = 2 * 4 * 7 = 56.
    """
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # Split A into two 2x2 blocks: A_left = A[:,0:2], A_right = A[:,2:4]
    # B is 4x8: rows j=0..3, cols k=0..7
    # Split B into top 2x8 and bottom 2x8: B_top = B[0:2,:], B_bot = B[2:4,:]
    # C is 2x8: C[i,k] = sum_j A[i,j]*B[j,k]
    #   = sum_{j=0}^1 A[i,j]*B[j,k] + sum_{j=2}^3 A[i,j]*B[j,k]
    #   = (A_left @ B_top)[i,k] + (A_right @ B_bot)[i,k]
    
    # For A_left (2x2) * B_top (2x8):
    # B_top is 2x8, so we can split it into 4 blocks of 2x2:
    # B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    # Each 2x2 * 2x2 uses Strassen rank 7
    
    col_offset = 0
    for block in range(4):  # 4 sub-blocks for left part (B_top)
        for f in range(7):
            col = col_offset + f
            # U: A_left[i,j] where i,j in {0,1} -> index i*2+j in 0..3
            for r in range(4):
                U[r][col] = U_s[f][r]
            # V: B_top[j, k] where j in {0,1}, k = 2*block..2*block+1
            for r in range(4):
                j = r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_s[f][r]
            # W: C[i, k] where i in {0,1}, k = 2*block..2*block+1
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_s[r][f]
        col_offset += 7
    
    for block in range(4):  # 4 sub-blocks for right part (B_bot)
        for f in range(7):
            col = col_offset + f
            # U: A_right[i,j] where i,j in {0,1} -> index i*2+2+j in 4..7
            for r in range(4):
                U[2 + r][col] = U_s[f][r]
            # V: B_bot[j, k] where j in {2,3}, k = 2*block..2*block+1
            for r in range(4):
                j = 2 + r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_s[f][r]
            # W: C[i, k] where i in {0,1}, k = 2*block..2*block+1
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_s[r][f]
        col_offset += 7
    
    return U.tolist(), V.tolist(), W.tolist()


def _try_linear_reduction(U, V, W, n, m, p, start_time, deadline):
    """
    Try to reduce rank by solving linear systems.
    If we have a rank-r decomposition, we can try to find a rank-(r-1) one
    by fixing U, V and solving for W, or by perturbations.
    """
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try reducing by 1, 2, ... up to 10
    for reduction in range(1, 11):
        if time.time() - start_time > deadline - 5:
            break
        target_rank = best_rank - reduction
        if target_rank <= 0:
            break
        
        # Try a few random perturbations
        for attempt in range(3):
            if time.time() - start_time > deadline - 5:
                break
            
            # Perturb U, V slightly
            U_pert = np.array(U, dtype=float) + np.random.randn(n*m, best_rank) * 0.01
            V_pert = np.array(V, dtype=float) + np.random.randn(m*p, best_rank) * 0.01
            
            # Solve for W: T_{i,j,k} = sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c
            # This is linear in W.
            # For each (k,i), we need W_{k,i}^c.
            # The tensor T has 64 components.
            # We have target_rank * n * p = target_rank * 16 unknowns for W.
            # We need at least 64 equations, so target_rank >= 4.
            
            # Form the linear system:
            # For each (i,j,k), T_{i,j,k} = sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c
            # Let's use the fact that T is the matrix multiplication tensor.
            # T_{i,j,k} = 1 if (i,j,k) is a valid index for the product, but actually
            # the tensor for matrix multiplication is:
            # T_{i,j,k} = sum over... no, the tensor is defined as:
            # (A*B)_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}
            # where T_{i,j,k} = 1 if the indices match, i.e., for matrix mult:
            # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
            # So T_{i,j,k} = 1 for all valid i,j,k? No.
            # The tensor T is such that:
            # sum_j A_{i,j} B_{j,k} = sum_c (sum_j A_{i,j} U_{i,j}^c) (sum_k B_{j,k} V_{j,k}^c) W_{k,i}^c
            # This must equal sum_j A_{i,j} B_{j,k} for all A, B.
            # So T_{i,j,k} in the original tensor is 1 if we're multiplying, but actually
            # the tensor for matrix multiplication is:
            # T_{i,j,k} = delta_{j,j} ... no.
            # Let's use the standard: the tensor is T_{i,j,k} = 1 for all i,j,k? No.
            # For n x m * m x p:
            # (A*B)_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}
            # So the tensor has T_{i,j,k} = 1 for all valid (i,j,k) where j is the shared index.
            # But in the decomposition, we have:
            # sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c = 1 for all (i,j,k) that are part of the product.
            # Actually, the tensor is:
            # T_{i,j,k} = 1 if we're looking at the (i,j,k) entry of the multiplication tensor.
            # For matrix multiplication, T_{i,j,k} = 1 for all (i,j,k) with i in [0,n), j in [0,m), k in [0,p).
            # Wait, no. The tensor T is a 3-tensor such that:
            # (A*B)_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}
            # where T_{i,j,k} = 1 for all (i,j,k) because (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
            # So T_{i,j,k} = 1 for all valid i,j,k.
            
            # So we need: sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c = 1 for all (i,j,k) with i<n, j<m, k<p.
            # This is a system of n*m*p = 64 equations in n*p*target_rank = 16*target_rank unknowns.
            
            # Let's form the matrix:
            # For each (i,j,k), we have an equation:
            # sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c = 1
            # Let's vectorize W: W_vec of length n*p*target_rank.
            # W_vec[(k*n+i)*target_rank + c] = W_{k,i}^c
            
            # The equation is:
            # sum_c U_{i,j}^c V_{j,k}^c W_vec[(k*n+i)*target_rank + c] = 1
            
            # We can form a 64 x (16*target_rank) matrix.
            A_mat = np.zeros((64, 16 * target_rank))
            b_vec = np.zeros(64)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        b_vec[idx] = 1
                        for c in range(target_rank):
                            A_mat[idx, (k*n+i)*target_rank + c] = U_pert[i*m+j, c] * V_pert[j*p+k, c]
                        idx += 1
            
            # Solve the least squares problem
            try:
                W_vec, residuals, rank_solve, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-6:
                    W_pert = np.zeros((p*n, target_rank), dtype=float)
                    for k in range(p):
                        for i in range(n):
                            for c in range(target_rank):
                                W_pert[k*n+i, c] = W_vec[(k*n+i)*target_rank + c]
                    
                    U_pert_list = U_pert[:, :target_rank].tolist()
                    V_pert_list = V_pert[:, :target_rank].tolist()
                    W_pert_list = W_pert.tolist()
                    
                    if _verify_fast(U_pert_list, V_pert_list, W_pert_list, n, m, p, trials=10, tol=1e-9):
                        best_U = U_pert_list
                        best_V = V_pert_list
                        best_W = W_pert_list
                        best_rank = target_rank
                        break
            except:
                pass
    
    return best_U, best_V, best_W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the block-Strassen decomposition (rank 56)
    U, V, W = _build_block_decomposition(n, m, p)
    
    if not _verify_fast(U, V, W, n, m, p, trials=30, tol=1e-9):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to reduce rank via linear solve
    best_U, best_V, best_W = _try_linear_reduction(U, V, W, n, m, p, start_time, deadline)
    
    if time.time() - start_time > deadline - 5:
        return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
