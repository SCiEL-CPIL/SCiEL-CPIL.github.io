import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    # Construct the target tensor T as a 3D array T[i, j, k] where:
    # i index in U (0..N-1) -> (i_idx, j_idx)
    # j index in V (0..M-1) -> (j_idx, k_idx)
    # k index in W (0..P-1) -> (k_idx, i_idx)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) match the multiplication structure
    # Specifically: u=(i,j), v=(j',k), w=(i',k')
    # T[u,v,w] = 1 if j==j' and i==i' and k==k'
    
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def verify(U, V, W):
        # Check if sum_r U[:,r] outer V[:,r] outer W[:,r] == T
        # Efficient check:
        # Residual = T - sum(U[:,r,None,None] * V[:,None,r,None] * W[:,None,None,r])
        # This is expensive if done naively.
        # Better:
        # For each r, add U[:,r] * V[:,r] * W[:,r] to accumulator
        # Or use einsum
        # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        # return np.allclose(T_recon, T, atol=1e-9)
        
        # Given the "integer/half-integer" constraint, we can check exactness by rounding
        # But first, let's assume the search finds a float solution close to the exact one.
        # We will snap to 0.5 grid and verify exactly.
        pass

    def snap_to_half(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap

    def check_exact(U, V, W):
        # Exact check using integers (multiply by 2)
        U_int = (U * 2).astype(int)
        V_int = (V * 2).astype(int)
        W_int = (W * 2).astype(int)
        
        # Check if sum_r (2U_ir) (2V_jr) (2W_kr) / 8 == T_ijk
        # sum_r U_ir V_jr W_kr = T_ijk
        # (1/8) sum_r (2U_ir) (2V_jr) (2W_kr) = T_ijk
        # sum_r U_int_ir V_int_jr W_int_kr = 8 * T_ijk
        
        # Compute product
        # U_int shape (N, R), V_int (M, R), W_int (P, R)
        # Product tensor shape (N, M, P)
        # We can do this in chunks or use einsum
        # T_recon = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        # Target = 8 * T
        
        # To save memory/time, check a few random indices or all?
        # N*M*P = 9*15*15 = 2025. R <= 45.
        # 2025 * 45 ops is small.
        
        T_recon = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        T_target = (8 * T).astype(int)
        
        return np.array_equal(T_recon, T_target)

    best_result = None
    best_rank = n * m * p # 45

    # Try ranks from 28 upwards
    # It is known that rank 28 is possible for 3x3x5
    for R in range(28, best_rank + 1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try multiple random initializations
        # Number of restarts depends on time
        # Let's allow 20 restarts per rank
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                # Fix V, W.
                # T_ijk = sum_r U_ir V_jr W_kr
                # For each i:
                # T_i = sum_r U_ir (V_jr W_kr)
                # Let M_r = V[:,r] * W[:,r] (element-wise product, shape MxP flattened to M*P? No, M and P are indices)
                # Actually, V is (M, R), W is (P, R).
                # V[:,r] is length M. W[:,r] is length P.
                # The term for rank r is U_ir * (V[:,r] outer W[:,r])
                # We want to solve for U[:,r] such that sum_r U_ir * (V[:,r] outer W[:,r]) approximates T
                # This is a linear system for each i?
                # No, U is a matrix.
                # T (N, M, P) = sum_r U[:,r,None,None] * V[:,None,r,None] * W[:,None,None,r]
                # Reshape T to (N, M*P)
                # T_mat = T.reshape(N, M*P)
                # A_r = (V[:,r] * W[:,r]).reshape(M*P)
                # Then T_mat = U @ A_r^T ? No.
                # T_mat[i, :] = sum_r U[i,r] * A_r
                # So T_mat = U @ A, where A is (R, M*P) with A[r, :] = A_r
                # So U = T_mat @ A^T @ (A A^T)^{-1}
                
                A = np.zeros((R, M * P))
                for r in range(R):
                    A[r, :] = (V[:, r] * W[:, r]).flatten()
                
                T_mat = T.reshape(N, M * P)
                # Solve U = T_mat @ A.T @ inv(A @ A.T)
                # Use lstsq for stability
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A.T, T_mat.T, rcond=None)
                    U = U_new.T
                except:
                    pass

                # Update V
                # T_mat (M, N*P)
                # B_r = (U[:,r] * W[:,r]).flatten()
                B = np.zeros((R, N * P))
                for r in range(R):
                    B[r, :] = (U[:, r] * W[:, r]).flatten()
                T_mat_V = T.transpose(1, 0, 2).reshape(M, N * P)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(B.T, T_mat_V.T, rcond=None)
                    V = V_new.T
                except:
                    pass

                # Update W
                # T_mat (P, N*M)
                # C_r = (U[:,r] * V[:,r]).flatten()
                C = np.zeros((R, N * M))
                for r in range(R):
                    C[r, :] = (U[:, r] * V[:, r]).flatten()
                T_mat_W = T.transpose(2, 0, 1).reshape(P, N * M)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(C.T, T_mat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass

            # After ALS, check residual
            # If residual is small, try snapping
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = np.linalg.norm(T - T_recon)
            
            if residual < 1e-4:
                U_snap, V_snap, W_snap = snap_to_half(U, V, W)
                if check_exact(U_snap, V_snap, W_snap):
                    # Found a valid exact decomposition
                    # Convert to lists
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # If snap fails, maybe try rounding to integers?
                # Or just accept the float solution if the evaluator allows floats?
                # Prompt says "integer/half-integer entries".
                # So we must snap.
                # If snap fails, maybe the solution is not on the 0.5 grid?
                # But the prompt implies the optimal solution has such entries.
                # We can try snapping to 0.25 or 1/3?
                # Let's stick to 0.5 for now, it's the most common for these small tensors.
                pass

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
