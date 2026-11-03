import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    n, m, p = 2, 3, 6
    
    # Build target tensor T for 2x3x6
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Strategy: Decompose 2x3x6 into two 2x3x3 blocks.
    # Find rank-minimal decomposition for 2x3x3 (known to be 15).
    # If found, rank for 2x3x6 is 2 * 15 = 30.
    
    n2, m2, p2 = 2, 3, 3
    T233 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
    # We want to find U233, V233, W233 of rank R233=15
    # such that T233[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    
    best_233 = None
    
    # Use Alternating Least Squares (ALS) with rounding
    # This is more robust for integer/half-integer solutions than gradient descent
    
    R233 = 15
    rows_u = n2 * m2  # 6
    rows_v = m2 * p2  # 9
    rows_w = p2 * n2  # 6
    
    # Precompute matrices for linear solves in ALS
    # T is flattened.
    # For fixed V, W, we solve for U.
    # T_flat = U @ (V kron W)_mat
    # (V kron W)_mat has shape (rows_v * rows_w, R) ? No.
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Let's fix V, W. For each i, we have a linear system for U[i, :].
    # sum_r U[i, r] * (V[j, r] * W[k, r]) = T[i, j, k]
    # This is not a simple matrix multiply for U because V and W are coupled in the index r.
    # However, we can vectorize.
    # Let M_VW be a matrix of size (rows_v * rows_w, R) where M_VW[j, k, r] = V[j, r] * W[k, r].
    # Then T_vec[i, j, k] = sum_r U[i, r] * M_VW[j, k, r].
    # This is still not a standard linear system U * A = B because A depends on j,k.
    # Actually, for each i, the equation is:
    # sum_r U[i, r] * (V[:, r] * W[:, r]) = T[i, :, :]
    # Let Z_r = V[:, r] * W[:, r]. This is a vector of length rows_v * rows_w.
    # Then T[i, :, :].flatten() = sum_r U[i, r] * Z_r.
    # So T_flat[i, :] = U[i, :] @ Z.T
    # Where Z is a matrix of shape (R, rows_v * rows_w).
    # Z[r, idx(j,k)] = V[j, r] * W[k, r].
    
    def generate_Z(V, W):
        # V: (rows_v, R), W: (rows_w, R)
        # Z: (R, rows_v * rows_w)
        # Z[r, j*rows_w + k] = V[j, r] * W[k, r]
        Z = np.zeros((R233, rows_v * rows_w))
        for r in range(R233):
            # V[:, r] is (rows_v, 1), W[:, r] is (rows_w, 1)
            # Outer product is (rows_v, rows_w)
            outer = np.outer(V[:, r], W[:, r])
            Z[r, :] = outer.flatten()
        return Z

    def solve_U(V, W, T_target):
        Z = generate_Z(V, W) # (R, rows_v*rows_w)
        # T_target: (rows_u, rows_v, rows_w)
        T_flat = T_target.flatten().reshape(rows_u, -1)
        # T_flat[i, :] = U[i, :] @ Z.T
        # U[i, :] = T_flat[i, :] @ inv(Z.T)  -> use lstsq
        # U = T_flat @ pinv(Z.T)
        # Z.T is (rows_v*rows_w, R)
        # We want U (rows_u, R) such that U @ Z.T = T_flat
        # This is a linear system for U.
        # U = lstsq(Z.T, T_flat.T).T ?
        # Let A = Z.T (size rows_v*rows_w x R)
        # Let B = T_flat.T (size rows_v*rows_w x rows_u)
        # We want A @ U.T = B
        # U.T = lstsq(A, B)
        try:
            U_sol, residuals, rank, sv = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return U_sol.T
        except:
            return None

    def solve_V(U, W, T_target):
        # Symmetric to solve_U but with roles swapped?
        # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # For fixed U, W.
        # T[i, j, k] = V[j, r] * (sum_r U[i, r] W[k, r])
        # Let Y_r = U[:, r] * W[:, r] ? No.
        # For each j, we have T[:, j, :] = sum_r V[j, r] * (U[:, r] * W[:, r])
        # Let Z_r = U[:, r] * W[:, r]. Shape (rows_u, rows_w).
        # T[:, j, :].flatten() = sum_r V[j, r] * Z_r.flatten()
        # So T_flat_V (size rows_v, rows_u*rows_w) = V (rows_v, R) @ Z.T (R, rows_u*rows_w)
        Z = np.zeros((R233, rows_u * rows_w))
        for r in range(R233):
            outer = np.outer(U[:, r], W[:, r])
            Z[r, :] = outer.flatten()
        
        T_flat = T_target.transpose(1, 0, 2).flatten().reshape(rows_v, -1)
        # T_flat_V = V @ Z.T
        # V = lstsq(Z.T, T_flat_V.T).T
        try:
            V_sol, residuals, rank, sv = np.linalg.lstsq(Z.T, T_flat_V.T, rcond=None)
            return V_sol.T
        except:
            return None

    def solve_W(U, V, T_target):
        # T[i, j, k] = W[k, r] * (sum_r U[i, r] V[j, r])
        # For each k, T[:, :, k].flatten() = sum_r W[k, r] * (U[:, r] * V[:, r]).flatten()
        Z = np.zeros((R233, rows_u * rows_v))
        for r in range(R233):
            outer = np.outer(U[:, r], V[:, r])
            Z[r, :] = outer.flatten()
            
        T_flat = T_target.transpose(2, 0, 1).flatten().reshape(rows_w, -1)
        # T_flat_W = W @ Z.T
        try:
            W_sol, residuals, rank, sv = np.linalg.lstsq(Z.T, T_flat_W.T, rcond=None)
            return W_sol.T
        except:
            return None

    # ALS Loop
    for restart in range(50):
        if time.time() > deadline - 20:
            break
        
        # Initialize
        U = np.random.randn(rows_u, R233) * 0.5
        V = np.random.randn(rows_v, R233) * 0.5
        W = np.random.randn(rows_w, R233) * 0.5
        
        converged = False
        for it in range(100):
            if time.time() > deadline - 20:
                break
                
            U_new = solve_U(V, W, T233)
            if U_new is None: break
            V_new = solve_V(U_new, W, T233)
            if V_new is None: break
            W_new = solve_W(U_new, V_new, T233)
            if W_new is None: break
            
            # Check convergence
            diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            
            if diff < 1e-9:
                converged = True
                break
        
        if not converged:
            continue
            
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        if np.allclose(T_check, T233, atol=1e-9):
            best_233 = (U_rounded, V_rounded, W_rounded)
            break

    if best_233:
        U233, V233, W233 = best_233
        # Embed into 2x3x6
        # The 2x3x6 multiplication is two independent 2x3x3 multiplications.
        # A (2x3) * B (3x6) -> C (2x6)
        # B is split into B1 (3x3) and B2 (3x3).
        # C1 = A * B1, C2 = A * B2.
        # Indices:
        # U236: (6, 30). Columns 0-14 for block 1, 15-29 for block 2.
        # V236: (18, 30). Rows 0-8 for block 1 (B1), 9-17 for block 2 (B2).
        # W236: (12, 30). Rows 0-5 for block 1 (C1), 6-11 for block 2 (C2).
        
        R236 = 2 * R233
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        # Block 1
        U236[:, :R233] = U233
        V236[:m2*p2, :R233] = V233
        W236[:p2*n2, :R233] = W233
        
        # Block 2
        U236[:, R233:] = U233
        V236[m2*p2:, R233:] = V233
        W236[p2*n2:, R233:] = W233
        
        # Verify full tensor
        T_final = np.einsum('ir,jr,kr->ijk', U236, V236, W236)
        if np.allclose(T_final, T, atol=1e-9):
            return {
                "U": U236.tolist(),
                "V": V236.tolist(),
                "W": W236.tolist()
            }

    # Fallback to schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
