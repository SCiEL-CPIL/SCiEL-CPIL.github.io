# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    start_time = time.time()
    deadline = start_time + 145  # 145 seconds for search (5s margin)
    
    # Build the matrix multiplication tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I_idx = i*m + j
                J_idx = j*p + k
                K_idx = k*n + i
                T[I_idx, J_idx, K_idx] = 1.0

    def verify(U, V, W):
        """Verify that U, V, W exactly decompose T with integer/half-integer entries."""
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        U_round = np.round(U2)
        V_round = np.round(V2)
        W_round = np.round(W2)
        
        # Check if rounding changed the values significantly
        if not (np.allclose(U2, U_round, atol=1e-9) and 
                np.allclose(V2, V_round, atol=1e-9) and 
                np.allclose(W2, W_round, atol=1e-9)):
            return False
            
        U_check = U_round / 2.0
        V_check = V_round / 2.0
        W_check = W_round / 2.0
        
        # Reconstruct T
        T_approx = np.einsum('ir, jr, kr -> ijk', U_check, V_check, W_check)
        if np.allclose(T_approx, T, atol=1e-9):
            return True
        return False

    def try_find_solution(R, time_limit, num_restarts=20):
        for restart in range(num_restarts):
            if time.time() > time_limit:
                return None
            
            # Initialize with random integers in {-1, 0, 1}
            # Using different seeds for diversity
            np.random.seed(12345 + restart * 7919)
            U = np.random.randint(-1, 2, (n*m, R)).astype(float)
            V = np.random.randint(-1, 2, (m*p, R)).astype(float)
            W = np.random.randint(-1, 2, (p*n, R)).astype(float)
            
            # ALS Loop
            for it in range(50):
                if time.time() > time_limit:
                    return None
                
                # Update U
                # Precompute V @ W^T? No, V is (mp, R), W is (pn, R).
                # We need for each i: T[i, :, :] = sum_r U[i, r] * (V[:, r] * W[:, r]^T)
                # So A_U[i] is not constant.
                # However, we can batch the least squares?
                # T is sparse.
                
                # Efficient update:
                # For each i, we solve: sum_r U[i, r] * M_r = T[i, :, :]
                # where M_r = np.outer(V[:, r], W[:, r])
                # This is a linear system: M_matrix @ U[i, :] = T[i, :, :].flatten()
                # M_matrix is (mp*pn, R)
                
                M_U = np.einsum('jr, kr -> jkr', V, W).reshape(-1, R)
                for i in range(n*m):
                    b_U = T[i, :, :].flatten()
                    # Solve M_U @ u_i = b_U
                    u_i, _, _, _ = np.linalg.lstsq(M_U, b_U, rcond=None)
                    U[i, :] = u_i

                # Update V
                M_V = np.einsum('ir, kr -> ikr', U, W).reshape(-1, R)
                for j in range(m*p):
                    b_V = T[:, j, :].flatten()
                    v_j, _, _, _ = np.linalg.lstsq(M_V, b_V, rcond=None)
                    V[j, :] = v_j

                # Update W
                M_W = np.einsum('ir, jr -> ijr', U, V).reshape(-1, R)
                for k in range(p*n):
                    b_W = T[:, :, k].flatten()
                    w_k, _, _, _ = np.linalg.lstsq(M_W, b_W, rcond=None)
                    W[k, :] = w_k
                
                # Check error
                T_approx = np.einsum('ir, jr, kr -> ijk', U, V, W)
                err = np.linalg.norm(T_approx - T)
                
                # Snap to half-integers if close
                if err < 1e-6:
                    if verify(U, V, W):
                        return U, V, W
                        
                if err < 1e-12:
                    # Very close, try snapping harder
                    U_snap = np.round(U * 2) / 2.0
                    V_snap = np.round(V * 2) / 2.0
                    W_snap = np.round(W * 2) / 2.0
                    if verify(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap

        return None

    # Try rank 28 first (hypothesis from directive)
    res = try_find_solution(28, deadline, num_restarts=10)
    if res is not None:
        U, V, W = res
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Try rank 47 (known optimal or near-optimal)
    res = try_find_solution(47, deadline, num_restarts=20)
    if res is not None:
        U, V, W = res
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback to Schoolbook rank 48
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
# EVOLVE-BLOCK-END
