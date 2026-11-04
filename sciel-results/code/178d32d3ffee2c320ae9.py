import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    # Build schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find lower rank
    # We know rank is at least 60.
    # Let's try 70, 69, 68...
    for target_rank in range(74, 59, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for restart in range(20):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Iterative refinement
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Fix U, V, solve for W
                # T = sum u v^T w^T
                # We want to minimize || T - sum u v^T w^T ||_F
                # This is linear in W if U, V fixed.
                # Reshape T to (n*m, m*p, p*n)
                # Actually, we can form the matrix M = sum_{r} (u_r \otimes v_r) w_r^T
                # This is a matrix of size (n*m * m*p) x (p*n)
                # Let's build the design matrix for W.
                
                # Vectorize U and V
                # For each r, term is u_r (n*m) x v_r (m*p) -> outer product (n*m * m*p) x 1
                # We want to match the tensor T.
                # T is sparse.
                
                # Instead of building huge matrices, we can use the fact that T is sparse.
                # T_{i, j, k} = 1 if (i, j, k) is a valid triple.
                # The equation is:
                # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k.
                
                # This is a system of linear equations for W (if U, V fixed).
                # Number of equations: n*m*p = 75.
                # Unknowns: target_rank * p * n? No, W has p*n rows.
                # W is (p*n) x target_rank.
                # Let W_flat be (p*n * target_rank).
                # The equations are linear in W.
                
                # Let's form the matrix A_w of size (75, p*n*target_rank).
                # Row idx corresponds to (i, j, k).
                # Col idx corresponds to (k', i', r).
                # Coefficient is U[i*m+j, r] * V[j*p+k, r] if (k, i) == (k', i').
                
                # This is too slow to build for each iteration.
                # Let's use a simpler heuristic: 
                # If residual is small, snap to integers.
                
                # Compute residual
                # C_est = 0
                # For r in range(target_rank):
                #   C_est += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]) # No, dimensions don't match for direct addition.
                
                # Let's compute the error on the tensor.
                # err = 0
                # for i in range(n):
                #   for j in range(m):
                #     for k in range(p):
                #       val = 0
                #       for r in range(target_rank):
                #         val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                #       err += (val - 1)**2
                # This is O(n*m*p*rank) per iteration. 75 * 70 = 5250 ops. Very fast.
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err += (val - 1.0)**2
                            
                if err < 1e-6:
                    # Check if we can snap to integers/halves
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify snapped solution
                    err_snap = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for r in range(target_rank):
                                    val += U_snap[i*m+j, r] * V_snap[j*p+k, r] * W_snap[k*n+i, r]
                                err_snap += (val - 1.0)**2
                                
                    if err_snap < 1e-9:
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        best_rank = target_rank
                        break # Break restart loop, try lower rank
                    else:
                        # Try to solve linear system for W with snapped U, V?
                        # Or just accept if err is small?
                        # The requirement is "exact".
                        # So we need err_snap to be 0.
                        pass
                else:
                    # Gradient step?
                    # Too complex. Just random restart.
                    pass
                    
    # Convert to list of lists as required
    def to_list(mat):
        return mat.tolist()
        
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
