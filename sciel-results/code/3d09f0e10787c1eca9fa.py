"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Build the target tensor T of shape (dim_A, dim_B, dim_C)
    # T[i, j, k] = 1 if C[k] depends on A[i] and B[j] in the product C = A * B
    # A is (n, m), B is (m, p), C is (n, p)
    # Index mapping:
    # A index i: row i_n, col i_m -> i = i_n * m + i_m
    # B index j: row j_m, col j_p -> j = j_m * p + j_p
    # C index k: row k_n, col k_p -> k = k_n * p + k_p
    # C[k_n, k_p] = sum_m A[i_n, i_m] * B[i_m, k_p]
    # So T[i, j, k] = 1 if i_n == k_n and i_m == j_m and j_p == k_p
    T = np.zeros((dim_A, dim_B, dim_C))
    for i_n in range(n):
        for i_m in range(m):
            for k_p in range(p):
                i = i_n * m + i_m
                j = i_m * p + k_p
                k = i_n * p + k_p
                T[i, j, k] = 1.0
                
    def verify_decomposition(U, V, W, rank):
        # U: (dim_A, rank), V: (dim_B, rank), W: (dim_C, rank)
        # Reconstruct tensor: T_recon[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # This is equivalent to: for each r, outer(U[:, r], V[:, r]) * W[:, r]
        # More efficiently: T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        return err
        
    def search_rank(rank, time_limit=10):
        start_time = time.time()
        rng = np.random.RandomState(42)
        
        # Try a few random initializations
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
                
            U = rng.randn(dim_A, rank)
            V = rng.randn(dim_B, rank)
            W = rng.randn(dim_C, rank)
            
            # Scale to avoid numerical issues
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            prev_err = verify_decomposition(U, V, W, rank)
            
            # Alternating Least Squares
            for step in range(100):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update U: fix V, W
                # T_recon[i, j, k] = U[i, r] * (V[j, r] * W[k, r])
                # For fixed i, we want U[i, :] to minimize sum_{j,k} (T[i,j,k] - U[i,:] . M[j,k,:])^2
                # This is a linear least squares problem.
                # Let M_r[j,k] = V[j,r] * W[k,r]
                # We want U[i,r] such that U[i,:] @ M_r[j,k] approx T[i,j,k]
                # Vectorize: U[i,:] @ M_flat[j,k] = T[i, j, k]
                # Actually, we can solve for all i at once.
                # Let M be a matrix of shape (dim_B * dim_C, rank) where M[(j,k), r] = V[j,r] * W[k,r]
                M = np.einsum('jr,kr->jkr', V, W).reshape(dim_B * dim_C, rank)
                T_flat = T.reshape(dim_A, dim_B * dim_C)
                # Solve U @ M.T = T_flat
                # U = T_flat @ M.T @ (M @ M.T)^-1
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                    
                # Update V
                M = np.einsum('ir,kr->ikr', U, W).reshape(dim_A * dim_C, rank)
                T_flat = T.reshape(dim_B, dim_A * dim_C).T # T[i,j,k] -> T[j, i,k]
                # Wait, T_flat for V update:
                # T[i,j,k] = V[j,r] * (U[i,r] * W[k,r])
                # For fixed j, V[j,:] @ N[i,k,:] = T[i,j,k]
                # N[i,k,r] = U[i,r] * W[k,r]
                N = np.einsum('ir,kr->ikr', U, W).reshape(dim_A * dim_C, rank)
                T_flat_V = T.transpose(1, 0, 2).reshape(dim_B, dim_A * dim_C)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(N, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                    
                # Update W
                M = np.einsum('ir,jr->ijr', U, V).reshape(dim_A * dim_B, rank)
                T_flat_W = T.transpose(2, 0, 1).reshape(dim_C, dim_A * dim_B)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                    
                err = verify_decomposition(U, V, W, rank)
                if err < 1e-6:
                    return U, V, W
                if err > prev_err:
                    # Diverging, break
                    break
                prev_err = err
                
        return None

    # Try rank 12
    best_result = None
    for rank in [12, 13]:
        res = search_rank(rank, time_limit=20)
        if res is not None:
            U, V, W = res
            # Convert to list of lists
            U_list = [[U[i, r] for r in range(rank)] for i in range(dim_A)]
            V_list = [[V[i, r] for r in range(rank)] for i in range(dim_B)]
            W_list = [[W[i, r] for r in range(rank)] for i in range(dim_C)]
            # Check if entries are close to integers/half-integers
            # The task requires integer/half-integer entries.
            # Numerical solutions might not be exact.
            # We need to round or verify if they are close.
            is_valid = True
            for mat in [U_list, V_list, W_list]:
                for row in mat:
                    for val in row:
                        # Check if close to 0.5 * integer
                        k = round(val * 2)
                        if abs(val - k * 0.5) > 1e-3:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            if is_valid:
                # Snap to half-integers
                U_list = [[round(val * 2) * 0.5 for val in row] for row in U_list]
                V_list = [[round(val * 2) * 0.5 for val in row] for row in V_list]
                W_list = [[round(val * 2) * 0.5 for val in row] for row in W_list]
                # Final verification
                U_np = np.array(U_list)
                V_np = np.array(V_list)
                W_np = np.array(W_list)
                if verify_decomposition(U_np, V_np, W_np, rank) < 1e-6:
                    return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to rank 14
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for B1 (cols 0,1): B indices 0,1,4,5; W indices 0,1,2,3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Strassen for B2 (cols 2,3): B indices 2,3,6,7; W indices 4,5,6,7
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
