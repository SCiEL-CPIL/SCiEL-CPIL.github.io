import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 96
    
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid i,j,k
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    def verify(U, V, W):
        """Check if the decomposition exactly reproduces T"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(S, T, atol=1e-10)
    
    def round_to_half_integer(arr):
        """Round array to nearest half-integer (multiple of 0.5)"""
        return np.round(arr * 2) / 2
    
    def als_decompose(rank, init_U=None, init_V=None, init_W=None, max_iters=200, tol=1e-12):
        """ALS tensor decomposition with structured initialization"""
        if init_U is not None:
            U = init_U.astype(float).copy()
            V = init_V.astype(float).copy()
            W = init_W.astype(float).copy()
        else:
            # Structured initialization based on matrix multiplication pattern
            np.random.seed(42)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Optimize U: for each column r, U[:,r] should satisfy
            # sum_r U[:,r] outer V[:,r] outer W[:,r] = T
            # Fix V and W, solve for U
            # T = sum_r U_r outer (V_r outer W_r)
            # Reshape: T_flat[i, j*p+k, k*n+i] ... 
            # Actually, let's use matmul form
            # For fixed V,W: T_ijk = sum_r U_ir * V_jr * W_kr
            # This is: T = U @ (V^T * W^T) in some sense
            # Better: for each (j,k) pair, T[:, j, k] = U * (V[:,r]*W[:,r] summed over r)
            # Actually: T_ijk = sum_r U_ir V_jr W_kr
            # = sum_r U_ir * (V_jr * W_kr)
            # Let M_r = V_r outer W_r, shape (m*p, p*n)
            # Then T = sum_r U_r outer M_r
            # Reshape T to (n*m, m*p*p*n)
            T_reshaped = T.reshape(n * m, -1)
            # For each r, M_r flattened is (V_r outer W_r).flatten()
            # We want: T_reshaped = U @ M^T where M is (rank, m*p*p*n)
            # Build M: each row r is (V[:,r] outer W[:,r]).flatten()
            M = np.zeros((rank, m * p * p * n))
            for r in range(rank):
                M[r, :] = (V[:, r] * W[:, r][None, :]).flatten()
            # Solve: T_reshaped = U @ M^T
            # U = T_reshaped @ pinv(M^T)
            try:
                U_new = T_reshaped @ np.linalg.pinv(M.T)
            except:
                break
            U = U_new
            
            # Optimize V
            T_reshaped2 = T.transpose(1, 0, 2).reshape(m * p, -1)
            M2 = np.zeros((rank, n * m * p * n))
            for r in range(rank):
                M2[r, :] = (U[:, r] * W[:, r][None, :]).flatten()
            try:
                V_new = T_reshaped2 @ np.linalg.pinv(M2.T)
            except:
                break
            V = V_new
            
            # Optimize W
            T_reshaped3 = T.transpose(2, 0, 1).reshape(p * n, -1)
            M3 = np.zeros((rank, n * m * m * p))
            for r in range(rank):
                M3[r, :] = (U[:, r] * V[:, r][None, :]).flatten()
            try:
                W_new = T_reshaped3 @ np.linalg.pinv(M3.T)
            except:
                break
            W = W_new
            
            # Check convergence
            S = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = np.linalg.norm(S - T)
            if residual < tol:
                break
        
        return U, V, W, residual
    
    # Try to find lower rank decompositions
    # Strategy 1: Try ranks that are factors or near-factors of 96
    # Strategy 2: Use structured initialization
    
    # Try a few target ranks
    target_ranks = [95, 94, 92, 90, 88, 84, 80, 72, 64, 48]
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple initializations
        for init_seed in range(3):
            if time.time() > deadline:
                break
            
            # Structured init: start from schoolbook and try to compress
            if init_seed == 0:
                # Random init
                U, V, W, res = als_decompose(rank)
            elif init_seed == 1:
                # Init from schoolbook by averaging columns
                U_school = np.zeros((n * m, rank_schoolbook))
                V_school = np.zeros((m * p, rank_schoolbook))
                W_school = np.zeros((p * n, rank_schoolbook))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_school[i * m + j, col] = 1
                            V_school[j * p + k, col] = 1
                            W_school[k * n + i, col] = 1
                            col += 1
                
                # Average groups of columns to get lower rank init
                group_size = rank_schoolbook // rank
                U_init = np.zeros((n * m, rank))
                V_init = np.zeros((m * p, rank))
                W_init = np.zeros((p * n, rank))
                for r in range(rank):
                    start = r * group_size
                    end = start + group_size
                    U_init[:, r] = U_school[:, start:end].mean(axis=1)
                    V_init[:, r] = V_school[:, start:end].mean(axis=1)
                    W_init[:, r] = W_school[:, start:end].mean(axis=1)
                
                U, V, W, res = als_decompose(rank, U_init, V_init, W_init)
            else:
                # Block-structured init: split p=8 into two blocks of 4
                # C = A*B1 + A*B2 where B1 is 4x4, B2 is 4x4
                # Rank(3,4,4) * 2 as upper bound, but try to share
                U_init = np.zeros((n * m, rank))
                V_init = np.zeros((m * p, rank))
                W_init = np.zeros((p * n, rank))
                
                # First half: for k in 0..3
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(4):
                            if col < rank:
                                U_init[i * m + j, col] = 1
                                V_init[j * p + k, col] = 1
                                W_init[k * n + i, col] = 1
                                col += 1
                # Second half: for k in 4..7
                for i in range(n):
                    for j in range(m):
                        for k in range(4, 8):
                            if col < rank:
                                U_init[i * m + j, col] = 1
                                V_init[j * p + k, col] = 1
                                W_init[k * n + i, col] = 1
                                col += 1
                
                U, V, W, res = als_decompose(rank, U_init, V_init, W_init)
            
            # Check if we got an exact solution
            if res < 1e-10:
                # Round to half-integers and verify
                U_rounded = round_to_half_integer(U)
                V_rounded = round_to_half_integer(V)
                W_rounded = round_to_half_integer(W)
                
                if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        break
                else:
                    # Try without rounding - maybe it's already integer
                    if verify(U.tolist(), V.tolist(), W.tolist()):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
                            break
    
    # Final check: if we found a better rank, return it; else schoolbook
    if best_rank < rank_schoolbook:
        return best_result
    else:
        return best_result
# EVOLVE-BLOCK-END
