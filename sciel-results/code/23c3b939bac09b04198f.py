"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook fallback (rank 24)
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    def verify(U, V, W, n, m, p):
        """Check if the decomposition is valid with exact integer/half-integer arithmetic."""
        rank = len(U[0])
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Check if entries are half-integers
        if not np.allclose(U_np * 2, np.round(U_np * 2)):
            return False
        if not np.allclose(V_np * 2, np.round(V_np * 2)):
            return False
        if not np.allclose(W_np * 2, np.round(W_np * 2)):
            return False
            
        # Construct tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        U_t = U_np.reshape(n, m, -1)
        V_t = V_np.reshape(m, p, -1)
        W_t = W_np.reshape(p, n, -1)
        
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            if np.allclose(T, np.ones((n,m,p)), atol=1e-9):
                return True
        except:
            return False
        return False

    def round_to_half_int(arr):
        """Round a numpy array to the nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def try_rank(r, max_als_iters=500, max_restarts=20):
        """Try to find a valid rank-r decomposition using vectorized ALS and rounding."""
        if time.time() > deadline:
            return None
            
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n, m, r)
            V = np.random.randn(m, p, r)
            W = np.random.randn(p, n, r)
            
            for iteration in range(max_als_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = V[j, :, :] * W[:, i, :]
                        try:
                            U[i, j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = U[:, j, :] * W[k, :, :]
                        try:
                            V[j, k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = U[i, :, :] * V[:, k, :]
                        try:
                            W[k, i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        except:
                            pass
                
                if iteration % 10 == 0:
                    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                    err = np.sum((T - 1.0)**2)
                    if err < 1e-4:
                        # Try to round
                        # 1. Simple rounding
                        U_r = round_to_half_int(U)
                        V_r = round_to_half_int(V)
                        W_r = round_to_half_int(W)
                        
                        if verify(U_r.reshape(n*m, r).tolist(), V_r.reshape(m*p, r).tolist(), W_r.reshape(p*n, r).tolist(), n, m, p):
                            return {"U": U_r.reshape(n*m, r).tolist(), "V": V_r.reshape(m*p, r).tolist(), "W": W_r.reshape(p*n, r).tolist()}
                        
                        # 2. Try scaling columns to improve rounding
                        for _ in range(20):
                            U_s = U.copy()
                            V_s = V.copy()
                            W_s = W.copy()
                            
                            for c in range(r):
                                s = np.random.choice([0.5, 1.0, 2.0, 0.25, 4.0])
                                U_s[:, :, c] *= s
                                V_s[:, :, c] /= s
                            
                            U_rs = round_to_half_int(U_s)
                            V_rs = round_to_half_int(V_s)
                            W_rs = round_to_half_int(W_s)
                            
                            if verify(U_rs.reshape(n*m, r).tolist(), V_rs.reshape(m*p, r).tolist(), W_rs.reshape(p*n, r).tolist(), n, m, p):
                                return {"U": U_rs.reshape(n*m, r).tolist(), "V": V_rs.reshape(m*p, r).tolist(), "W": W_rs.reshape(p*n, r).tolist()}
                                
                        # 3. Local search if err is very small
                        if err < 1e-6:
                            for _ in range(100):
                                U_p = round_to_half_int(U)
                                V_p = round_to_half_int(V)
                                W_p = round_to_half_int(W)
                                
                                U_flat = U_p.flatten()
                                V_flat = V_p.flatten()
                                W_flat = W_p.flatten()
                                
                                U_flat[np.random.choice(len(U_flat))] += np.random.choice([-0.5, 0.5])
                                V_flat[np.random.choice(len(V_flat))] += np.random.choice([-0.5, 0.5])
                                W_flat[np.random.choice(len(W_flat))] += np.random.choice([-0.5, 0.5])
                                
                                U_p = U_flat.reshape(n, m, r)
                                V_p = V_flat.reshape(m, p, r)
                                W_p = W_flat.reshape(p, n, r)
                                
                                if verify(U_p.reshape(n*m, r).tolist(), V_p.reshape(m*p, r).tolist(), W_p.reshape(p*n, r).tolist(), n, m, p):
                                    return {"U": U_p.reshape(n*m, r).tolist(), "V": V_p.reshape(m*p, r).tolist(), "W": W_p.reshape(p*n, r).tolist()}
            if time.time() > deadline:
                break
                
        return None

    # Try rank 11
    result = try_rank(11)
    if result is not None:
        best_result = result
        best_rank = 11
        return best_result
    
    # Try rank 12
    result = try_rank(12)
    if result is not None:
        best_result = result
        best_rank = 12
        return best_result

    # If random search fails, return the best we have (schoolbook)
    return best_result
# EVOLVE-BLOCK-END
