import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Schoolbook fallback
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a rank 16 decomposition for 3x3x2
    # If found, combine two to get rank 32 for 3x3x4
    try:
        U32, V32, W32 = find_rank_decomposition(3, 3, 2, 16, time_limit=deadline - time.time() - 10)
        if U32 is not None:
            # Combine
            U_full = U32
            V_full = []
            W_full = []
            # V and W need to be extended for the second half of columns
            # V indices: j*p+k. For p=2, indices 0..5.
            # For p=4, indices 0..11.
            # The first 16 terms use V32 (for k=0,1).
            # The next 16 terms use V32 shifted (for k=2,3).
            # V32 is (m*p32, rank32) = (3*2, 16) = (6, 16).
            # We need V_full (m*p_full, rank_full) = (3*4, 32) = (12, 32).
            
            # V_full[:, 0:16] corresponds to k=0,1.
            # V_full[:, 16:32] corresponds to k=2,3.
            # The structure of V is V[j*p+k, r].
            # For r in 0..15, k is 0 or 1.
            # For r in 16..31, k is 2 or 3.
            # V32[j*2+k, r] for r in 0..15.
            # We need V_full[j*4+k, r].
            
            V_full = [[0] * 32 for _ in range(12)]
            W_full = [[0] * 32 for _ in range(12)] # p*n = 4*3 = 12
            
            for j in range(3):
                for k in range(2):
                    # Map to V32 index
                    idx_v32 = j * 2 + k
                    for r in range(16):
                        V_full[j * 4 + k][r] = V32[idx_v32][r]
                        V_full[j * 4 + k + 2][r + 16] = V32[idx_v32][r]
                        
            for k in range(2):
                for i in range(3):
                    idx_w32 = k * 3 + i
                    for r in range(16):
                        W_full[k * 3 + i][r] = W32[idx_w32][r]
                        W_full[k * 3 + i + 3][r + 16] = W32[idx_w32][r]
                        
            # U is the same
            return {"U": U_full, "V": V_full, "W": W_full}
            
    except Exception as e:
        pass
        
    return schoolbook(n, m, p)

def find_rank_decomposition(n, m, p, rank, time_limit):
    # Initialize random
    U = np.random.randint(-1, 2, (n*m, rank))
    V = np.random.randint(-1, 2, (m*p, rank))
    W = np.random.randint(-1, 2, (p*n, rank))
    
    # Target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We flatten to vectors for ALS
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    start = time.time()
    while time.time() - start < time_limit:
        # ALS step
        for _ in range(10):
            # Update U
            for i in range(n*m):
                # Construct system for U[i, :]
                # Sum_r U[i,r] V[j*p+k,r] W[k*n+i,r] = 1
                # This is complex. Let's use gradient descent on the error.
                pass
                
        # Instead of complex ALS, use simple gradient descent on the error
        # Error = || T - sum_r U_r V_r W_r ||^2
        # This is expensive.
        
        # Let's try a simpler heuristic: Random restarts and rounding.
        # Given the small size, maybe we can just try random integer matrices and check?
        # No, probability is low.
        
        # Let's use a known property: The rank of 3x3x2 is 16.
        # I will hard-code a known solution if I can find one online? No.
        
        # Let's try to solve the linear system for a fixed U, V.
        # Given U, V, find W.
        # For each r, W[:, r] is a vector.
        # The equation is sum_r U[i,r] V[j,r] W[k,i,r] = 1.
        # This is not linear in W if we consider all r simultaneously?
        # Actually, for a fixed r, the term is U[i,r] V[j,r] W[k,i,r].
        # The sum is over r.
        # This is a system of equations for the entries of W.
        # Number of equations: n*m*p = 18.
        # Number of unknowns in W: p*n*rank = 6*16 = 96.
        # We have more unknowns than equations. We can find a solution.
        # But we need it to be integer.
        
        # Strategy:
        # 1. Random U, V (integers).
        # 2. Solve for W (real numbers) using least squares.
        # 3. Round W to nearest integer/half-integer.
        # 4. Check error.
        # 5. If error 0, return.
        
        U = np.random.randint(-2, 3, (n*m, rank))
        V = np.random.randint(-2, 3, (m*p, rank))
        
        # Build A matrix for W
        # Variables: W[k*n+i, r]
        # Eq for (i,j,k): sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
        
        num_eqs = n * m * p
        num_vars = p * n * rank
        
        A_mat = np.zeros((num_eqs, num_vars))
        b_vec = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        var_idx = (k * n + i) * rank + r
                        A_mat[eq_idx, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                    eq_idx += 1
                    
        # Solve least squares
        try:
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
            
        W_sol = W_flat.reshape(p * n, rank)
        
        # Round to nearest integer
        W_int = np.round(W_sol)
        
        # Check error
        # Reconstruct tensor
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W_int[k * n + i, r]
                    error += (val - 1) ** 2
                    
        if error == 0:
            return U.tolist(), V.tolist(), W_int.tolist()
            
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.tolist(), V.tolist(), W_int.tolist()
            
    return None, None, None
