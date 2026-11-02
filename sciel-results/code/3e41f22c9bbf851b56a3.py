import time
import numpy as np
import random

DIMENSIONS = [3, 3, 3]

def _get_strassen_core():
    """
    Returns the 7 Strassen terms for the 2x2 submatrix (indices 0,1).
    Returns U, V, W as lists of lists (9x7 each).
    """
    n, m, p = 3, 3, 3
    U = np.zeros((9, 7))
    V = np.zeros((9, 7))
    W = np.zeros((9, 7))
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # r=0: m1 = (a00 + a11)(b00 + b11)
    U[idx_u(0,0), 0] = 1; U[idx_u(1,1), 0] = 1
    V[idx_v(0,0), 0] = 1; V[idx_v(1,1), 0] = 1
    W[idx_w(0,0), 0] = 1; W[idx_w(1,1), 0] = 1
    
    # r=1: m2 = (a10 + a11)b00
    U[idx_u(1,0), 1] = 1; U[idx_u(1,1), 1] = 1
    V[idx_v(0,0), 1] = 1
    W[idx_w(0,1), 1] = 1; W[idx_w(1,1), 1] = -1
    
    # r=2: m3 = a00(b01 - b11)
    U[idx_u(0,0), 2] = 1
    V[idx_v(0,1), 2] = 1; V[idx_v(1,1), 2] = -1
    W[idx_w(1,0), 2] = 1; W[idx_w(1,1), 2] = 1
    
    # r=3: m4 = a11(b10 - b00)
    U[idx_u(1,1), 3] = 1
    V[idx_v(1,0), 3] = 1; V[idx_v(0,0), 3] = -1
    W[idx_w(0,0), 3] = 1; W[idx_w(0,1), 3] = 1
    
    # r=4: m5 = (a00 + a01)b11
    U[idx_u(0,0), 4] = 1; U[idx_u(0,1), 4] = 1
    V[idx_v(1,1), 4] = 1
    W[idx_w(0,0), 4] = -1; W[idx_w(1,0), 4] = 1
    
    # r=5: m6 = (a10 - a00)(b00 + b01)
    U[idx_u(1,0), 5] = 1; U[idx_u(0,0), 5] = -1
    V[idx_v(0,0), 5] = 1; V[idx_v(0,1), 5] = 1
    W[idx_w(1,1), 5] = 1
    
    # r=6: m7 = (a01 - a11)(b10 + b11)
    U[idx_u(0,1), 6] = 1; U[idx_u(1,1), 6] = -1
    V[idx_v(1,0), 6] = 1; V[idx_v(1,1), 6] = 1
    W[idx_w(0,0), 6] = 1
    
    return U, V, W

def _get_border_tensor():
    n, m, p = 3, 3, 3
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if max(i, j, k) == 2:
                    T[i, j, k] = 1
    return T

def _search_border_decomposition(T_border, rank=16, time_limit=10):
    """
    Attempts to find a rank-16 decomposition of T_border using alternating minimization.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 3, 3, 3
    target = T_border.flatten()
    start_time = time.time()
    
    # We will try multiple random restarts
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize U, V, W with small random integers
        # To help convergence, we can initialize with sparse random vectors
        U = np.random.randint(-2, 3, (9, rank))
        V = np.random.randint(-2, 3, (9, rank))
        W = np.random.randint(-2, 3, (9, rank))
        
        # Ensure non-zero
        if np.all(U == 0): U[0,0] = 1
        if np.all(V == 0): V[0,0] = 1
        if np.all(W == 0): W[0,0] = 1
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(200):
            # Update U
            # T_border[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Fix V, W. For each (i,j), we have:
            # T_border[i,j,:] = U[i,j,:] * (V[j,:] * W[:,i]) ... no, it's a sum over r.
            # Let M_U[i,j,k] = sum_r V[j,k,r] W[k,i,r]
            # Then T_border[i,j,k] = sum_r U[i,j,r] M_U[i,j,k]_r? No.
            # T_border[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
            # This is not a standard linear least squares for U because U[i,j,r] is multiplied by a term that depends on r.
            # However, for fixed i,j, we have a system:
            # For each k, T_border[i,j,k] = sum_r U[i,j,r] * Q_r(i,j,k)
            # where Q_r(i,j,k) = V[j,k,r] * W[k,i,r]
            # This is a linear system in U[i,j,:].
            
            # We can solve for U by flattening.
            # Vectorize U to (9*9*rank)? No, U is 9xrank.
            # Let's construct the matrix A_U for the linear system.
            # Variables: U[0..8, 0..15] (144 variables)
            # Equations: T_border[0..2, 0..2, 0..2] (27 equations)
            # This is underdetermined. We use least squares.
            
            # Construct A_U
            # A_U is 27 x 144
            A_U = np.zeros((27, 9*rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Coefficient for U[i,j,r] is V[j,k,r] * W[k,i,r]
                        for r in range(rank):
                            A_U[i*m*p + j*p + k, i*rank + j*rank + r] = V[j*p+k, r] * W[k*n+i, r]
            
            # Solve A_U * U_flat = target
            U_flat, residuals, rank_u, s = np.linalg.lstsq(A_U, target, rcond=None)
            U = U_flat.reshape((9, rank))
            
            # Update V
            A_V = np.zeros((27, 9*rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            # Coefficient for V[j,k,r] is U[i,j,r] * W[k,i,r]
                            A_V[i*m*p + j*p + k, j*rank + k*rank + r] = U[i*m+j, r] * W[k*n+i, r]
            
            V_flat, residuals, rank_v, s = np.linalg.lstsq(A_V, target, rcond=None)
            V = V_flat.reshape((9, rank))
            
            # Update W
            A_W = np.zeros((27, 9*rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            # Coefficient for W[k,i,r] is U[i,j,r] * V[j,k,r]
                            A_W[i*m*p + j*p + k, k*rank + i*rank + r] = U[i*m+j, r] * V[j*p+k, r]
            
            W_flat, residuals, rank_w, s = np.linalg.lstsq(A_W, target, rcond=None)
            W = W_flat.reshape((9, rank))
            
            # Calculate error
            T_recon = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            error = np.linalg.norm(T_recon.flatten() - target)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if error < 1e-9:
                # Check if we can round to integers/half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_check = np.einsum('ijr,jkr,kir->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_check, T_border, atol=1e-9):
                    return U_rounded, V_rounded, W_rounded
                    
                # If rounding fails, continue searching
                # But if error is very small, maybe just return the float version?
                # The task requires integer/half-integer entries.
                # So we must find an exact rational solution.
                
        # End iterations for this restart
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Get Core Strassen Terms
    U_core, V_core, W_core = _get_strassen_core()
    
    # 2. Get Border Tensor
    T_border = _get_border_tensor()
    
    # 3. Try to find Rank-16 Border Decomposition
    border_result = _search_border_decomposition(T_border, rank=16, time_limit=100)
    
    if border_result:
        U_b, V_b, W_b = border_result
        # Combine
        U = np.hstack([U_core, U_b])
        V = np.hstack([V_core, V_b])
        W = np.hstack([W_core, W_b])
        
        # Verify full decomposition
        T_recon = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        T_target = np.ones((3,3,3))
        if np.allclose(T_recon, T_target, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # 4. Fallback to Rank-26
    # 7 Core + 19 Standard Border Terms
    U_std = np.zeros((9, 19))
    V_std = np.zeros((9, 19))
    W_std = np.zeros((9, 19))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue
                U_std[i*3+j, col] = 1
                V_std[j*3+k, col] = 1
                W_std[k*3+i, col] = 1
                col += 1
                
    U = np.hstack([U_core, U_std])
    V = np.hstack([V_core, V_std])
    W = np.hstack([W_core, W_std])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

