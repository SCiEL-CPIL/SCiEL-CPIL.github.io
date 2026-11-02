import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _snap_to_half(x):
    """Snap values to nearest 0.5"""
    return np.round(x * 2) / 2

def _construct_rank_29():
    """
    Construct a rank 29 decomposition using Strassen for the 2x2x2 block
    and schoolbook for the remaining parts.
    """
    n, m, p = 2, 3, 5
    rank = 29
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] using Strassen (Rank 7)
    # A indices: 0,1,3,4 (i*m+j: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4)
    # B indices: 0,1,5,6 (j*p+k: 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6)
    # C indices: 0,1,2,3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3-M6
    
    # U for Strassen (A part)
    # A00=0, A01=1, A10=3, A11=4
    U_strassen = np.array([
        [1, 0, 0, 0, 1, 1, -1], # A00 (idx 0)
        [0, 0, 1, 0, 0, 0, 1],  # A01 (idx 1)
        [0, 0, 0, 0, 0, 0, 0],  # A10 (idx 2 - wait, A10 is idx 3 in global)
        [0, 0, 0, 0, 0, 0, 0],  # A11 (idx 3 - wait, A11 is idx 4 in global)
    ])
    # Correction: Global U indices for A[0:2, 0:2] are 0, 1, 3, 4.
    # Let's map local (i,j) to global idx.
    # Local 0,0 -> Global 0
    # Local 0,1 -> Global 1
    # Local 1,0 -> Global 3
    # Local 1,1 -> Global 4
    
    # Strassen U rows (local 0,0,1,1)
    # M1: A00+A11 -> [1, 0, 0, 1]
    # M2: A10+A11 -> [0, 0, 1, 1]
    # M3: A00      -> [1, 0, 0, 0]
    # M4: A11      -> [0, 0, 0, 1]
    # M5: A00+A10  -> [1, 0, 1, 0]
    # M6: A10-A00  -> [-1, 0, 1, 0]
    # M7: A01-A11  -> [0, 1, 0, -1]
    
    U_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 0], # Placeholder, we will fill specific rows
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    # Actually, easier to just define the 7 columns for the 4 rows.
    # Rows: 0 (A00), 1 (A01), 3 (A10), 4 (A11)
    U_block1 = np.zeros((6, 7))
    U_block1[0, :] = [1, 0, 1, 0, 1, -1, 0] # A00: M1, M3, M5, M6
    U_block1[1, :] = [0, 0, 0, 0, 0, 0, 1]   # A01: M7
    U_block1[3, :] = [0, 1, 0, 0, 1, 1, 0]   # A10: M2, M5, M6
    U_block1[4, :] = [1, 1, 0, 1, 0, 0, -1]  # A11: M1, M2, M4, M7
    
    # V for Strassen (B part)
    # Local 0,0 -> Global 0
    # Local 0,1 -> Global 1
    # Local 1,0 -> Global 5
    # Local 1,1 -> Global 6
    V_block1 = np.zeros((15, 7))
    V_block1[0, :] = [1, 1, 0, 0, 0, 1, 1]   # B00: M1, M2, M6, M7
    V_block1[1, :] = [1, 0, 1, 0, 0, 0, 1]   # B01: M1, M3, M7
    V_block1[5, :] = [0, 1, 0, 1, 1, 1, 0]   # B10: M2, M4, M5, M6
    V_block1[6, :] = [1, 0, -1, -1, 1, 0, -1] # B11: M1, M3, M4, M5, M7
    
    # W for Strassen (C part)
    # Local 0,0 -> Global 0
    # Local 0,1 -> Global 1
    # Local 1,0 -> Global 2
    # Local 1,1 -> Global 3
    W_block1 = np.zeros((10, 7))
    W_block1[0, :] = [1, 0, 0, 1, -1, 0, -1] # C00: M1, M4, M5, M7
    W_block1[1, :] = [0, 0, 1, 0, 1, 0, 0]   # C01: M3, M5
    W_block1[2, :] = [0, 1, 0, 1, 0, 0, 0]   # C10: M2, M4
    W_block1[3, :] = [1, -1, 1, 0, 0, -1, 0] # C11: M1, M2, M3, M6
    
    U[:, 0:7] = U_block1
    V[:, 0:7] = V_block1
    W[:, 0:7] = W_block1
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:5] (Schoolbook 2x2x3, Rank 12)
    # A indices: 0, 1, 3, 4
    # B indices: 2, 3, 4, 7, 8, 9
    # C indices: 4, 5, 6, 7, 8, 9
    
    U_block2 = np.zeros((6, 12))
    V_block2 = np.zeros((15, 12))
    W_block2 = np.zeros((10, 12))
    
    col_offset = 7
    for i in range(2):
        for j in range(2):
            for k in range(3):
                idx = col_offset + (i * 2 + j) * 3 + k
                u_idx = i * 3 + j
                v_idx = j * 5 + (2 + k)
                w_idx = (2 + k) * 2 + i
                U_block2[u_idx, idx - col_offset] = 1
                V_block2[v_idx, idx - col_offset] = 1
                W_block2[w_idx, idx - col_offset] = 1
                
    U[:, 7:19] = U_block2
    V[:, 7:19] = V_block2
    W[:, 7:19] = W_block2

    # Block 3: A[0:2, 2:3] * B[2:3, 0:5] (Schoolbook 2x1x5, Rank 10)
    # A indices: 2, 5
    # B indices: 10, 11, 12, 13, 14
    # C indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    
    U_block3 = np.zeros((6, 10))
    V_block3 = np.zeros((15, 10))
    W_block3 = np.zeros((10, 10))
    
    col_offset = 19
    for i in range(2):
        for k in range(5):
            idx = col_offset + i * 5 + k
            u_idx = i * 3 + 2
            v_idx = 2 * 5 + k
            w_idx = k * 2 + i
            U_block3[u_idx, idx - col_offset] = 1
            V_block3[v_idx, idx - col_offset] = 1
            W_block3[w_idx, idx - col_offset] = 1
            
    U[:, 19:29] = U_block3
    V[:, 19:29] = V_block3
    W[:, 19:29] = W_block3

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return U_list, V_list, W_list

def _optimize_rank(n, m, p, rank, deadline, max_restarts=10, init=None):
    """Try to find a decomposition of given rank using optimization."""
    n_vars = n * m * rank + m * p * rank + p * n * rank
    
    def objective(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        error = np.sum((T - 1.0)**2)
        return error

    for restart in range(max_restarts):
        if time.time() > deadline:
            return None, None, None
        
        # Initialization
        if init is not None and restart == 0:
            # Use provided init
            x0 = init
        else:
            # Random initialization
            init_U = np.random.randn(n, m, rank) * 0.5
            init_V = np.random.randn(m, p, rank) * 0.5
            init_W = np.random.randn(p, n, rank) * 0.5
            x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if result.fun < 1e-8:
                # Extract solution
                U_opt = result.x[:n*m*rank].reshape(n, m, rank)
                V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
                W_opt = result.x[n*m*rank + m*p*rank:].reshape(p, n, rank)
                
                # Snap to half-integers
                U_snap = _snap_to_half(U_opt)
                V_snap = _snap_to_half(V_opt)
                W_snap = _snap_to_half(W_opt)
                
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U_snap.reshape(n*m, rank)]
                V_list = [[float(x) for x in row] for row in V_snap.reshape(m*p, rank)]
                W_list = [[float(x) for x in row] for row in W_snap.reshape(p*n, rank)]
                
                # Verify exactly
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
                    return U_list, V_list, W_list
                    
        except Exception as e:
            pass
            
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # 1. Try to find Rank 29 (Explicit Construction)
    U_29, V_29, W_29 = _construct_rank_29()
    if _verify_decomposition(U_29, V_29, W_29, n, m, p, tol=1e-9):
        best_result = {"U": U_29, "V": V_29, "W": W_29}
        best_rank = 29
    else:
        # Fallback to Schoolbook if construction fails
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank

    # 2. Try to find Rank < best_rank using Optimization
    # Prepare init for Rank 28 based on Rank 29 (pad with zeros)
    # This is a heuristic: maybe the extra dimension in 29 is redundant?
    # But 29 is exact. We need to find a different 28.
    # We will just try random restarts for 28, 27, etc.
    
    for rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline:
            break
        U, V, W = _optimize_rank(n, m, p, rank, deadline, max_restarts=10)
        if U is not None:
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found a valid decomposition, stop
            
    return best_result
