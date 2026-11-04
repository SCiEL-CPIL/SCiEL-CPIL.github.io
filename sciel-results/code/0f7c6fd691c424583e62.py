import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Target: 3x4x5
    # Split:
    # A (3x4) = [A1 (3x3), A2 (3x1)]
    # B (4x5) = [B1 (3x5); B2 (1x5)]
    # C = A1*B1 + A2*B2
    # B1 (3x5) = [B11 (3x3), B12 (3x2)]
    # C = A1*B11 + A1*B12 + A2*B2
    
    # Ranks:
    # A1*B11: 3x3x3. Try Rank 23 (Laderman). Fallback 27.
    # A1*B12: 3x3x2. Try Rank 12. Fallback 18.
    # A2*B2: 3x1x5. Rank 15 (Schoolbook).
    
    # We will construct the U, V, W matrices for the whole problem.
    # U shape: (12, R)
    # V shape: (20, R)
    # W shape: (15, R)
    
    # Helper to solve for a decomposition of a sub-problem
    def solve_subproblem(n_sub, m_sub, p_sub, target_rank, seed=42):
        """
        Finds a decomposition of rank target_rank for n_sub x m_sub x p_sub.
        Returns (U_sub, V_sub, W_sub) if found, else None.
        """
        if time.time() > deadline:
            return None
            
        # Target tensor is all ones
        T = np.ones((n_sub, m_sub, p_sub))
        
        # Parameters: U (n_sub*m_sub, R), V (m_sub*p_sub, R), W (p_sub*n_sub, R)
        n_params = target_rank * (n_sub*m_sub + m_sub*p_sub + p_sub*n_sub)
        
        def residuals(params):
            U = params[:n_sub*m_sub*target_rank].reshape(n_sub*m_sub, target_rank)
            V = params[n_sub*m_sub*target_rank : n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape(m_sub*p_sub, target_rank)
            W = params[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank :].reshape(p_sub*n_sub, target_rank)
            
            res = []
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        val = np.sum(U[i*m_sub+j, :] * V[j*p_sub+k, :] * W[k*n_sub+i, :])
                        res.append(val - T[i,j,k])
            return np.array(res)
        
        # Initialize
        np.random.seed(seed)
        # Use small integers as initial guess?
        # Or random floats?
        # Random floats usually work better for least_squares
        x0 = np.random.randn(n_params)
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if result.success or result.cost < 1e-10:
                U_sol = result.x[:n_sub*m_sub*target_rank].reshape(n_sub*m_sub, target_rank)
                V_sol = result.x[n_sub*m_sub*target_rank : n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape(m_sub*p_sub, target_rank)
                W_sol = result.x[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank :].reshape(p_sub*n_sub, target_rank)
                
                # Check if close to integers
                U_int = np.round(U_sol)
                V_int = np.round(V_sol)
                W_int = np.round(W_sol)
                
                if np.allclose(U_sol, U_int, atol=1e-5) and np.allclose(V_sol, V_int, atol=1e-5) and np.allclose(W_sol, W_int, atol=1e-5):
                    return U_int, V_int, W_int
        except Exception:
            pass
        
        return None

    # 1. Solve A1*B11 (3x3x3) with Rank 23
    U_11, V_11, W_11 = solve_subproblem(3, 3, 3, 23, seed=1)
    if U_11 is None:
        # Fallback to schoolbook rank 27
        U_11 = np.zeros((9, 27))
        V_11 = np.zeros((9, 27))
        W_11 = np.zeros((9, 27))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = i*9 + j*3 + k
                    U_11[i*3+j, col] = 1
                    V_11[j*3+k, col] = 1
                    W_11[k*3+i, col] = 1
        rank_11 = 27
    else:
        rank_11 = 23

    # 2. Solve A1*B12 (3x3x2) with Rank 12
    U_12, V_12, W_12 = solve_subproblem(3, 3, 2, 12, seed=2)
    if U_12 is None:
        # Fallback to schoolbook rank 18
        U_12 = np.zeros((9, 18))
        V_12 = np.zeros((9, 18)) # Wait, B12 is 3x2, so V is 3x2 = 6? No.
        # Subproblem 3x3x2:
        # U: 3x3 (9)
        # V: 3x2 (6)
        # W: 2x3 (6)
        U_12 = np.zeros((9, 18))
        V_12 = np.zeros((6, 18))
        W_12 = np.zeros((6, 18))
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    col = i*6 + j*2 + k
                    U_12[i*3+j, col] = 1
                    V_12[j*2+k, col] = 1
                    W_12[k*3+i, col] = 1
        rank_12 = 18
    else:
        rank_12 = 12
        # Ensure shapes are correct
        # U_12 is 9x12, V_12 is 6x12, W_12 is 6x12

    # 3. A2*B2 (3x1x5) with Rank 15
    # U: 3x1 (3)
    # V: 1x5 (5)
    # W: 5x3 (15)
    U_22 = np.zeros((3, 15))
    V_22 = np.zeros((5, 15))
    W_22 = np.zeros((15, 15))
    for i in range(3):
        for k in range(5):
            col = i*5 + k
            U_22[i, col] = 1
            V_22[k, col] = 1
            W_22[k*3+i, col] = 1
    rank_22 = 15

    # Total Rank
    total_rank = rank_11 + rank_12 + rank_22
    
    # Construct Global U, V, W
    U_global = np.zeros((12, total_rank))
    V_global = np.zeros((20, total_rank))
    W_global = np.zeros((15, total_rank))
    
    col_offset = 0
    
    # Add Part 1: A1*B11
    # U_11 is 9xR1. Map to U_global rows 0..8 (j=0..2)
    U_global[0:9, col_offset:col_offset+rank_11] = U_11
    # V_11 is 9xR1. Map to V_global rows 0..8 (j=0..2, k=0..2)
    # V_global index: j*5 + k.
    # For j=0..2, k=0..2, indices are 0,1,2, 5,6,7, 10,11,12.
    v_indices_11 = []
    for j in range(3):
        for k in range(3):
            v_indices_11.append(j*5 + k)
    V_global[v_indices_11, col_offset:col_offset+rank_11] = V_11
    # W_11 is 9xR1. Map to W_global rows 0..8 (k=0..2, i=0..2)
    # W_global index: k*3 + i.
    # For k=0..2, i=0..2, indices are 0..8.
    W_global[0:9, col_offset:col_offset+rank_11] = W_11
    
    col_offset += rank_11
    
    # Add Part 2: A1*B12
    # U_12 is 9xR2. Map to U_global rows 0..8 (j=0..2)
    U_global[0:9, col_offset:col_offset+rank_12] = U_12
    # V_12 is 6xR2. Map to V_global rows for j=0..2, k=3..4.
    # V_global index: j*5 + k.
    # j=0, k=3,4 -> 3, 4
    # j=1, k=3,4 -> 8, 9
    # j=2, k=3,4 -> 13, 14
    v_indices_12 = []
    for j in range(3):
        for k in range(3, 5):
            v_indices_12.append(j*5 + k)
    V_global[v_indices_12, col_offset:col_offset+rank_12] = V_12
    # W_12 is 6xR2. Map to W_global rows for k=3..4, i=0..2.
    # W_global index: k*3 + i.
    # k=3, i=0..2 -> 9, 10, 11
    # k=4, i=0..2 -> 12, 13, 14
    w_indices_12 = []
    for k in range(3, 5):
        for i in range(3):
            w_indices_12.append(k*3 + i)
    W_global[w_indices_12, col_offset:col_offset+rank_12] = W_12
    
    col_offset += rank_12
    
    # Add Part 3: A2*B2
    # U_22 is 3xR3. Map to U_global rows for j=3, i=0..2.
    # U_global index: i*4 + 3.
    # i=0 -> 3, i=1 -> 7, i=2 -> 11
    u_indices_22 = [i*4 + 3 for i in range(3)]
    U_global[u_indices_22, col_offset:col_offset+rank_22] = U_22
    # V_22 is 5xR3. Map to V_global rows for j=3, k=0..4.
    # V_global index: 3*5 + k = 15 + k.
    v_indices_22 = [15 + k for k in range(5)]
    V_global[v_indices_22, col_offset:col_offset+rank_22] = V_22
    # W_22 is 15xR3. Map to W_global rows 0..14.
    W_global[0:15, col_offset:col_offset+rank_22] = W_22
    
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
# EVOLVE-BLOCK-END
