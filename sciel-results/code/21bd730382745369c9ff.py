import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 3, 6
    
    # Build target tensor T for 2x3x6
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Strategy: Use two independent 2x3x3 blocks
    # Block 1: k in {0,1,2}, Block 2: k in {3,4,5}
    # Each 2x3x3 has rank 15, so total rank = 30
    
    # For 2x3x3, construct explicit rank-15 decomposition
    n2, m2, p2 = 2, 3, 3
    
    # The 2x3x3 multiplication tensor T233[i*3+j, j*3+k, k*2+i] = 1
    # We need U233 (6x15), V233 (9x15), W233 (6x15)
    
    def solve_233_rank15():
        """Find rank-15 decomposition of 2x3x3 multiplication tensor"""
        R = 15
        rows_u = 6
        rows_v = 9
        rows_w = 6
        
        T233 = np.zeros((rows_u, rows_v, rows_w))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
        # Pre-compute target matrices for least squares
        T_flat_u = T233.reshape(rows_u, -1).T # (36, 6)
        T_flat_v = T233.transpose(1, 0, 2).reshape(rows_v, -1).T # (36, 9)
        T_flat_w = T233.transpose(2, 0, 1).reshape(rows_w, -1).T # (54, 6)
        
        best = None
        
        # Use more restarts with fewer iterations
        for seed in range(1000):
            if time.time() > deadline - 20:
                break
                
            rng = np.random.RandomState(seed)
            U = rng.randn(rows_u, R) * 0.5
            V = rng.randn(rows_v, R) * 0.5
            W = rng.randn(rows_w, R) * 0.5
            
            for it in range(50):
                # Solve for U given V, W
                # Z_vw[r, :] = outer(V[:, r], W[:, r])
                Z_vw = np.einsum('ir,jr->rij', V, W).reshape(R, rows_v * rows_w)
                U_sol, _, _, _ = np.linalg.lstsq(Z_vw.T, T_flat_u, rcond=None)
                U = U_sol.T
                
                # Solve for V given U, W
                Z_uw = np.einsum('ir,jr->rij', U, W).reshape(R, rows_u * rows_w)
                V_sol, _, _, _ = np.linalg.lstsq(Z_uw.T, T_flat_v, rcond=None)
                V = V_sol.T
                
                # Solve for W given U, V
                Z_uv = np.einsum('ir,jr->rij', U, V).reshape(R, rows_u * rows_v)
                W_sol, _, _, _ = np.linalg.lstsq(Z_uv.T, T_flat_w, rcond=None)
                W = W_sol.T
            
            # Check if solution is close to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_check = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.allclose(T_check, T233, atol=1e-9):
                return (U_r, V_r, W_r)
            
            # Try integer rounding
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            T_check_i = np.einsum('ir,jr,kr->ijk', U_i, V_i, W_i)
            if np.allclose(T_check_i, T233, atol=1e-9):
                return (U_i, V_i, W_i)
        
        return None
    
    result_233 = solve_233_rank15()
    
    if result_233 is not None:
        U233, V233, W233 = result_233
        R233 = 15
        R236 = 2 * R233
        
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        # Embed block 1 (k=0,1,2) and block 2 (k=3,4,5)
        # The indices for U are the same for both blocks.
        # V indices: j*p + k. Block 1: k in 0..2. Block 2: k in 3..5.
        # W indices: k*n + i. Block 1: k in 0..2. Block 2: k in 3..5.
        
        # Copy U for both blocks
        U236[:, :R233] = U233
        U236[:, R233:] = U233
        
        # Copy V and W for block 1
        for j in range(m2):
            for k in range(p2):
                v_idx_233 = j * p2 + k
                w_idx_233 = k * n2 + i # Wait, i is not in scope here.
                # We need to map the 2x3x3 indices to 2x3x6 indices.
                # In 2x3x3:
                # U index: i * m2 + j
                # V index: j * p2 + k
                # W index: k * n2 + i
                # In 2x3x6:
                # U index: i * m + j (same as i * m2 + j since m=m2=3)
                # V index: j * p + k
                # W index: k * n + i (same as k * n2 + i since n=n2=2)
                
                # So for each (i,j,k) in 2x3x3:
                # U[i*m2+j] is used.
                # V[j*p2+k] maps to V[j*p+k] for block 1, V[j*p+k+p2] for block 2.
                # W[k*n2+i] maps to W[k*n+i] for block 1, W[k*n+i+n] for block 2?
                # Wait, W index is k_C * n + i_C.
                # In 2x3x3, k is 0..2. In 2x3x6, k is 0..5.
                # Block 1: k in 0..2. W index = k * n + i.
                # Block 2: k in 3..5. Let k' = k + 3. W index = k' * n + i.
                
                # So we can just copy the rows.
        
        # Better way to embed:
        # U236[:, 0:15] = U233
        # U236[:, 15:30] = U233
        
        # V236 has 18 rows.
        # Rows 0-8 correspond to j=0,1,2 and k=0,1,2 (Block 1)
        # Rows 9-17 correspond to j=0,1,2 and k=3,4,5 (Block 2)
        # V233 has 9 rows.
        # V233[j*3+k] corresponds to V236[j*6+k] for block 1?
        # No, V index in 2x3x6 is j*p + k = j*6 + k.
        # For block 1 (k=0,1,2): indices are j*6 + k.
        # For block 2 (k=3,4,5): indices are j*6 + k.
        # So V233 row j*3+k maps to V236 row j*6+k for block 1?
        # Wait, in 2x3x3, V index is j*3+k.
        # In 2x3x6, V index is j*6+k.
        # So for a fixed j, the rows in V233 are 3j, 3j+1, 3j+2.
        # In V236, for block 1, the rows are 6j, 6j+1, 6j+2.
        # For block 2, the rows are 6j+3, 6j+4, 6j+5.
        
        for j in range(m2):
            for k in range(p2):
                v_idx_233 = j * p2 + k
                v_idx_b1 = j * p + k
                v_idx_b2 = j * p + k + p2
                
                V236[v_idx_b1, :R233] = V233[v_idx_233, :]
                V236[v_idx_b2, R233:] = V233[v_idx_233, :]
                
                w_idx_233 = k * n2 + i # i is not fixed!
                # W index in 2x3x3 is k*n2 + i.
                # In 2x3x6, W index is k*n + i.
                # For block 1, k is 0..2. W index is k*n + i.
                # For block 2, k is 3..5. Let k' = k+3. W index is k'*n + i.
                # So W233 row k*n2+i maps to W236 row k*n+i for block 1?
                # And W233 row k*n2+i maps to W236 row (k+3)*n + i for block 2?
                
                # We need to iterate over i as well?
                # No, W233 is a matrix. Each row corresponds to a specific (k,i) pair.
                # We can just copy the rows directly.
                
        # Let's do it more cleanly.
        # W233 has 6 rows. Indices 0..5.
        # W236 has 12 rows. Indices 0..11.
        # Block 1 uses W236 rows 0..5.
        # Block 2 uses W236 rows 6..11.
        # The mapping is direct: W233[r] -> W236[r, 0:15] and W233[r] -> W236[r+6, 15:30].
        
        W236[:6, :R233] = W233
        W236[6:, R233:] = W233
        
        # Verify
        T_final = np.einsum('ir,jr,kr->ijk', U236, V236, W236)
        if np.allclose(T_final, T, atol=1e-9):
            return {
                "U": U236.tolist(),
                "V": V236.tolist(),
                "W": W236.tolist()
            }
    
    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
