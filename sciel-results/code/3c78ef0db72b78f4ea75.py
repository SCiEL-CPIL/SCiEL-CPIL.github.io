import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 2, 12
    # Target rank 40 (4 blocks of rank 10)
    # First, find a rank 10 decomposition for p=3
    
    def find_rank_10_block():
        # Tensor for 2x2x3
        n_b, m_b, p_b = 2, 2, 3
        rank_b = 10
        
        # Initialize
        # Use random initialization
        np.random.seed(42)
        U = np.random.randn(n_b * m_b, rank_b)
        V = np.random.randn(m_b * p_b, rank_b)
        W = np.random.randn(p_b * n_b, rank_b)
        
        # Objective function
        def objective(params):
            U_flat = params[0:n_b*m_b*rank_b]
            V_flat = params[n_b*m_b*rank_b:n_b*m_b*rank_b + m_b*p_b*rank_b]
            W_flat = params[n_b*m_b*rank_b + m_b*p_b*rank_b:]
            
            U_mat = U_flat.reshape(n_b*m_b, rank_b)
            V_mat = V_flat.reshape(m_b*p_b, rank_b)
            W_mat = W_flat.reshape(p_b*n_b, rank_b)
            
            error = 0.0
            for i in range(n_b):
                for j in range(m_b):
                    for k in range(p_b):
                        # T[i,j,k] = 1
                        val = 0.0
                        for c in range(rank_b):
                            val += U_mat[i*m_b+j, c] * V_mat[j*p_b+k, c] * W_mat[k*n_b+i, c]
                        error += (val - 1.0)**2
            return error
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Optimization
        start_time = time.time()
        best_x = x0
        best_err = objective(x0)
        
        # Try a few restarts if time permits
        for restart in range(5):
            if time.time() - start_time > 20:
                break
            
            # Random restart
            if restart > 0:
                U = np.random.randn(n_b * m_b, rank_b) * 0.1
                V = np.random.randn(m_b * p_b, rank_b) * 0.1
                W = np.random.randn(p_b * n_b, rank_b) * 0.1
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            res = minimize(objective, x0, method='BFGS', options={'maxiter': 5000, 'disp': False})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
                
            if best_err < 1e-9:
                break
        
        if best_err < 1e-6:
            U_b = best_x[0:n_b*m_b*rank_b].reshape(n_b*m_b, rank_b)
            V_b = best_x[n_b*m_b*rank_b:n_b*m_b*rank_b + m_b*p_b*rank_b].reshape(m_b*p_b, rank_b)
            W_b = best_x[n_b*m_b*rank_b + m_b*p_b*rank_b:].reshape(p_b*n_b, rank_b)
            return U_b, V_b, W_b
        else:
            return None

    block = find_rank_10_block()
    
    if block is not None:
        U_b, V_b, W_b = block
        # Construct full rank 40 decomposition
        rank = 40
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        num_blocks = p // 3 # 4 blocks
        
        for b in range(num_blocks):
            u_offset = b * 10
            v_offset_row = b * 3 # B has 2 rows, 12 cols. Block b covers cols 3b, 3b+1, 3b+2
            w_offset_row = b * 3 # C has 2 rows, 12 cols. Block b covers cols 3b, 3b+1, 3b+2
            
            # U is shared structure? No, U is just the coefficients for A.
            # The block decomposition implies that the terms for block b only involve A, B_block, C_block.
            # So U for block b is just U_b.
            U[:, u_offset:u_offset+10] = U_b
            
            # V: B is 2x12. Block b is cols 3b..3b+2.
            # V_b is 6x10 (2 rows * 3 cols).
            # We need to place V_b into V[:, u_offset:u_offset+10]
            # V indices: row j (0,1), col k (0..11).
            # For block b, local col k' (0,1,2) maps to global k = 3b + k'.
            for j in range(2):
                for k_local in range(3):
                    k_global = 3*b + k_local
                    V[j*12 + k_global, u_offset:u_offset+10] = V_b[j*3 + k_local, :]
            
            # W: C is 2x12. Block b is cols 3b..3b+2.
            # W_b is 6x10 (2 cols * 2 rows? No, W is p*n = 12*2 = 24).
            # W indices: row k (0..11), col i (0,1).
            # For block b, local col k' (0,1,2) maps to global k = 3b + k'.
            for k_local in range(3):
                for i in range(2):
                    k_global = 3*b + k_local
                    W[k_global*2 + i, u_offset:u_offset+10] = W_b[k_local*2 + i, :]
        
        # Verify
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
        
        if max_err < 1e-9:
            return {"U": U, "V": V, "W": W}
    
    # Fallback to Rank 42
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    num_pairs = p // 2
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                U[0][c] = 1
                U[3][c] = 1
                V[0*p + 2*t][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 1:
                U[2][c] = 1
                U[3][c] = 1
                V[0*p + 2*t][c] = 1
                W[2*t*n + 1][c] = 1
                W[(2*t+1)*n + 1][c] = -1
            elif s == 2:
                U[0][c] = 1
                V[0*p + 2*t+1][c] = 1
                V[1*p + 2*t+1][c] = -1
                W[(2*t+1)*n + 0][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 3:
                U[3][c] = 1
                V[1*p + 2*t][c] = 1
                V[0*p + 2*t][c] = -1
                W[2*t*n + 0][c] = 1
                W[2*t*n + 1][c] = 1
            elif s == 4:
                U[0][c] = 1
                U[1][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = -1
                W[(2*t+1)*n + 0][c] = 1
            elif s == 5:
                U[2][c] = 1
                U[0][c] = -1
                V[0*p + 2*t][c] = 1
                V[0*p + 2*t+1][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 6:
                U[1][c] = 1
                U[3][c] = -1
                V[1*p + 2*t][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = 1

    return {"U": U, "V": V, "W": W}
