import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Build decomposition using block Strassen
    U = np.zeros((n*m, 0))
    V = np.zeros((m*p, 0))
    W = np.zeros((p*n, 0))
    
    def add_block(U_block, V_block, W_block, u_offset, v_offset, w_offset):
        """Add a block decomposition to the global U, V, W."""
        nonlocal U, V, W
        rank_b = U_block.shape[1]
        U_new = np.zeros((U.shape[0], rank_b))
        V_new = np.zeros((V.shape[0], rank_b))
        W_new = np.zeros((W.shape[0], rank_b))
        U_new[u_offset:u_offset+U_block.shape[0], :] = U_block
        V_new[v_offset:v_offset+V_block.shape[0], :] = V_block
        W_new[w_offset:w_offset+W_block.shape[0], :] = W_block
        U = np.hstack([U, U_new])
        V = np.hstack([V, V_new])
        W = np.hstack([W, W_new])
    
    # Strassen 2x2x2
    def strassen_2x2x2():
        U_b = np.zeros((4, 7))
        V_b = np.zeros((4, 7))
        W_b = np.zeros((4, 7))
        # Need correct formulas - let me verify numerically
        # For now, use schoolbook 2x2x2 (rank 4) as fallback
        for i in range(2):
            for j in range(2):
                col = i*2+j
                U_b[i*2+j, col] = 1
                V_b[j*2+i, col] = 1  # B[j,i]
                W_b[i*2+j, col] = 1  # C[i,j] -> W[j*2+i]
        return U_b, V_b, W_b
    
    # For now, just use schoolbook to get a working baseline
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
