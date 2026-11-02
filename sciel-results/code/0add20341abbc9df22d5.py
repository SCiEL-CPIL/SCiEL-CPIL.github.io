import numpy as np
import time

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Construct the rank-minimal tensor decomposition for 2x2 * 2x16.
    Uses Strassen's algorithm for the 2x2 core, replicated for 8 column blocks.
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    start_time = time.time()
    n, m, p = 2, 2, 16
    num_blocks = 8
    core_rank = 7
    total_rank = num_blocks * core_rank

    # Strassen coefficients for 2x2 multiplication
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U coefficients for A (indices 0,1,2,3 -> a00, a01, a10, a11)
    # V coefficients for B (indices 0,1,2,3 -> b00, b01, b10, b11)
    # W coefficients for C (indices 0,1,2,3 -> c00, c01, c10, c11)
    
    # Define core vectors as lists of tuples (index, value) to build sparse matrices efficiently
    # Or just define the dense 4x7 matrices directly using list comprehensions
    
    def get_core_matrices():
        U_core = [[0]*7 for _ in range(4)]
        V_core = [[0]*7 for _ in range(4)]
        W_core = [[0]*7 for _ in range(4)]
        
        # M1
        U_core[0][0] = 1; U_core[3][0] = 1
        V_core[0][0] = 1; V_core[3][0] = 1
        W_core[0][0] = 1; W_core[3][0] = 1
        
        # M2
        U_core[2][1] = 1; U_core[3][1] = 1
        V_core[0][1] = 1
        W_core[2][1] = 1; W_core[3][1] = -1
        
        # M3
        U_core[0][2] = 1
        V_core[1][2] = 1; V_core[3][2] = -1
        W_core[1][2] = 1; W_core[3][2] = 1
        
        # M4
        U_core[3][3] = 1
        V_core[2][3] = 1; V_core[0][3] = -1
        W_core[0][3] = 1; W_core[2][3] = 1
        
        # M5
        U_core[0][4] = 1; U_core[1][4] = 1
        V_core[3][4] = 1
        W_core[0][4] = -1; W_core[1][4] = 1
        
        # M6
        U_core[2][5] = 1; U_core[0][5] = -1
        V_core[0][5] = 1; V_core[1][5] = 1
        W_core[3][5] = 1
        
        # M7
        U_core[1][6] = 1; U_core[3][6] = -1
        V_core[2][6] = 1; V_core[3][6] = 1
        W_core[0][6] = 1
        
        return U_core, V_core, W_core

    U_core, V_core, W_core = get_core_matrices()
    
    # Build full U, V, W matrices
    # U: (4, 56)
    # V: (32, 56)
    # W: (32, 56)
    
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    for t in range(num_blocks):
        base_col = t * core_rank
        
        # Update U
        for i in range(n*m):
            for r in range(core_rank):
                U_full[i][base_col + r] = U_core[i][r]
                
        # Update V
        # B indices: j*p + k. j in 0,1. k in 0..15.
        # Block t covers k in 2t, 2t+1.
        # B_block rows correspond to B_core rows 0,1,2,3
        # B_core row 0 (b00) -> B[0, 2t] -> index 0*p + 2t
        # B_core row 1 (b01) -> B[0, 2t+1] -> index 0*p + 2t+1
        # B_core row 2 (b10) -> B[1, 2t] -> index 1*p + 2t
        # B_core row 3 (b11) -> B[1, 2t+1] -> index 1*p + 2t+1
        
        v_indices = [
            0 * p + 2 * t,
            0 * p + 2 * t + 1,
            1 * p + 2 * t,
            1 * p + 2 * t + 1
        ]
        
        for r in range(core_rank):
            for i in range(4):
                V_full[v_indices[i]][base_col + r] = V_core[i][r]
                
        # Update W
        # C indices: k*n + i. k in 0..15. i in 0,1.
        # Block t covers k in 2t, 2t+1.
        # C_block cols correspond to C_core cols 0,1,2,3
        # C_core col 0 (c00) -> C[0, 2t] -> index 2t*n + 0
        # C_core col 1 (c01) -> C[0, 2t+1] -> index 2t*n + 1
        # C_core col 2 (c10) -> C[1, 2t] -> index (2t+1)*n + 0
        # C_core col 3 (c11) -> C[1, 2t+1] -> index (2t+1)*n + 1
        
        # W_core rows:
        # Row 0 (c00) -> index 2t*n + 0
        # Row 1 (c01) -> index 2t*n + 1
        # Row 2 (c10) -> index (2t+1)*n + 0
        # Row 3 (c11) -> index (2t+1)*n + 1
        
        w_indices = [
            2 * t * n + 0,
            2 * t * n + 1,
            (2 * t + 1) * n + 0,
            (2 * t + 1) * n + 1
        ]
        
        for r in range(core_rank):
            for i in range(4):
                W_full[w_indices[i]][base_col + r] = W_core[i][r]
                
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > 140:
        pass # Should not happen
        
    return {
        "U": U_full,
        "V": V_full,
        "W": W_full
    }

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    np.random.seed(0)
    A = np.random.randint(-10, 10, (2, 2))
    B = np.random.randint(-10, 10, (2, 16))
    C_exp = A @ B
    
    C_rec = np.zeros((2, 16))
    for i in range(2):
        for k in range(16):
            s = 0
            for j in range(2):
                for r in range(56):
                    s += U[i*2+j, r] * V[j*16+k, r] * W[k*2+i, r]
            C_rec[i, k] = s
            
    if np.allclose(C_exp, C_rec):
        print("OK")
    else:
        print("FAIL")
        print(C_exp)
        print(C_rec)
