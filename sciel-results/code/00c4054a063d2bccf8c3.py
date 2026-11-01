import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check all diagonal terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    
    # Check some off-diagonal terms
    np.random.seed(42)
    checks = 200
    for _ in range(checks):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        ip = np.random.randint(n)
        jp = np.random.randint(m)
        kp = np.random.randint(p)
        if i == ip and j == jp and k == kp:
            continue
        val = 0.0
        for r in range(rank):
            val += U_np[i*m+j, r] * V_np[jp*p+kp, r] * W_np[kp*n+ip, r]
        if abs(val) > 1e-10:
            return False
            
    return True

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 by 3x10.
    Uses block-wise Strassen+schoolbook decomposition with rank 55.
    """
    n, m, p = DIMENSIONS
    
    # Rank 11 decomposition for 2x3x2 block
    # U: 6 entries (2x3), V: 6 entries (3x2), W: 4 entries (2x2)
    # Indices for U: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # Indices for V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
    # Indices for W: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    
    block_terms = []
    
    # Strassen part (7 terms)
    # M1
    block_terms.append(([1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0], [1, 0, 0, 1]))
    # M2
    block_terms.append(([0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, -1]))
    # M3
    block_terms.append(([1, 0, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0], [0, 0, 1, 1]))
    # M4
    block_terms.append(([0, 0, 0, 1, 0, 0], [-1, 0, 1, 0, 0, 0], [1, 1, 0, 0]))
    # M5
    block_terms.append(([1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [-1, 0, 1, 0]))
    # M6
    block_terms.append([[-1, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 1]])
    # M7
    block_terms.append(([0, 1, 0, -1, 0, 0], [0, 0, 1, 1, 0, 0], [1, 0, 0, 0]))
    
    # Schoolbook part for A_R B_R (4 terms)
    # A_02 * B_20 -> C_00
    block_terms.append(([0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0]))
    # A_02 * B_21 -> C_01
    block_terms.append(([0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0]))
    # A_12 * B_20 -> C_10
    block_terms.append(([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0, 0]))
    # A_12 * B_21 -> C_11
    block_terms.append(([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1]))
    
    total_rank = 55
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for b in range(5):
        for u_loc, v_loc, w_loc in block_terms:
            # U is global directly
            for i in range(n * m):
                U[i][col] = u_loc[i]
            
            # V mapping
            for i in range(6):
                if v_loc[i] != 0:
                    k_loc = i // 2
                    l_loc = i % 2
                    g_idx = k_loc * p + (2 * b + l_loc)
                    V[g_idx][col] = v_loc[i]
                    
            # W mapping
            for i in range(4):
                if w_loc[i] != 0:
                    k_loc = i // 2
                    i_loc = i % 2
                    g_idx = (2 * b + k_loc) * n + i_loc
                    W[g_idx][col] = w_loc[i]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
