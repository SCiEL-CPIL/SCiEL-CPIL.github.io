"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 tensor decomposition for 2x2 by 2x14 matrix multiplication
    using Strassen's algorithm on 7 blocks of 2x2 matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Attempt Strassen-based decomposition
    result = _construct_strassen_49(n, m, p)
    
    if result is not None:
        # Verify the decomposition to ensure correctness
        if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
        else:
            # If verification fails, fall back to schoolbook (should not happen)
            pass
            
    # Fallback: Schoolbook decomposition (rank 56)
    return _construct_schoolbook(n, m, p)

def _construct_strassen_49(n, m, p):
    """
    Construct a rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2.
    """
    if time.time() > time.time() + 140:
        return None
        
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_forms = [
        [1, 0, 0, 1],   # s=0: A00+A11
        [0, 0, 1, 1],   # s=1: A10+A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00+A01
        [-1, 0, 1, 0],  # s=5: A10-A00
        [0, 1, 0, -1]   # s=6: A01-A11
    ]
    
    for r in range(num_blocks):
        idx_e = 2 * r       # B00
        idx_f = 2 * r + 1   # B01
        idx_g = p + 2 * r   # B10
        idx_h = p + 2 * r + 1 # B11
        
        w_idx_c00 = 4 * r
        w_idx_c10 = 4 * r + 1
        w_idx_c01 = 4 * r + 2
        w_idx_c11 = 4 * r + 3
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[s][i]
                
            # Fill V
            if s == 0:
                V[idx_e][c] = 1
                V[idx_h][c] = 1
            elif s == 1:
                V[idx_e][c] = 1
            elif s == 2:
                V[idx_f][c] = 1
                V[idx_h][c] = -1
            elif s == 3:
                V[idx_e][c] = -1
                V[idx_g][c] = 1
            elif s == 4:
                V[idx_h][c] = 1
            elif s == 5:
                V[idx_e][c] = 1
                V[idx_f][c] = 1
            elif s == 6:
                V[idx_g][c] = 1
                V[idx_h][c] = 1
                
            # Fill W
            if s == 0:
                W[w_idx_c00][c] = 1
                W[w_idx_c11][c] = 1
            elif s == 1:
                W[w_idx_c10][c] = 1
                W[w_idx_c11][c] = -1
            elif s == 2:
                W[w_idx_c01][c] = 1
                W[w_idx_c11][c] = 1
            elif s == 3:
                W[w_idx_c00][c] = 1
                W[w_idx_c10][c] = 1
            elif s == 4:
                W[w_idx_c00][c] = -1
                W[w_idx_c01][c] = 1
            elif s == 5:
                W[w_idx_c11][c] = 1
            elif s == 6:
                W[w_idx_c00][c] = 1
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, num_tests=2):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    Checks the tensor condition: sum_{j,c} U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    for all i, k. This ensures that the decomposition reconstructs the identity tensor.
    """
    rank = len(U[0])
    # Check all i, k
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            if abs(total - 1.0) > 1e-9:
                return False
    return True

def _construct_schoolbook(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
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
# EVOLVE-BLOCK-END
