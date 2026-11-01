"""Matrix multiplication 2,3,10 tensor decomposition."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check whether the decomposition exactly reproduces the matmul tensor."""
    import numpy as np
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < 1e-12


def _build_strassen_232():
    """
    Construct U, V, W for 2x3x2 matrix multiplication with rank 11.
    U: (2*3, 11)
    V: (3*2, 11)
    W: (2*2, 11)
    """
    n, m, p = 2, 3, 2
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part (7 mults) for A_left (2x2) * B_top (2x2)
    # c=0: m0 = (a00+a11)(b00+b11)
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    
    # c=1: m1 = (a10+a11)b00
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1; W[1*2+1][1] = -1
    
    # c=2: m2 = a00(b10-b11)
    U[0*3+0][2] = 1
    V[1*2+0][2] = 1; V[1*2+1][2] = -1
    W[1*2+0][2] = 1; W[1*2+1][2] = 1
    
    # c=3: m3 = a11(b01-b00)
    U[1*3+1][3] = 1
    V[0*2+1][3] = 1; V[0*2+0][3] = -1
    W[0*2+0][3] = -1; W[1*2+0][3] = 1
    
    # c=4: m4 = (a00+a01)b11
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+1][4] = 1; W[1*2+1][4] = 1
    
    # c=5: m5 = (a10-a00)(b00+b01)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[0*2+0][5] = 1; V[0*2+1][5] = 1
    W[0*2+0][5] = 1
    
    # c=6: m6 = (a01-a11)(b10+b11)
    U[0*3+1][6] = 1; U[1*3+1][6] = -1
    V[1*2+0][6] = 1; V[1*2+1][6] = 1
    W[0*2+0][6] = 1
    
    # Outer product part (4 mults) for a_right (2x1) * b_bottom (1x2)
    # c=7: a02 * b20 -> C00
    U[0*3+2][7] = 1
    V[2*2+0][7] = 1
    W[0*2+0][7] = 1
    
    # c=8: a02 * b21 -> C01
    U[0*3+2][8] = 1
    V[2*2+1][8] = 1
    W[1*2+0][8] = 1
    
    # c=9: a12 * b20 -> C10
    U[1*3+2][9] = 1
    V[2*2+0][9] = 1
    W[0*2+1][9] = 1
    
    # c=10: a12 * b21 -> C11
    U[1*3+2][10] = 1
    V[2*2+1][10] = 1
    W[1*2+1][10] = 1
    
    return U, V, W


def _construct_block_rank():
    """
    Construct the 55-rank solution by tiling 5 blocks of 2x3x2 (rank 11 each).
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    block_rank = 11
    total_rank = num_blocks * block_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U_block, V_block, W_block = _build_strassen_232()
    
    for t in range(num_blocks):
        offset = t * block_rank
        for i in range(n * m):
            for c in range(block_rank):
                U[i][offset + c] = U_block[i][c]
        
        for j in range(m):
            for k_local in range(2):
                k_global = t * 2 + k_local
                local_row = j * 2 + k_local
                global_row = j * p + k_global
                for c in range(block_rank):
                    V[global_row][offset + c] = V_block[local_row][c]
                    
        for k_local in range(2):
            k_global = t * 2 + k_local
            for i in range(n):
                local_row = k_local * n + i
                global_row = k_global * n + i
                for c in range(block_rank):
                    W[global_row][offset + c] = W_block[local_row][c]
                    
    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Fallback schoolbook decomposition."""
    n, m, p = DIMENSIONS
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


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Try the constructive 55-rank solution
    try:
        sol = _construct_block_rank()
        if _verify(sol["U"], sol["V"], sol["W"], n, m, p):
            return sol
    except Exception:
        pass
        
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
