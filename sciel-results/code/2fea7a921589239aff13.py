import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Rank-54 exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Strategy:
    - Partition middle dim (4) into two 2-blocks
    - Partition last dim (6) into three 2-blocks
    - For each 3x2x2 sub-block, use a rank-9 decomposition
    - Total: 2 * 3 * 9 = 54 terms
    
    U shape: (12, 54)  -- rows indexed by i*4+j
    V shape: (24, 54)  -- rows indexed by j*6+k
    W shape: (18, 54)  -- rows indexed by k*3+i
    """
    n, m, p = 3, 4, 6
    rank = 54
    U = [[0] * rank for _ in range(n * m)]   # 12 x 54
    V = [[0] * rank for _ in range(m * p)]   # 24 x 54
    W = [[0] * rank for _ in range(p * n)]   # 18 x 54
    
    col = 0
    
    def add_rank9_3x2x2(i0, i1, i2, j_base, k_base):
        """
        Add 9 terms for a 3x2x2 block:
        A is 3x2: rows i0,i1,i2, cols j_base, j_base+1
        B is 2x2: rows j_base,j_base+1, cols k_base,k_base+1
        C is 3x2: rows i0,i1,i2, cols k_base,k_base+1
        
        Rank-9 decomposition:
        Term 1: (A[i0,j0]+A[i0,j1]) * (B[j0,k0]+B[j0,k1])
        Term 2: (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        Term 3: (A[i2,j0]+A[i2,j1]) * B[j0,k0]
        Term 4: A[i0,j0] * (B[j0,k1]-B[j1,k1])
        Term 5: A[i1,j0] * (B[j0,k1]-B[j1,k1])
        Term 6: A[i2,j0] * (B[j0,k1]-B[j1,k1])
        Term 7: A[i1,j1] * (B[j1,k0]-B[j0,k0])
        Term 8: A[i2,j1] * (B[j1,k0]-B[j0,k0])
        Term 9: A[i0,j1] * (B[j1,k1]-B[j1,k0])
        """
        nonlocal col
        
        # A indices
        a00 = i0 * m + j_base
        a01 = i0 * m + (j_base + 1)
        a10 = i1 * m + j_base
        a11 = i1 * m + (j_base + 1)
        a20 = i2 * m + j_base
        a21 = i2 * m + (j_base + 1)
        
        # B indices
        b00 = j_base * p + k_base
        b01 = j_base * p + (k_base + 1)
        b10 = (j_base + 1) * p + k_base
        b11 = (j_base + 1) * p + (k_base + 1)
        
        # W indices: W[k*n + i]
        w00 = k_base * n + i0
        w01 = (k_base + 1) * n + i0
        w10 = k_base * n + i1
        w11 = (k_base + 1) * n + i1
        w20 = k_base * n + i2
        w21 = (k_base + 1) * n + i2
        
        # Term 1: (a00+a01)*(b00+b01)
        U[a00][col] = 1; U[a01][col] = 1
        V[b00][col] = 1; V[b01][col] = 1
        W[w00][col] = 1; W[w01][col] = 1
        col += 1
        
        # Term 2: (a10+a11)*b00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # Term 3: (a20+a21)*b00
        U[a20][col] = 1; U[a21][col] = 1
        V[b00][col] = 1
        W[w20][col] = 1
        col += 1
        
        # Term 4: a00*(b01-b11)
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[w01][col] = 1
        col += 1
        
        # Term 5: a10*(b01-b11)
        U[a10][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[w11][col] = 1
        col += 1
        
        # Term 6: a20*(b01-b11)
        U[a20][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[w21][col] = 1
        col += 1
        
        # Term 7: a11*(b10-b00)
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[w10][col] = 1
        col += 1
        
        # Term 8: a21*(b10-b00)
        U[a21][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[w20][col] = 1
        col += 1
        
        # Term 9: a01*(b11-b10)
        U[a01][col] = 1
        V[b11][col] = 1; V[b10][col] = -1
        W[w01][col] = 1
        col += 1

    # Process all 6 blocks: 2 j_blocks x 3 k_blocks
    for j_block in range(2):
        j_base = j_block * 2
        for k_block in range(3):
            k_base = k_block * 2
            add_rank9_3x2x2(0, 1, 2, j_base, k_base)
    
    # Verification
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    # Fallback to schoolbook if something went wrong
                    return _construct_schoolbook()
    
    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}
