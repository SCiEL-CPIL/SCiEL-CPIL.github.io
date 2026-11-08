"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p, offset=0):
    """Return (U, V, W) lists for schoolbook decomposition of n×m times m×p.
    U has shape (n*m, n*m*p), V has shape (m*p, n*m*p), W has shape (p*n, n*m*p).
    """
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, r

def _strassen_4x4_4x4():
    """Build recursive Strassen decomposition for 4x4 * 4x4.
    
    Level 1: 4x4 * 4x4 -> 7 multiplications of 2x2 * 2x2
    Level 2: Each 2x2 * 2x2 -> 7 multiplications of 1x1 * 1x1 (rank 1)
    Total rank: 7 * 7 = 49
    
    Returns U (16x49), V (16x49), W (16x49)
    """
    total_rank = 49
    n = m = p = 4
    U = [[0] * total_rank for _ in range(n * m)]   # (16, 49)
    V = [[0] * total_rank for _ in range(m * p)]   # (16, 49)
    W = [[0] * total_rank for _ in range(p * n)]   # (16, 49)
    
    # Level 1 Strassen coefficients (4x4 split into 2x2 blocks of 2x2)
    # alpha[t]: {(block_row, block_col): coeff} for A
    # beta[t]: {(block_row, block_col): coeff} for B  
    # gamma[t]: {(block_row, block_col): coeff} for C
    alpha = [
        {(0,0):1, (1,1):1},      # M1: A11+A22
        {(1,0):1, (1,1):1},      # M2: A21+A22
        {(0,0):1},               # M3: A11
        {(1,1):1},               # M4: A22
        {(0,0):1, (0,1):1},      # M5: A11+A12
        {(1,0):1, (0,0):-1},     # M6: A21-A11
        {(0,1):1, (1,1):-1},     # M7: A12-A22
    ]
    beta = [
        {(0,0):1, (1,1):1},      # M1: B11+B22
        {(0,0):1},               # M2: B11
        {(0,1):1, (1,1):-1},     # M3: B12-B22
        {(1,0):1, (0,0):-1},     # M4: B21-B11
        {(1,1):1},               # M5: B22
        {(0,0):1, (0,1):1},      # M6: B11+B12
        {(1,0):1, (1,1):1},      # M7: B21+B22
    ]
    gamma = [
        {(0,0):1, (1,1):1},      # M1: +C11, +C22
        {(1,0):1, (1,1):-1},     # M2: +C21, -C22
        {(0,1):1, (1,1):-1},     # M3: +C12, -C22
        {(0,0):1, (1,0):1},      # M4: +C11, +C21
        {(0,0):-1, (0,1):1},     # M5: -C11, +C12
        {(1,1):-1},              # M6: -C22
        {(0,0):1},               # M7: +C11
    ]
    
    # Level 2 Strassen coefficients (2x2 split into 2x2 blocks of 1x1)
    alpha2 = [
        {(0,0):1, (1,1):1},      # m1: A11+A22
        {(1,0):1, (1,1):1},      # m2: A21+A22
        {(0,0):1},               # m3: A11
        {(1,1):1},               # m4: A22
        {(0,0):1, (0,1):1},      # m5: A11+A12
        {(1,0):1, (0,0):-1},     # m6: A21-A11
        {(0,1):1, (1,1):-1},     # m7: A12-A22
    ]
    beta2 = [
        {(0,0):1, (1,1):1},      # m1: B11+B22
        {(0,0):1},               # m2: B11
        {(0,1):1, (1,1):-1},     # m3: B12-B22
        {(1,0):1, (0,0):-1},     # m4: B21-B11
        {(1,1):1},               # m5: B22
        {(0,0):1, (0,1):1},      # m6: B11+B12
        {(1,0):1, (1,1):1},      # m7: B21+B22
    ]
    gamma2 = [
        {(0,0):1, (1,1):1},      # m1: +C11, +C22
        {(1,0):1, (1,1):-1},     # m2: +C21, -C22
        {(0,1):1, (1,1):-1},     # m3: +C12, -C22
        {(0,0):1, (1,0):1},      # m4: +C11, +C21
        {(0,0):-1, (0,1):1},     # m5: -C11, +C12
        {(1,1):-1},              # m6: -C22
        {(0,0):1},               # m7: +C11
    ]
    
    # Build the full decomposition
    for t in range(7):  # Level 1 Strassen term
        for t2 in range(7):  # Level 2 Strassen term
            col = t * 7 + t2
            
            # For U: accumulate contributions from alpha[t] and alpha2[t2]
            # Level 1 block (br1, bc1) covers rows br1*2..br1*2+1, cols bc1*2..bc1*2+1 in A
            # Level 2 block (br2, bc2) within that covers row br1*2+br2, col bc1*2+bc2
            for (br1, bc1), a1 in alpha[t].items():
                for (br2, bc2), a2 in alpha2[t2].items():
                    full_i = br1 * 2 + br2
                    full_j = bc1 * 2 + bc2
                    U[full_i * 4 + full_j][col] += a1 * a2
            
            # For V: accumulate contributions from beta[t] and beta2[t2]
            for (br1, bc1), b1 in beta[t].items():
                for (br2, bc2), b2 in beta2[t2].items():
                    full_j = br1 * 2 + br2
                    full_k = bc1 * 2 + bc2
                    V[full_j * 4 + full_k][col] += b1 * b2
            
            # For W: accumulate contributions from gamma[t] and gamma2[t2]
            for (cr1, cc1), g1 in gamma[t].items():
                for (cr2, cc2), g2 in gamma2[t2].items():
                    full_k = cc1 * 2 + cc2
                    full_i = cr1 * 2 + cr2
                    W[full_k * 4 + full_i][col] += g1 * g2
    
    return U, V, W, total_rank

def _strassen_4x4_4x8():
    """Build decomposition for 4x4 * 4x8 by splitting into two 4x4 * 4x4 multiplications.
    
    Left half: A * B[:, 0:4] using recursive Strassen (rank 49)
    Right half: A * B[:, 4:8] using recursive Strassen (rank 49)
    Total rank: 98
    """
    n, m, p = 4, 4, 8
    total_rank = 98
    
    U = [[0] * total_rank for _ in range(n * m)]   # (16, 98)
    V = [[0] * total_rank for _ in range(m * p)]   # (32, 98)
    W = [[0] * total_rank for _ in range(p * n)]   # (32, 98)
    
    # Left half: columns 0-3 of B
    U_left, V_left, W_left, rank_left = _strassen_4x4_4x4()
    
    # Right half: columns 4-7 of B
    U_right, V_right, W_right, rank_right = _strassen_4x4_4x4()
    
    # Place left half in columns 0-48
    for i in range(n * m):
        for c in range(rank_left):
            U[i][c] = U_left[i][c]
    
    # Place right half in columns 49-97
    for i in range(n * m):
        for c in range(rank_right):
            U[i][rank_left + c] = U_right[i][c]
    
    # V: left half uses rows 0-15 (j=0..3, k=0..3), right half uses rows 0-15 (j=0..3, k=4..7)
    for j in range(m):
        for k in range(4):
            # Left half: V[j*4 + k] corresponds to V_left[j*4 + k]
            for c in range(rank_left):
                V[j * 8 + k][c] = V_left[j * 4 + k][c]
            # Right half: V[j*4 + (k+4)] corresponds to V_right[j*4 + k]
            for c in range(rank_right):
                V[j * 8 + (k + 4)][rank_left + c] = V_right[j * 4 + k][c]
    
    # W: left half uses rows 0-15 (k=0..3, i=0..3), right half uses rows 16-31 (k=4..7, i=0..3)
    for k in range(4):
        for i in range(n):
            # Left half: W[k*4 + i] corresponds to W_left[k*4 + i]
            for c in range(rank_left):
                W[k * 4 + i][c] = W_left[k * 4 + i][c]
            # Right half: W[(k+4)*4 + i] corresponds to W_right[k*4 + i]
            for c in range(rank_right):
                W[(k + 4) * 4 + i][rank_left + c] = W_right[k * 4 + i][c]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    try:
        result = _strassen_4x4_4x8()
        
        # Verify the decomposition numerically
        U, V, W = result["U"], result["V"], result["W"]
        n, m, p = 4, 4, 8
        rank = len(U[0])
        
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # For matrix multiplication tensor: T[i,j,k] = 1 for all i,j,k
        # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return _schoolbook_fallback()
        
        return result
    except Exception:
        return _schoolbook_fallback()

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U, V, W, _ = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
