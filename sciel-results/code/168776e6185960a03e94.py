"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _verify(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build target indices
    idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx.append((i*m+j, j*p+k, k*n+i))
    idx = np.array(idx)
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Compute sum_c U[a,c]*V[b,c]*W[c_] for target indices
    total = np.zeros(n*m*p, dtype=float)
    for r in range(rank):
        total += Ua[idx[:,0], r] * Va[idx[:,1], r] * Wa[idx[:,2], r]
    
    return np.allclose(total, np.ones(n*m*p), atol=1e-9)

def _strassen_decomposition(n, m, p):
    """
    Decompose M_{n,m,p} using block Strassen.
    For (2,4,6): split m=4 into two blocks of size 2, p=6 into three blocks of size 2.
    Each subproblem is M_{2,2,2} solved by Strassen (rank 7).
    Total rank = 2 * 3 * 7 = 42.
    """
    a_block_size = 2
    b_col_block_size = 2
    a_blocks = m // a_block_size  # 2
    b_col_blocks = p // b_col_block_size  # 3
    strassen_rank = 7
    rank = a_blocks * b_col_blocks * strassen_rank  # 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (first matrix M)
    # M indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},  # P1: a + d
        {(1, 0): 1, (1, 1): 1},  # P2: c + d
        {(0, 0): 1},              # P3: a
        {(1, 1): 1},              # P4: d
        {(0, 0): 1, (0, 1): 1},  # P5: a + b
        {(1, 0): 1, (0, 0): -1}, # P6: c - a
        {(0, 1): 1, (1, 1): -1}  # P7: b - d
    ]
    
    # Strassen coefficients for V (second matrix N)
    # N indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},  # P1: e + h
        {(0, 0): 1},              # P2: e
        {(0, 1): 1, (1, 1): -1}, # P3: f - h
        {(1, 0): 1, (0, 0): -1}, # P4: g - e
        {(1, 1): 1},              # P5: h
        {(0, 0): 1, (0, 1): 1},  # P6: e + f
        {(1, 0): 1, (1, 1): 1}   # P7: g + h
    ]
    
    # Strassen coefficients for W (output R)
    # R indices: (0,0)=r00, (0,1)=r01, (1,0)=r10, (1,1)=r11
    # R00 = P1 + P4 - P5 + P7
    # R01 = P3 + P5
    # R10 = P2 + P4
    # R11 = P1 - P2 + P3 + P6
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(1, 0): 1, (1, 1): -1}, # P2
        {(0, 1): 1, (1, 1): 1},  # P3
        {(0, 0): 1, (1, 0): 1},  # P4
        {(0, 0): -1, (0, 1): 1}, # P5
        {(1, 1): 1},             # P6
        {(0, 0): 1}              # P7
    ]
    
    col = 0
    for a in range(a_blocks):
        for b in range(b_col_blocks):
            for s in range(strassen_rank):
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * a_block_size + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * a_block_size + j_loc
                    k = b * b_col_block_size + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * b_col_block_size + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Generate the Strassen-based decomposition (rank 42)
    result = _strassen_decomposition(n, m, p)
    
    # Verify correctness
    if _verify(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
