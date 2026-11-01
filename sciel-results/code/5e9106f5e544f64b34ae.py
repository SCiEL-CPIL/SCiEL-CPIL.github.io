"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 by 2x12 matrix multiplication.
    
    Strategy: Decompose the 2x12 multiplication into 6 independent 2x2x2 blocks.
    Each 2x2x2 block uses Strassen's algorithm (rank 7 instead of schoolbook rank 4).
    Total rank: 6 * 7 = 42 (vs schoolbook 2*2*12 = 48).
    """
    n, m, p = DIMENSIONS
    
    assert n == 2 and m == 2 and p == 12, "This implementation is for 2x2 * 2x12"
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]    # (4, 42)
    V = [[0] * rank for _ in range(m * p)]    # (24, 42)
    W = [[0] * rank for _ in range(p * n)]    # (24, 42)
    
    # Strassen's 7 products for 2x2 * 2x2:
    # Each entry: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: list of (i, j, coeff) for A[i][j]
    # B_coeffs: list of (j, k, coeff) for B[j][k]
    # C_coeffs: list of (i, k, coeff) for C[i][k]
    
    strassen = [
        # M1 = (A11+A22)(B11+B22): C11+=1, C22+=1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A21+A22)B11: C21+=1, C22+=-1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3 = A11(B12-B22): C12+=1, C22+=1
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4 = A22(B21-B11): C11+=1, C21+=1
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5 = (A11+A12)B22: C12+=1, C11+=-1
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)]),
        # M6 = (A21-A11)(B11+B12): C22+=1
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7 = (A12-A22)(B21+B22): C11+=1
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    # Build decomposition for 2x2 * 2x12
    # B is split into 6 blocks of 2 columns each
    col = 0
    for block in range(6):
        for prod_idx in range(7):
            A_coeffs, B_coeffs, C_coeffs = strassen[prod_idx]
            
            # U: index = i*m + j
            for (i, j, coeff) in A_coeffs:
                u_idx = i * m + j
                U[u_idx][col] += coeff
            
            # V: index = j*p + k, where k = 2*block + k_local
            for (j, k_local, coeff) in B_coeffs:
                k_actual = 2 * block + k_local
                v_idx = j * p + k_actual
                V[v_idx][col] += coeff
            
            # W: index = k*n + i, where k = 2*block + k_local
            for (i, k_local, coeff) in C_coeffs:
                k_actual = 2 * block + k_local
                w_idx = k_actual * n + i
                W[w_idx][col] += coeff
            
            col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
