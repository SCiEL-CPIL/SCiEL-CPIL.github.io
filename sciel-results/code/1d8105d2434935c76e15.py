import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 4 * 7 (Strassen) + 1 (Rank-1 for 2x4x1) + 1 (Rank-1 for 1x4x5) = 30
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i_off, j_off, k_off, c_off):
        nonlocal col
        # Local indices 0,1
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U coefficients for A
        # A11 at (0,0), A12 at (0,1), A21 at (1,0), A22 at (1,1)
        
        # M1
        U[(i_off+0)*m + (j_off+0)][c_off+0] += 1
        U[(i_off+1)*m + (j_off+1)][c_off+0] += 1
        V[(j_off+0)*p + (k_off+0)][c_off+0] += 1
        V[(j_off+0)*p + (k_off+1)][c_off+0] += 1
        V[(j_off+1)*p + (k_off+0)][c_off+0] += 1
        V[(j_off+1)*p + (k_off+1)][c_off+0] += 1
        W[(k_off+0)*n + (i_off+0)][c_off+0] += 1
        W[(k_off+0)*n + (i_off+1)][c_off+0] += 1
        W[(k_off+1)*n + (i_off+0)][c_off+0] += 1
        W[(k_off+1)*n + (i_off+1)][c_off+0] += 1
        
        # M2
        U[(i_off+1)*m + (j_off+0)][c_off+1] += 1
        U[(i_off+1)*m + (j_off+1)][c_off+1] += 1
        V[(j_off+0)*p + (k_off+0)][c_off+1] += 1
        W[(k_off+0)*n + (i_off+1)][c_off+1] += 1
        
        # M3
        U[(i_off+0)*m + (j_off+0)][c_off+2] += 1
        V[(j_off+1)*p + (k_off+0)][c_off+2] += 1
        V[(j_off+1)*p + (k_off+1)][c_off+2] += -1
        W[(k_off+0)*n + (i_off+0)][c_off+2] += 1
        W[(k_off+1)*n + (i_off+0)][c_off+2] += 1
        
        # M4
        U[(i_off+1)*m + (j_off+1)][c_off+3] += 1
        V[(j_off+1)*p + (k_off+1)][c_off+3] += 1
        V[(j_off+1)*p + (k_off+0)][c_off+3] += -1
        W[(k_off+0)*n + (i_off+1)][c_off+3] += 1
        W[(k_off+1)*n + (i_off+1)][c_off+3] += 1
        
        # M5
        U[(i_off+0)*m + (j_off+0)][c_off+4] += 1
        U[(i_off+0)*m + (j_off+1)][c_off+4] += 1
        V[(j_off+1)*p + (k_off+1)][c_off+4] += 1
        W[(k_off+1)*n + (i_off+0)][c_off+4] += 1
        
        # M6
        U[(i_off+1)*m + (j_off+0)][c_off+5] += 1
        U[(i_off+0)*m + (j_off+0)][c_off+5] += -1
        V[(j_off+0)*p + (k_off+0)][c_off+5] += 1
        V[(j_off+0)*p + (k_off+1)][c_off+5] += 1
        W[(k_off+0)*n + (i_off+1)][c_off+5] += 1
        W[(k_off+1)*n + (i_off+1)][c_off+5] += 1
        
        # M7
        U[(i_off+0)*m + (j_off+1)][c_off+6] += 1
        U[(i_off+1)*m + (j_off+1)][c_off+6] += -1
        V[(j_off+1)*p + (k_off+0)][c_off+6] += 1
        V[(j_off+1)*p + (k_off+1)][c_off+6] += 1
        W[(k_off+0)*n + (i_off+0)][c_off+6] += 1
        W[(k_off+1)*n + (i_off+0)][c_off+6] += 1
        
        col += 7

    # Helper to add Rank-1 block for 1xm x mxp or 2xm x mx1
    # This specific helper is for the remaining parts
    def add_rank1_block(i_indices, j_indices, k_indices, c_off):
        nonlocal col
        # U[i*m+j, c] = 1/2
        # V[j*p+k, c] = 1/2
        # W[k*n+i, c] = 1
        # Sum over j of 1/2 * 1/2 * 1 = len(j_indices)/4
        # We need this to be 1.
        # So len(j_indices) must be 4.
        # In our cases:
        # 1. i=[2], j=[0,1,2,3], k=[0,1,2,3,4]. len(j)=4. OK.
        # 2. i=[0,1], j=[0,1,2,3], k=[4]. len(j)=4. OK.
        
        for i in i_indices:
            for j in j_indices:
                U[i*m + j][c_off] += 0.5
        for j in j_indices:
            for k in k_indices:
                V[j*p + k][c_off] += 0.5
        for k in k_indices:
            for i in i_indices:
                W[k*n + i][c_off] += 1.0
        col += 1

    # 4 Strassen blocks for i in {0,1}
    # j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0, col)
    # j in {2,3}, k in {0,1}
    add_strassen(0, 2, 0, col)
    # j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2, col)
    # j in {2,3}, k in {2,3}
    add_strassen(0, 2, 2, col)
    
    # Rank-1 block for i in {0,1}, j in {0..3}, k in {4}
    add_rank1_block([0, 1], [0, 1, 2, 3], [4], col)
    
    # Rank-1 block for i in {2}, j in {0..3}, k in {0..4}
    add_rank1_block([2], [0, 1, 2, 3], [0, 1, 2, 3, 4], col)
    
    return {"U": U, "V": V, "W": W}
