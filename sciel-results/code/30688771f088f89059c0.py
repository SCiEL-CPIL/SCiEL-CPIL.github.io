import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen decomposition for 2x2x2 block
    # i_offset: offset for i index (0 or 1, but here n=2 so i is 0,1)
    # j_offset: offset for j index (0 or 1)
    # k_offset: offset for k index (0 or 2)
    # base_rank: starting rank index
    def add_strassen(i_off, j_off, k_off, base_rank):
        # M1
        # (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        U[(i_off)*m + (j_off)][base_rank] += 1
        U[(i_off+1)*m + (j_off+1)][base_rank] += 1
        V[(j_off)*p + (k_off)][base_rank] += 1
        V[(j_off+1)*p + (k_off+1)][base_rank] += 1
        W[(k_off)*n + (i_off)][base_rank] += 1
        W[(k_off)*n + (i_off+1)][base_rank] += 1
        W[(k_off+1)*n + (i_off)][base_rank] += 1
        W[(k_off+1)*n + (i_off+1)][base_rank] += 1
        
        # M2
        # (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        U[(i_off+1)*m + (j_off)][base_rank+1] += 1
        U[(i_off+1)*m + (j_off+1)][base_rank+1] += 1
        V[(j_off)*p + (k_off)][base_rank+1] += 1
        W[(k_off)*n + (i_off+1)][base_rank+1] += 1
        
        # M3
        # A[i0,j0] * (B[j0,k1]-B[j1,k1])
        U[(i_off)*m + (j_off)][base_rank+2] += 1
        V[(j_off)*p + (k_off+1)][base_rank+2] += 1
        V[(j_off+1)*p + (k_off+1)][base_rank+2] -= 1
        W[(k_off+1)*n + (i_off)][base_rank+2] += 1
        
        # M4
        # A[i1,j1] * (B[j1,k0]-B[j0,k0])
        U[(i_off+1)*m + (j_off+1)][base_rank+3] += 1
        V[(j_off+1)*p + (k_off)][base_rank+3] += 1
        V[(j_off)*p + (k_off)][base_rank+3] -= 1
        W[(k_off)*n + (i_off+1)][base_rank+3] += 1
        
        # M5
        # (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        U[(i_off)*m + (j_off)][base_rank+4] += 1
        U[(i_off)*m + (j_off+1)][base_rank+4] += 1
        V[(j_off+1)*p + (k_off+1)][base_rank+4] += 1
        W[(k_off+1)*n + (i_off)][base_rank+4] += 1
        
        # M6
        # (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        U[(i_off+1)*m + (j_off)][base_rank+5] += 1
        U[(i_off)*m + (j_off)][base_rank+5] -= 1
        V[(j_off)*p + (k_off)][base_rank+5] += 1
        V[(j_off)*p + (k_off+1)][base_rank+5] += 1
        W[(k_off)*n + (i_off+1)][base_rank+5] += 1
        W[(k_off+1)*n + (i_off+1)][base_rank+5] += 1
        
        # M7
        # (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U[(i_off)*m + (j_off+1)][base_rank+6] += 1
        U[(i_off+1)*m + (j_off+1)][base_rank+6] -= 1
        V[(j_off+1)*p + (k_off)][base_rank+6] += 1
        V[(j_off+1)*p + (k_off+1)][base_rank+6] += 1
        W[(k_off)*n + (i_off)][base_rank+6] += 1
        W[(k_off+1)*n + (i_off)][base_rank+6] += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0, 0)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2, 7)
    
    # Block 3: j=2, k in {0,1,2,3}
    # Rank 1 decomposition
    r = 14
    for i in range(n):
        U[i*m + 2][r] = 1
    for k in range(p):
        V[2*p + k][r] = 1
    for k in range(p):
        for i in range(n):
            W[k*n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
