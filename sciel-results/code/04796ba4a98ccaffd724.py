import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # 4 Strassen 2x2x2 blocks
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    uncovered = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    uncovered.add((i, j, k))
    
    # Look for 2x2x1 subproblems (2 i's, 2 j's, 1 k) - can be done in rank 3
    # Or 2x1x2 (2 i's, 1 j, 2 k's) - rank 3
    # Or 1x2x2 (1 i, 2 j's, 2 k's) - rank 3
    
    used = set()
    special_decomp = []  # (type, params, terms, U_coeffs, V_coeffs, W_coeffs)
    
    # 2x2x1: fixed k, i in {i1,i2}, j in {j1,j2}
    # M1 = (A[i1,j1]+A[i2,j2])*B[j1,k]  -- no, B is 2x1 so B[j1,k], B[j2,k]
    # Actually for 2x2 * 2x1 -> 2x1 result
    # C[i1,k] = A[i1,j1]*B[j1,k] + A[i1,j2]*B[j2,k]
    # C[i2,k] = A[i2,j1]*B[j1,k] + A[i2,j2]*B[j2,k]
    # This is 4 terms. Can we do it in 3?
    # M1 = (A[i1,j1]+A[i1,j2])*B[j1,k] = A[i1,j1]*B[j1,k] + A[i1,j2]*B[j1,k]  -- wrong B index
    # This doesn't work cleanly for 2x2x1.
    
    # Let me try 1x2x2: fixed i, j in {j1,j2}, k in {k1,k2}
    # C[i,k1] = A[i,j1]*B[j1,k1] + A[i,j2]*B[j2,k1]
    # C[i,k2] = A[i,j1]*B[j1,k2] + A[i,j2]*B[j2,k2]
    # 4 terms. Rank 3?
    # M1 = A[i,j1]*(B[j1,k1]+B[j1,k2])
    # M2 = A[i,j1]*B[j1,k2]
    # M3 = A[i,j2]*(B[j2,k1]+B[j2,k2])
    # M4 = A[i,j2]*B[j2,k2]  -- need 4
    # 
    # Alternative:
    # M1 = (A[i,j1]+A[i,j2])*(B[j1,k1]+B[j2,k2])  -- cross terms
    # This is getting complicated. Let me just stick with the current approach
    # and ensure it's correct and efficient.
    
    total_rank = len(blocks) * 7 + len(uncovered)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    for (i, j, k) in sorted(uncovered):
        if time.time() > deadline:
            break
        U[i*m+j][col] += 1
        V[j*p+k][col] += 1
        W[k*n+i][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
