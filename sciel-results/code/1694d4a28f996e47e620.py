import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank 23 decomposition for 3x3x3 is complex to hardcode reliably in short time.
    # Fallback to the 4-Strassen-block strategy (Rank 71) but with a local polish.
    # Or try the 68-rank structure with a generic 3x3x3 solver if time permits?
    # Given constraints, I will stick to the robust 71-rank solution but attempt
    # a specific optimization: checking if any schoolbook term can be merged.
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1; W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1; V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1; V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1; W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1; U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1; W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1; U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1; V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1; U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
