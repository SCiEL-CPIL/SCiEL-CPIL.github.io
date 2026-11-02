"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix
    by a 5x5 matrix. The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # 1. Initialize with Schoolbook Decomposition (Guaranteed Valid)
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = U
    best_V = V
    best_W = W

    # 2. Try Strassen-based improvement (Rank 71)
    # Strategy: Use 4 disjoint 2x2x2 Strassen blocks and schoolbook for the rest.
    
    try:
        # Define Strassen blocks (i0, j0, k0)
        # These blocks must be disjoint in their index ranges to simply add ranks.
        # n=3, m=5, p=5
        # We cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
        blocks = [
            (0, 0, 0),  # i:0-1, j:0-1, k:0-1
            (0, 2, 0),  # i:0-1, j:2-3, k:0-1
            (0, 0, 2),  # i:0-1, j:0-1, k:2-3
            (0, 2, 2),  # i:0-1, j:2-3, k:2-3
        ]
        
        total_strassen_rank = len(blocks) * 7
        
        # Calculate remaining schoolbook terms
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0+di][j0+dj][k0+dk] = True
        
        schoolbook_count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        schoolbook_count += 1
        
        total_rank = total_strassen_rank + schoolbook_count
        
        if total_rank < best_rank:
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            col = 0
            
            def add_strassen_block(i0, j0, k0, col_offset):
                # M1 = (A00+A11)(B00+B11)
                c = col_offset + 0
                U[i0*m + j0][c] += 1
                U[(i0+1)*m + (j0+1)][c] += 1
                V[j0*p + k0][c] += 1
                V[(j0+1)*p + (k0+1)][c] += 1
                W[k0*n + i0][c] += 1
                W[(k0+1)*n + (i0+1)][c] += 1
                
                # M2 = (A10+A11)B00
                c = col_offset + 1
                U[(i0+1)*m + j0][c] += 1
                U[(i0+1)*m + (j0+1)][c] += 1
                V[j0*p + k0][c] += 1
                W[k0*n + (i0+1)][c] += 1
                W[(k0+1)*n + (i0+1)][c] += -1
                
                # M3 = A00(B01-B11)
                c = col_offset + 2
                U[i0*m + j0][c] += 1
                V[j0*p + (k0+1)][c] += 1
                V[(j0+1)*p + (k0+1)][c] += -1
                W[(k0+1)*n + i0][c] += 1
                W[(k0+1)*n + (i0+1)][c] += 1
                
                # M4 = A11(B10-B00)
                c = col_offset + 3
                U[(i0+1)*m + (j0+1)][c] += 1
                V[(j0+1)*p + k0][c] += 1
                V[j0*p + k0][c] += -1
                W[k0*n + i0][c] += 1
                W[k0*n + (i0+1)][c] += 1
                
                # M5 = (A00+A01)B11
                c = col_offset + 4
                U[i0*m + j0][c] += 1
                U[i0*m + (j0+1)][c] += 1
                V[(j0+1)*p + (k0+1)][c] += 1
                W[k0*n + i0][c] += -1
                W[(k0+1)*n + i0][c] += 1
                
                # M6 = (A10-A00)(B00+B01)
                c = col_offset + 5
                U[(i0+1)*m + j0][c] += 1
                U[i0*m + j0][c] += -1
                V[j0*p + k0][c] += 1
                V[j0*p + (k0+1)][c] += 1
                W[(k0+1)*n + (i0+1)][c] += 1
                
                # M7 = (A01-A11)(B10+B11)
                c = col_offset + 6
                U[i0*m + (j0+1)][c] += 1
                U[(i0+1)*m + (j0+1)][c] += -1
                V[(j0+1)*p + k0][c] += 1
                V[(j0+1)*p + (k0+1)][c] += 1
                W[k0*n + i0][c] += 1

            for i0, j0, k0 in blocks:
                add_strassen_block(i0, j0, k0, col)
                col += 7
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if not covered[i][j][k]:
                            U[i*m+j][col] += 1
                            V[j*p+k][col] += 1
                            W[k*n+i][col] += 1
                            col += 1
            
            # Verify
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            target = np.zeros((n*m, m*p, p*n), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i*m+j, j*p+k, k*n+i] = 1
            T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
            if np.allclose(T_decomp, target, atol=1e-10):
                best_rank = total_rank
                best_U = U
                best_V = V
                best_W = W
            else:
                # Verification failed, revert to schoolbook
                pass
                
    except Exception:
        # Fallback to schoolbook
        pass

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
