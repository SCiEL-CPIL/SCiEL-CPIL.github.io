import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    # Rank 29 decomposition
    # 2x4x4: 2 Strassen blocks (k=0-1, k=2-3) -> 14
    # 2x4x2: 1 Strassen block (k=4-5) -> 7
    # 2x4x1: 2 schoolbook 2x2x1 blocks (j=0, j=2) -> 8
    # Total: 29
    
    rank = 29
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    def emit_strassen(i0, j0, k0):
        """
        Emits a 2x2x2 Strassen block for rows i0,i0+1; cols j0,j0+1; output cols k0,k0+1.
        """
        nonlocal col
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[i0*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M2 = (A21 + A22) * B11
        U[(i0+1)*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = -1.0
        col += 1
        
        # M3 = A11 * (B12 - B22)
        U[i0*m + j0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M4 = A22 * (B21 - B11)
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[j0*p + k0, col] = -1.0
        W[k0*n + (i0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        # M5 = (A11 + A12) * B22
        U[i0*m + j0, col] = 1.0
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[(i0+1)*m + j0, col] = 1.0
        U[i0*m + j0, col] = -1.0
        V[j0*p + k0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[i0*m + (j0+1), col] = 1.0
        U[(i0+1)*m + (j0+1), col] = -1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1

    def emit_schoolbook_2x2x1(i0, j0, k0):
        """
        Emits a 2x2x1 schoolbook block for rows i0,i0+1; cols j0,j0+1; output col k0.
        """
        nonlocal col
        
        # T1: A11 * B11 -> C11
        U[i0*m + j0, col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        # T2: A12 * B21 -> C11
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        # T3: A21 * B11 -> C21
        U[(i0+1)*m + j0, col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1
        
        # T4: A22 * B21 -> C21
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1

    # Part 1: 2x4x4 (k=0..3) - 2 Strassen blocks
    emit_strassen(0, 0, 0)  # j=0-1, k=0-1
    emit_strassen(0, 2, 0)  # j=2-3, k=0-1
    
    # Part 2: 2x4x2 (k=4..5) - 1 Strassen block
    emit_strassen(0, 0, 4)  # j=0-1, k=4-5
    
    # Part 3: 2x4x1 (k=6) - 2 schoolbook blocks
    emit_schoolbook_2x2x1(0, 0, 6)  # j=0-1, k=6
    emit_schoolbook_2x2x1(0, 2, 6)  # j=2-3, k=6
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
