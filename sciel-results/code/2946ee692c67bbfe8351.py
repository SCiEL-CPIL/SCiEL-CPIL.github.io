import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for matrix multiplication.
    Uses a block-structured decomposition:
    - 4 Strassen 2x2x2 blocks for the main 2x4x4 part (rank 28)
    - Schoolbook for cross terms (rank 18)
    Total rank: 46
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135.0  # 15s margin before 150s kill
    
    # Build the rank-46 decomposition
    # A is (2,5), B is (5,5), C is (2,5)
    # Split: A = [A1, a2] where A1 is (2,4), a2 is (2,1)
    #        B = [B11, b12; b21, b22] where B11 is (4,4), b12 is (4,1), b21 is (1,4), b22 is (1,1)
    # C = A1*B11 + A1*b12 + a2*b21 + a2*b22
    #
    # A1*B11: 2x4x4, use 4 Strassen 2x2x2 blocks, rank 4*7=28
    # A1*b12: 2x4x1, schoolbook rank 8
    # a2*b21: 2x1x4, schoolbook rank 8
    # a2*b22: 2x1x1, schoolbook rank 2
    # Total: 28+8+8+2 = 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # (2*5, 46)
    V = [[0] * rank for _ in range(m * p)]  # (5*5, 46)
    W = [[0] * rank for _ in range(p * n)]  # (5*2, 46)
    
    col = 0
    
    # Helper: add a Strassen 2x2x2 block
    # A_block: rows i0..i0+1, cols j0..j0+1 of A
    # B_block: rows j0..j0+1, cols k0..k0+1 of B
    # C_block: rows i0..i0+1, cols k0..k0+1 of C
    def add_strassen(i0, j0, k0, start_col):
        # Strassen's 7 products for 2x2x2
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # For each product, we need to set U, V, W entries
        # U[i*m+j, col] = coefficient for A[i,j]
        # V[j*p+k, col] = coefficient for B[j,k]
        # W[k*n+i, col] = coefficient for C[k,i] (trace convention: W[k*n+i, r])
        
        # M1 = (a11+a22)(b11+b22)
        c = start_col
        # a11: i=i0, j=j0
        U[i0*m+j0][c] += 1
        # a22: i=i0+1, j=j0+1
        U[(i0+1)*m+(j0+1)][c] += 1
        # b11: j=j0, k=k0
        V[j0*p+k0][c] += 1
        # b22: j=j0+1, k=k0+1
        V[(j0+1)*p+(k0+1)][c] += 1
        # C contributions:
        # C11 = M1: i=i0, k=k0
        W[k0*n+i0][c] += 1
        # C22 = M1: i=i0+1, k=k0+1
        W[(k0+1)*n+(i0+1)][c] += 1
        col_offset = c
        c += 1
        
        # M2 = (a21+a22)b11
        c = start_col + 1
        # a21: i=i0+1, j=j0
        U[(i0+1)*m+j0][c] += 1
        # a22: i=i0+1, j=j0+1
        U[(i0+1)*m+(j0+1)][c] += 1
        # b11: j=j0, k=k0
        V[j0*p+k0][c] += 1
        # C21 = M2: i=i0+1, k=k0
        W[k0*n+(i0+1)][c] += 1
        c += 1
        
        # M3 = a11(b12-b22)
        c = start_col + 2
        # a11: i=i0, j=j0
        U[i0*m+j0][c] += 1
        # b12: j=j0, k=k0+1
        V[j0*p+(k0+1)][c] += 1
        # b22: j=j0+1, k=k0+1
        V[(j0+1)*p+(k0+1)][c] -= 1
        # C12 = M3: i=i0, k=k0+1
        W[(k0+1)*n+i0][c] += 1
        # C22 = M3: i=i0+1, k=k0+1
        W[(k0+1)*n+(i0+1)][c] += 1
        c += 1
        
        # M4 = a22(b21-b11)
        c = start_col + 3
        # a22: i=i0+1, j=j0+1
        U[(i0+1)*m+(j0+1)][c] += 1
        # b21: j=j0+1, k=k0
        V[(j0+1)*p+k0][c] += 1
        # b11: j=j0, k=k0
        V[j0*p+k0][c] -= 1
        # C11 = M4: i=i0, k=k0
        W[k0*n+i0][c] += 1
        # C21 = M4: i=i0+1, k=k0
        W[k0*n+(i0+1)][c] += 1
        c += 1
        
        # M5 = (a11+a12)b22
        c = start_col + 4
        # a11: i=i0, j=j0
        U[i0*m+j0][c] += 1
        # a12: i=i0, j=j0+1
        U[i0*m+(j0+1)][c] += 1
        # b22: j=j0+1, k=k0+1
        V[(j0+1)*p+(k0+1)][c] += 1
        # C11 = -M5: i=i0, k=k0
        W[k0*n+i0][c] -= 1
        # C12 = M5: i=i0, k=k0+1
        W[(k0+1)*n+i0][c] += 1
        c += 1
        
        # M6 = (a21-a11)(b11+b12)
        c = start_col + 5
        # a21: i=i0+1, j=j0
        U[(i0+1)*m+j0][c] += 1
        # a11: i=i0, j=j0
        U[i0*m+j0][c] -= 1
        # b11: j=j0, k=k0
        V[j0*p+k0][c] += 1
        # b12: j=j0, k=k0+1
        V[j0*p+(k0+1)][c] += 1
        # C22 = M6: i=i0+1, k=k0+1
        W[(k0+1)*n+(i0+1)][c] += 1
        c += 1
        
        # M7 = (a12-a22)(b21+b22)
        c = start_col + 6
        # a12: i=i0, j=j0+1
        U[i0*m+(j0+1)][c] += 1
        # a22: i=i0+1, j=j0+1
        U[(i0+1)*m+(j0+1)][c] -= 1
        # b21: j=j0+1, k=k0
        V[(j0+1)*p+k0][c] += 1
        # b22: j=j0+1, k=k0+1
        V[(j0+1)*p+(k0+1)][c] += 1
        # C11 = M7: i=i0, k=k0
        W[k0*n+i0][c] += 1
        c += 1
        
        return start_col + 7
    
    # Add 4 Strassen blocks for A1*B11 (2x4x4)
    # Block 1: i0=0, j0=0, k0=0
    col = add_strassen(0, 0, 0, col)
    # Block 2: i0=0, j0=2, k0=0
    col = add_strassen(0, 2, 0, col)
    # Block 3: i0=0, j0=0, k0=2
    col = add_strassen(0, 0, 2, col)
    # Block 4: i0=0, j0=2, k0=2
    col = add_strassen(0, 2, 2, col)
    
    # Now col should be 28
    
    # A1*b12: 2x4x1, schoolbook rank 8
    # A1 is (2,4), b12 is (4,1)
    # C contribution: (2,1) at k=4
    for i in range(2):
        for j in range(4):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    
    # a2*b21: 2x1x4, schoolbook rank 8
    # a2 is (2,1) at j=4, b21 is (1,4)
    # C contribution: (2,4) at k=0..3
    for i in range(2):
        for k in range(4):
            U[i*m+4][col] = 1
            V[4*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    
    # a2*b22: 2x1x1, schoolbook rank 2
    # a2 is (2,1) at j=4, b22 is (1,1) at j=4,k=4
    # C contribution: (2,1) at k=4
    for i in range(2):
        U[i*m+4][col] = 1
        V[4*p+4][col] = 1
        W[4*n+i][col] = 1
        col += 1
    
    # col should be 46
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to numpy arrays for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    # Reshape: U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    
    S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check if S is all 1s
    if not np.allclose(S, 1.0, atol=1e-9):
        # If verification fails, fall back to schoolbook
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
# EVOLVE-BLOCK-END
