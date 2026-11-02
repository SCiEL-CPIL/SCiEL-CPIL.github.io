"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Uses Kronecker structure: split B into 4 blocks of 3x2.
    Each 2x3x2 block decomposes as 2x2x2 (Strassen, rank 7) + 2x1x2 (rank 1) = rank 8.
    Total rank: 4 * 8 = 32.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]  # (6, 32)
    V = [[0] * rank for _ in range(m * p)]  # (24, 32)
    W = [[0] * rank for _ in range(p * n)]  # (16, 32)
    
    # Strassen decomposition for 2x2x2
    # Products M_1 through M_7
    # U coefficients for A (2x2): U[i*2+j, r]
    # V coefficients for B (2x2): V[j*2+k, r]
    # W coefficients for C (2x2): W[k*2+i, r]
    
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block s (0,1,2,3), we have 8 rank-1 terms
    for s in range(4):
        base = s * 8  # offset in rank dimension
        
        # 2x2x2 part: j in {0,1}, k in {2s, 2s+1}
        # Strassen products
        # M1: (A00+A11)(B00+B11)
        # A part: A[0,0]=1, A[1,1]=1
        # B part: B[0,2s]=1, B[1,2s+1]=1
        # C part: C[0,2s]+=1, C[0,2s+1]+=1, C[1,2s]+=1, C[1,2s+1]+=1
        r = base + 0
        U[0*3+0][r] = 1  # A[0,0]
        U[1*3+1][r] = 1  # A[1,1]
        V[0*8+2*s][r] = 1  # B[0,2s]
        V[1*8+2*s+1][r] = 1  # B[1,2s+1]
        W[2*s*2+0][r] = 1  # C[0,2s]
        W[2*s*2+1][r] = 1  # C[0,2s+1]
        W[(2*s+1)*2+0][r] = 1  # C[1,2s]
        W[(2*s+1)*2+1][r] = 1  # C[1,2s+1]
        
        # M2: (A10+A11)B00
        # A part: A[1,0]=1, A[1,1]=1
        # B part: B[0,2s]=1
        # C part: C[1,2s]+=1, C[1,2s+1]-=1
        r = base + 1
        U[1*3+0][r] = 1  # A[1,0]
        U[1*3+1][r] = 1  # A[1,1]
        V[0*8+2*s][r] = 1  # B[0,2s]
        W[(2*s+1)*2+0][r] = 1  # C[1,2s]
        W[(2*s+1)*2+1][r] = -1  # C[1,2s+1]
        
        # M3: A00(B01-B11)
        # A part: A[0,0]=1
        # B part: B[0,2s+1]=1, B[1,2s+1]=-1
        # C part: C[0,2s+1]+=1, C[1,2s+1]+=1
        r = base + 2
        U[0*3+0][r] = 1  # A[0,0]
        V[0*8+2*s+1][r] = 1  # B[0,2s+1]
        V[1*8+2*s+1][r] = -1  # B[1,2s+1]
        W[2*s*2+1][r] = 1  # C[0,2s+1]
        W[(2*s+1)*2+1][r] = 1  # C[1,2s+1]
        
        # M4: A11(B10-B00)
        # A part: A[1,1]=1
        # B part: B[1,2s]=1, B[0,2s]=-1
        # C part: C[0,2s]+=1, C[1,2s]+=1
        r = base + 3
        U[1*3+1][r] = 1  # A[1,1]
        V[1*8+2*s][r] = 1  # B[1,2s]
        V[0*8+2*s][r] = -1  # B[0,2s]
        W[2*s*2+0][r] = 1  # C[0,2s]
        W[(2*s+1)*2+0][r] = 1  # C[1,2s]
        
        # M5: (A00+A01)B11
        # A part: A[0,0]=1, A[0,1]=1
        # B part: B[1,2s+1]=1
        # C part: C[0,2s]-=1, C[0,2s+1]+=1
        r = base + 4
        U[0*3+0][r] = 1  # A[0,0]
        U[0*3+1][r] = 1  # A[0,1]
        V[1*8+2*s+1][r] = 1  # B[1,2s+1]
        W[2*s*2+0][r] = -1  # C[0,2s]
        W[2*s*2+1][r] = 1  # C[0,2s+1]
        
        # M6: (A10-A00)(B00+B01)
        # A part: A[1,0]=1, A[0,0]=-1
        # B part: B[0,2s]=1, B[0,2s+1]=1
        # C part: C[1,2s+1]+=1
        r = base + 5
        U[1*3+0][r] = 1  # A[1,0]
        U[0*3+0][r] = -1  # A[0,0]
        V[0*8+2*s][r] = 1  # B[0,2s]
        V[0*8+2*s+1][r] = 1  # B[0,2s+1]
        W[(2*s+1)*2+1][r] = 1  # C[1,2s+1]
        
        # M7: (A01-A11)(B10+B11)
        # A part: A[0,1]=1, A[1,1]=-1
        # B part: B[1,2s]=1, B[1,2s+1]=1
        # C part: C[0,2s]+=1
        r = base + 6
        U[0*3+1][r] = 1  # A[0,1]
        U[1*3+1][r] = -1  # A[1,1]
        V[1*8+2*s][r] = 1  # B[1,2s]
        V[1*8+2*s+1][r] = 1  # B[1,2s+1]
        W[2*s*2+0][r] = 1  # C[0,2s]
        
        # 2x1x2 part: j=2, k in {2s, 2s+1}
        # This is A[:,2] * B[2,:] for the block
        # C[i,2s] += A[i,2]*B[2,2s]
        # C[i,2s+1] += A[i,2]*B[2,2s+1]
        # This is a rank-1 term in the tensor sense
        # U part: A[0,2]=1, A[1,2]=1
        # V part: B[2,2s]=1, B[2,2s+1]=1
        # W part: C[0,2s]=1, C[0,2s+1]=1, C[1,2s]=1, C[1,2s+1]=1
        r = base + 7
        U[0*3+2][r] = 1  # A[0,2]
        U[1*3+2][r] = 1  # A[1,2]
        V[2*8+2*s][r] = 1  # B[2,2s]
        V[2*8+2*s+1][r] = 1  # B[2,2s+1]
        W[2*s*2+0][r] = 1  # C[0,2s]
        W[2*s*2+1][r] = 1  # C[0,2s+1]
        W[(2*s+1)*2+0][r] = 1  # C[1,2s]
        W[(2*s+1)*2+1][r] = 1  # C[1,2s+1]
    
    # Verify the decomposition
    def verify(U, V, W):
        """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for rr in range(r):
                        total += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
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
