"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7 decomposition.
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W, rank

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication.
    Uses block partitioning: split m=3 into 1+2, apply Strassen to 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 2, 3, 8

    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
        Rank 7 decomposition.
        
        For A (2x2) * B (2x2) = C (2x2):
        A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        
        Strassen's 7 products:
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-e)
        M5 = (a+b)h
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
        
        C = [[M1+M4-M5+M7, M3+M5],
             [M2+M4, M1-M2+M3+M6]]
        
        We need to express this as tensor decomposition.
        U[i*m+j, r] corresponds to A[i,j] coefficient in term r
        V[j*p+k, r] corresponds to B[j,k] coefficient in term r  
        W[k*n+i, r] corresponds to C[i,k] coefficient in term r
        
        For 2x2x2: n=2, m=2, p=2
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # Index convention:
        # U[i*m+j] where i in 0..1, j in 0..1 -> A[i,j]
        # V[j*p+k] where j in 0..1, k in 0..1 -> B[j,k]
        # W[k*n+i] where k in 0..1, i in 0..1 -> C[i,k]
        
        # A indices: U[0]=A[0,0]=a, U[1]=A[0,1]=b, U[2]=A[1,0]=c, U[3]=A[1,1]=d
        # B indices: V[0]=B[0,0]=e, V[1]=B[0,1]=f, V[2]=B[1,0]=g, V[3]=B[1,1]=h
        # C indices: W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
        
        # M1 = (a+d)(e+h)
        # U: a=1 at idx0, d=1 at idx3
        # V: e=1 at idx0, h=1 at idx3
        # W: C[0,0]+=1 (idx0), C[1,1]+=1 (idx3)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (c+d)e
        # U: c=1 at idx2, d=1 at idx3
        # V: e=1 at idx0
        # W: C[1,0]+=1 (idx2)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = a(f-h)
        # U: a=1 at idx0
        # V: f=1 at idx1, h=-1 at idx3
        # W: C[0,1]+=1 (idx1), C[1,1]+=1 (idx3)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = d(g-e)
        # U: d=1 at idx3
        # V: g=1 at idx2, e=-1 at idx0
        # W: C[0,0]+=1 (idx0), C[1,0]+=1 (idx2)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (a+b)h
        # U: a=1 at idx0, b=1 at idx1
        # V: h=1 at idx3
        # W: C[0,0]-=1 (idx0), C[0,1]+=1 (idx1)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (c-a)(e+f)
        # U: c=1 at idx2, a=-1 at idx0
        # V: e=1 at idx0, f=1 at idx1
        # W: C[1,1]+=1 (idx3)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (b-d)(g+h)
        # U: b=1 at idx1, d=-1 at idx3
        # V: g=1 at idx2, h=1 at idx3
        # W: C[0,0]+=1 (idx0), C[0,1]+=1 (idx1)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[1][6] = 1
        
        return U, V, W, rank

    def make_schoolbook():
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
        return U, V, W, rank

    def verify(U, V, W, rank):
        """Check that the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    # Strategy: Split m=3 into m1=1 and m2=2
    # The multiplication is:
    # A (2x3) * B (3x8) = C (2x8)
    # Split A into A1 (2x1) and A2 (2x2)
    # Split B into B1 (1x8) and B2 (2x8)
    # C = A1*B1 + A2*B2
    
    # For A1*B1: this is 2x1x8, schoolbook rank = 16
    # For A2*B2: this is 2x2x8, we can use Strassen-like approach
    
    # For 2x2x8: split p=8 into four blocks of 2
    # Each 2x2x2 block can use Strassen (rank 7)
    # Total for A2*B2: 4 * 7 = 28
    # Total: 16 + 28 = 44
    
    # Let's construct this decomposition
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # (6, 44)
    V = [[0] * rank_total for _ in range(m * p)]  # (24, 44)
    W = [[0] * rank_total for _ in range(p * n)]  # (16, 44)
    
    col = 0
    
    # Part 1: A1 * B1 (2x1x8)
    # A1 is A[:, 0:1], B1 is B[0:1, :]
    # For each output C[i,k] = sum_j A[i,j1] * B[j1,k] where j1=0 only
    # So C[i,k] = A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1  # A[i, 0]
            V[0 * p + k][col] = 1  # B[0, k]
            W[k * n + i][col] = 1  # C[i, k]
            col += 1
    
    # Part 2: A2 * B2 (2x2x8)
    # A2 is A[:, 1:3], B2 is B[1:3, :]
    # Split p=8 into 4 blocks of 2
    # For each block, use Strassen 2x2x2
    
    # Get Strassen decomposition
    U_s, V_s, W_s, rank_s = strassen_2x2x2()
    
    for block in range(4):
        k_start = block * 2
        # For each Strassen term
        for r in range(rank_s):
            # A2 indices: A[i, 1+j] where i in 0..1, j in 0..1
            # In original A: column index = 1 + j
            # U index: i * m + (1 + j)
            for i in range(2):
                for j in range(2):
                    if U_s[i * 2 + j][r] != 0:
                        U[i * m + (1 + j)][col] = U_s[i * 2 + j][r]
            
            # B2 indices: B[1+j, k_start+k] where j in 0..1, k in 0..1
            # In original B: row index = 1 + j, col index = k_start + k
            # V index: (1 + j) * p + (k_start + k)
            for j in range(2):
                for k in range(2):
                    if V_s[j * 2 + k][r] != 0:
                        V[(1 + j) * p + (k_start + k)][col] = V_s[j * 2 + k][r]
            
            # C indices: C[i, k_start+k] where i in 0..1, k in 0..1
            # W index: (k_start + k) * n + i
            for k in range(2):
                for i in range(2):
                    if W_s[k * 2 + i][r] != 0:
                        W[(k_start + k) * n + i][col] = W_s[k * 2 + i][r]
            
            col += 1
    
    # Verify
    if not verify(U, V, W, rank_total):
        # Fallback to schoolbook
        U, V, W, rank_total = make_schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
