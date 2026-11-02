"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a low-rank matrix multiplication decomposition using block partitioning.
    Partitions m=3 into m1=1 and m2=2, uses Strassen for 2x2x4 block and direct for 2x1x4 block.
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Verify a decomposition is valid
    def verify(U, V, W):
        """Check that for all i,j,k: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Schoolbook as fallback
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
        return U, V, W
    
    # Build Strassen decomposition for 2x2x4 block
    # Strassen 2x2x2 has rank 7
    # For 2x2x4, we can use 2 independent 2x2x2 Strassen decompositions
    # Total rank for 2x2x4: 2 * 7 = 14
    def strassen_2x2x2():
        """Return U,V,W for 2x2x2 Strassen decomposition (rank 7)"""
        # Strassen's 7 multiplications for 2x2x2
        # U[i,j,r] for i,j in {0,1}, r in 0..6
        # V[j,k,r] for j,k in {0,1}, r in 0..6  
        # W[k,i,r] for k,i in {0,1}, r in 0..6
        U = [[0]*7 for _ in range(4)]  # indexed by i*2+j
        V = [[0]*7 for _ in range(4)]  # indexed by j*2+k
        W = [[0]*7 for _ in range(4)]  # indexed by k*2+i
        
        # Standard Strassen 2x2x2 decomposition
        # r=0: (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # r=2: a00(b10-b11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # r=3: a11(b00-b01)
        U[3][3] = 1
        V[0][3] = 1; V[1][3] = -1
        W[0][3] = -1; W[3][3] = 1
        
        # r=4: (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[2][4] = 1
        
        # r=5: (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[0][5] = 0; W[1][5] = 0; W[2][5] = 0; W[3][5] = 0
        
        # r=6: (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1; W[1][6] = 1
        
        return U, V, W
    
    # Build direct multiplication for 2x1x4 block (rank 8)
    def direct_2x1x4():
        """Return U,V,W for 2x1x4 direct multiplication (rank 8)"""
        # U[i,j,r] for i in {0,1}, j in {0}, r in 0..7
        # V[j,k,r] for j in {0}, k in {0,1,2,3}, r in 0..7
        # W[k,i,r] for k in {0,1,2,3}, i in {0,1}, r in 0..7
        U = [[0]*8 for _ in range(2)]  # indexed by i (since m1=1, only j=0)
        V = [[0]*8 for _ in range(4)]  # indexed by k (since m1=1, only j=0)
        W = [[0]*8 for _ in range(8)]  # indexed by k*2+i
        
        for i in range(2):
            for k in range(4):
                r = i * 4 + k
                U[i][r] = 1
                V[k][r] = 1
                W[k * 2 + i][r] = 1
        
        return U, V, W
    
    # Now combine: partition m=3 into m1=1 (j=0) and m2=2 (j=1,2)
    # Block 1: 2x1x4 (j=0), rank 8
    # Block 2: 2x2x4 (j=1,2), rank 14 (two Strassen 2x2x2)
    # Total rank: 8 + 14 = 22
    
    # But wait - we need to be careful about how the blocks interact
    # The full tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,7}
    
    # Let's build the full decomposition:
    # U has shape (6, R) where R is total rank
    # V has shape (24, R)
    # W has shape (16, R)
    
    # Partition k into two blocks of 4: k in {0,1,2,3} and k in {4,5,6,7}
    # For each k-block, we have a 2x3x4 problem
    
    # For k-block 0 (k=0..3):
    #   j=0: 2x1x4 direct, rank 8
    #   j=1,2: 2x2x4 Strassen, rank 14
    #   Subtotal: 22
    
    # For k-block 1 (k=4..7):
    #   j=0: 2x1x4 direct, rank 8
    #   j=1,2: 2x2x4 Strassen, rank 14
    #   Subtotal: 22
    
    # Total rank: 44
    
    R = 44
    U = [[0] * R for _ in range(6)]   # 2*3 = 6 rows
    V = [[0] * R for _ in range(24)]  # 3*8 = 24 rows
    W = [[0] * R for _ in range(16)]  # 8*2 = 16 rows
    
    # Process k-block 0 (k=0..3)
    # j=0: direct 2x1x4
    U_0, V_0, W_0 = direct_2x1x4()
    for i in range(2):
        for k in range(4):
            r = i * 4 + k
            U[i * 3 + 0][r] = U_0[i][r]
            V[0 * 8 + k][r] = V_0[k][r]
            W[k * 2 + i][r] = W_0[k * 2 + i][r]
    
    # j=1,2: Strassen 2x2x4 (two 2x2x2 blocks)
    # For k=0,1: one Strassen 2x2x2
    # For k=2,3: another Strassen 2x2x2
    U_s, V_s, W_s = strassen_2x2x2()
    # First 2x2x2 for k=0,1
    for i in range(2):
        for j in range(2):  # j=0,1 correspond to original j=1,2
            for k in range(2):  # k=0,1
                for r in range(7):
                    U[i * 3 + (1 + j)][r + 8] = U_s[i * 2 + j][r]
                    V[(1 + j) * 8 + k][r + 8] = V_s[j * 2 + k][r]
                    W[k * 2 + i][r + 8] = W_s[k * 2 + i][r]
    
    # Second 2x2x2 for k=2,3
    for i in range(2):
        for j in range(2):  # j=0,1 correspond to original j=1,2
            for k in range(2):  # k=0,1 correspond to original k=2,3
                for r in range(7):
                    U[i * 3 + (1 + j)][r + 15] = U_s[i * 2 + j][r]
                    V[(1 + j) * 8 + (2 + k)][r + 15] = V_s[j * 2 + k][r]
                    W[(2 + k) * 2 + i][r + 15] = W_s[k * 2 + i][r]
    
    # Process k-block 1 (k=4..7)
    # j=0: direct 2x1x4
    for i in range(2):
        for k in range(4):
            r = i * 4 + k + 22
            U[i * 3 + 0][r] = 1
            V[0 * 8 + (4 + k)][r] = 1
            W[(4 + k) * 2 + i][r] = 1
    
    # j=1,2: Strassen 2x2x4
    # For k=4,5: one Strassen 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    U[i * 3 + (1 + j)][r + 30] = U_s[i * 2 + j][r]
                    V[(1 + j) * 8 + (4 + k)][r + 30] = V_s[j * 2 + k][r]
                    W[(4 + k) * 2 + i][r + 30] = W_s[k * 2 + i][r]
    
    # For k=6,7: another Strassen 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    U[i * 3 + (1 + j)][r + 37] = U_s[i * 2 + j][r]
                    V[(1 + j) * 8 + (6 + k)][r + 37] = V_s[j * 2 + k][r]
                    W[(6 + k) * 2 + i][r + 37] = W_s[k * 2 + i][r]
    
    # Verify the decomposition
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If verification failed, return schoolbook
    U_fb, V_fb, W_fb = make_schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
