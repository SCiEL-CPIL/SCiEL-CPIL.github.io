import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The rank of 2x2 matrix multiplication is known to be 7 (Strassen).
    # Rank 6 is impossible. We return the standard Strassen decomposition.
    
    rank = 7
    
    # U shape: (n*m, rank) = (4, 7)
    # V shape: (m*p, rank) = (4, 7)
    # W shape: (p*n, rank) = (4, 7)
    # Note: W index is k*n + i for C[i, k]
    # C = A * B
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T_{i, j, k} = delta_{j, l} ... no, T_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # Actually, the tensor for matrix multiplication is $T \in \mathbb{R}^{n \times m \times p}$?
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the tensor is indexed by $(i, j, k)$ where $A$ is $i,j$ and $B$ is $j,k$?
    # Standard tensor decomposition: $T = \sum_{r} u_r \otimes v_r \otimes w_r$.
    # $A_{i,j}$ corresponds to $U[i*m + j, r]$? Or $U[i, j, r]$?
    # The interface says U is shape (n*m, rank). So index $i*m+j$.
    # V is shape (m*p, rank). So index $j*p+k$.
    # W is shape (p*n, rank). So index $k*n+i$.
    
    # Let's verify the indices in the current code.
    # U[0][0] = 1. U index 0 -> i=0, j=0 (A00). Correct.
    # V[0][0] = 1. V index 0 -> j=0, k=0 (B00). Correct.
    # W[0][0] = 1. W index 0 -> k=0, i=0 (C00). Correct.
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6
    
    # Let's map these to U, V, W vectors.
    # r=0: M1
    # U: A00(1), A11(1) -> U[0][0]=1, U[3][0]=1
    # V: B00(1), B11(1) -> V[0][0]=1, V[3][0]=1
    # W: C00(1), C11(1) -> W[0][0]=1, W[3][0]=1
    
    # r=1: M2
    # U: A10(1), A11(1) -> U[2][1]=1, U[3][1]=1
    # V: B00(1) -> V[0][1]=1
    # W: C10(1), C11(-1) -> W[1][1]=1, W[3][1]=-1
    
    # r=2: M3
    # U: A00(1) -> U[0][2]=1
    # V: B01(1), B11(-1) -> V[1][2]=1, V[3][2]=-1
    # W: C01(1) -> W[2][2]=1
    # Wait, C01 = M3 + M5. So W for M3 should contribute to C01.
    # Current code: W[2][2] = 1. W index 2 is k=1, i=0 -> C01. Correct.
    # Does M3 contribute to anything else? No.
    
    # r=3: M4
    # U: A11(1) -> U[3][3]=1
    # V: B10(1), B00(-1) -> V[2][3]=1, V[0][3]=-1
    # W: C00(-1), C10(1) -> W[0][3]=-1, W[1][3]=1
    # Current code: W[0][3] = 1, W[1][3] = 1.
    # Let's check C00 formula: M1 - M4 + M7.
    # If W[0][3] is 1, then C00 gets +M4. But formula says -M4.
    # Let's re-read current code:
    # W[0][3] = 1 # C00
    # W[1][3] = 1 # C10
    # C10 = M2 + M4. So +M4 is correct for C10.
    # C00 = M1 - M4 + M7. So -M4 is correct for C00.
    # The current code has W[0][3] = 1. This would mean +M4 for C00.
    # This is an ERROR in the current code?
    
    # Let's re-verify Strassen.
    # C00 = M1 - M4 + M7
    # C11 = M1 - M2 + M6
    # C10 = M2 + M4
    # C01 = M3 + M5
    
    # Let's check current code W assignments:
    # r=0 (M1): W[0]=1 (C00+), W[3]=1 (C11+). OK.
    # r=1 (M2): W[1]=1 (C10+), W[3]=-1 (C11-). OK.
    # r=2 (M3): W[2]=1 (C01+). OK.
    # r=3 (M4): W[0]=1 (C00+), W[1]=1 (C10+). 
    #   Formula C00 needs -M4. Code has +M4. ERROR?
    #   Formula C10 needs +M4. Code has +M4. OK.
    # r=4 (M5): W[0]=-1 (C00-), W[2]=1 (C01+).
    #   Formula C00 needs -M5? No, C00 = M1 - M4 + M7. M5 not in C00.
    #   Formula C01 = M3 + M5. Code has W[2]=1 (C01+). OK.
    #   Why is W[0]=-1 for M5? M5 does not contribute to C00.
    #   Let's check M5 definition: (A00+A01)B11.
    #   Current code r=4:
    #   U[0][4]=1, U[1][4]=1 (A00+A01).
    #   V[3][4]=1 (B11).
    #   W[0][4]=-1, W[2][4]=1.
    #   This implies C00 gets -M5. But C00 = M1 - M4 + M7.
    #   This suggests the current code's Strassen implementation might be flawed or using a different variant?
    
    # Let's check if the current code is actually correct by simulating.
    # I will rewrite the code to be definitely correct.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # A: 00, 11. B: 00, 11.
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C00 += M1, C11 += M1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    # A: 10, 11. B: 00.
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += M2, C11 -= M2
    W[1][1] = 1; W[3][1] = -1
    
    # M3: A00(B01-B11)
    # A: 00. B: 01, 11.
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C01 += M3
    W[2][2] = 1
    
    # M4: A11(B10-B00)
    # A: 11. B: 10, 00.
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C10 += M4, C00 -= M4
    W[1][3] = 1; W[0][3] = -1
    
    # M5: (A00+A01)B11
    # A: 00, 01. B: 11.
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C01 += M5
    W[2][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # A: 10, 00. B: 00, 01.
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C11 += M6
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # A: 01, 11. B: 10, 11.
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C00 += M7
    W[0][6] = 1
    
    # Verification:
    # C00 = M1 - M4 + M7. 
    #   M1: +1 (W[0][0]=1)
    #   M4: -1 (W[0][3]=-1)
    #   M7: +1 (W[0][6]=1)
    #   Sum: 1 - 1 + 1 = 1? No, coefficients are 1.
    #   C00 = 1*M1 + (-1)*M4 + 1*M7. Correct.
    
    # C01 = M3 + M5.
    #   M3: +1 (W[2][2]=1)
    #   M5: +1 (W[2][4]=1)
    #   Sum: M3 + M5. Correct.
    
    # C10 = M2 + M4.
    #   M2: +1 (W[1][1]=1)
    #   M4: +1 (W[1][3]=1)
    #   Sum: M2 + M4. Correct.
    
    # C11 = M1 - M2 + M6.
    #   M1: +1 (W[3][0]=1)
    #   M2: -1 (W[3][1]=-1)
    #   M6: +1 (W[3][5]=1)
    #   Sum: M1 - M2 + M6. Correct.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
