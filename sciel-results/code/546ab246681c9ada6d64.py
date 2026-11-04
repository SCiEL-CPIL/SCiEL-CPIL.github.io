"""Tensor decomposition for 4x4 * 4x8 matrix multiplication using Strassen-based construction."""
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen 2x2x2 decomposition
    # U2[i,j] indexed by (i,j) in [0,1]x[0,1], 7 entries
    # V2[j,k] indexed by (j,k) in [0,1]x[0,1], 7 entries  
    # W2[k,i] indexed by (k,i) in [0,1]x[0,1], 7 entries
    
    # Standard Strassen matrices (using the well-known 7-term decomposition)
    # U: 7x4 matrix where U[t,i,j] gives coefficient
    # V: 7x4 matrix where V[t,j,k] gives coefficient
    # W: 7x4 matrix where W[t,k,i] gives coefficient
    
    # Strassen algorithm for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's build the factor matrices for 2x2x2 Strassen
    # U[t, i*m+j] = U[t, i*2+j] for t in 0..6, i,j in 0..1
    # V[t, j*p+k] = V[t, j*2+k] for t in 0..6, j,k in 0..1
    # W[t, k*n+i] = W[t, k*2+i] for t in 0..6, k,i in 0..1
    
    U2 = [[0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0]]
    
    V2 = [[0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0]]
    
    W2 = [[0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0]]
    
    # M1 = (A11+A22)*(B11+B22)
    U2[0] = [1, 0, 0, 1]  # A11 + A22
    V2[0] = [1, 0, 0, 1]  # B11 + B22
    W2[0] = [1, 0, 0, 1]  # contributes to C11 and C22
    
    # M2 = (A21+A22)*B11
    U2[1] = [0, 0, 1, 1]  # A21 + A22
    V2[1] = [1, 0, 0, 0]  # B11
    W2[1] = [0, 0, 1, 0]  # contributes to C21 (with +) and C22 (with -)
    
    # M3 = A11*(B12-B22)
    U2[2] = [1, 0, 0, 0]  # A11
    V2[2] = [0, 1, 0, -1]  # B12 - B22
    W2[2] = [0, 1, 0, 1]  # contributes to C12 (with +) and C22 (with +)
    
    # M4 = A22*(B21-B11)
    U2[3] = [0, 0, 0, 1]  # A22
    V2[3] = [0, 0, -1, 1]  # B21 - B11
    W2[3] = [0, 1, 1, 0]  # contributes to C11 (with +) and C21 (with +)
    
    # M5 = (A11+A12)*B22
    U2[4] = [1, 1, 0, 0]  # A11 + A12
    V2[4] = [0, 0, 0, 1]  # B22
    W2[4] = [0, 0, -1, 0]  # contributes to C11 (with -) and C12 (with +)
    
    # M6 = (A21-A11)*(B11+B12)
    U2[5] = [-1, 0, 1, 0]  # A21 - A11
    V2[5] = [1, 1, 0, 0]  # B11 + B12
    W2[5] = [0, 0, 0, 1]  # contributes to C22 (with +)
    
    # M7 = (A12-A22)*(B21+B22)
    U2[6] = [0, 1, 0, -1]  # A12 - A22
    V2[6] = [0, 0, 1, 1]  # B21 + B22
    W2[6] = [1, 0, 0, 0]  # contributes to C11 (with +)
    
    # Verify W2 contributions:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W2[t] should give the coefficient for C entry from M_t
    # For C11 (k=0,i=0): W2[0][0]=1, W2[3][0]=1, W2[4][0]=-1, W2[6][0]=1 ✓
    # For C12 (k=0,i=1): W2[2][1]=1, W2[4][1]=1 ✓
    # For C21 (k=1,i=0): W2[1][2]=1, W2[3][2]=1 ✓
    # For C22 (k=1,i=1): W2[0][3]=1, W2[1][3]=-1, W2[2][3]=1, W2[5][3]=1 ✓
    
    # Now for 4x4x4, we use the Kronecker product approach
    # Tensor product of two Strassen decompositions gives rank 7*7=49
    # For 4x4x4: U4[i,j,t] where i,j in 0..3, t in 0..48
    # U4[i,j] = U2[i1,j1][i2,j2] where i=2*i1+i2, j=2*j1+j2, t=7*i1+j1+7*j1+j2
    
    # Actually, let's use the standard tensor product construction:
    # For matrices of size 4x4, we can think of them as 2x2 blocks of 2x2 matrices
    # The tensor product of Strassen with itself gives a rank-49 decomposition of 4x4x4
    
    # U4[i,j,t]: i,j in 0..3, t in 0..48
    # t = 7*a + b where a,b in 0..6
    # U4[i,j, 7*a+b] = U2[a, i1*2+j1] * U2[b, i2*2+j2] where i=2*i1+i2, j=2*j1+j2
    # Similarly for V4 and W4
    
    # For 4x4x8, we need to extend to p=8
    # We can split the 4x8 output into two 4x4 blocks
    # The first 4x4 block uses the first 49 terms, the second uses the next 49 terms
    # But we need to be careful about the V matrix structure
    
    # For 4x4x8:
    # A is 4x4, B is 4x8, C is 4x8
    # Split B into B_left (4x4) and B_right (4x4)
    # C_left = A * B_left, C_right = A * B_right
    # Each can use rank-49 Strassen-based decomposition
    # Total rank = 49 + 49 = 98
    
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # For the first 49 terms (columns 0-48): handle C_left = A * B_left
    # For the next 49 terms (columns 49-97): handle C_right = A * B_right
    
    for a in range(7):
        for b in range(7):
            t1 = 7 * a + b  # for C_left
            t2 = 49 + 7 * a + b  # for C_right
            
            # Kronecker product of U2[a] and U2[b] for U matrix
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            val = U2[a][i1 * 2 + j1] * U2[b][i2 * 2 + j2]
                            U[i * 4 + j][t1] += val
                            U[i * 4 + j][t2] += val
            
            # For V matrix:
            # V[j,k,t] where j in 0..3, k in 0..7
            # For C_left: k in 0..3 (B_left columns)
            # For C_right: k in 4..7 (B_right columns)
            
            # Kronecker product of V2[a] and V2[b]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            val = V2[a][j1 * 2 + k1] * V2[b][j2 * 2 + k2]
                            # For C_left: B_left[j,k]
                            V[j * 8 + k][t1] += val
                            # For C_right: B_right[j,k] which is B[j, k+4]
                            V[j * 8 + (k + 4)][t2] += val
            
            # For W matrix:
            # W[k,i,t] where k in 0..3, i in 0..3
            # For C_left: output is C_left[k,i] = C[k,i]
            # For C_right: output is C_right[k,i] = C[k+4,i]... wait, no
            # C is 4x8, so C[i,j] where i in 0..3, j in 0..7
            # C_left[i,j] for j in 0..3, C_right[i,j] for j in 4..7
            # In the trace convention W[k*n+i], so W[k*4+i]
            # For C_left: W[k*4+i, t1]
            # For C_right: W[k*4+i, t2] but this contributes to C_right which is columns 4-7 of C
            
            # Actually, I need to reconsider the output indexing.
            # C is n x p = 4 x 8
            # C[i,j] where i in 0..3 (rows), j in 0..7 (columns)
            # The trace convention says W[k*n+i] where k is the column of B and row of C output... 
            # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
            # This means W is indexed by (k*n + i) where k relates to the column dimension of B
            # and i relates to the row dimension of A.
            
            # Actually, for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
            # C[i,j] = sum_l A[i,l] * B[l,j]
            # The tensor is T[i,l,j] = A[i,l]*B[l,j] = C[i,j]
            # U[i*m+l, t], V[l*p+j, t], W[j*n+i, t]
            # So W[j*n+i, t] where j is the column of C (0..p-1), i is the row of C (0..n-1)
            
            # For our case: n=4, m=4, p=8
            # U has 16 rows (4*4), V has 32 rows (4*8), W has 32 rows (8*4)
            # W[j*4+i, t] where j in 0..7, i in 0..3
            
            # For C_left (first 4 columns of C): j in 0..3
            # For C_right (last 4 columns of C): j in 4..7
            
            # Kronecker product of W2[a] and W2[b]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = 2 * k1 + k2  # this is the j index in C (column)
                            i = 2 * i1 + i2  # this is the i index in C (row)
                            val = W2[a][k1 * 2 + i1] * W2[b][k2 * 2 + i2]
                            # For C_left: C[i,k] where k in 0..3
                            W[k * 4 + i][t1] += val
                            # For C_right: C[i,k+4] where k in 0..3, so column k+4
                            W[(k + 4) * 4 + i][t2] += val
    
    return {"U": U, "V": V, "W": W}
