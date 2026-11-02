import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct a decomposition of rank 33 for the 2x3x6 matrix multiplication tensor.
    # This is achieved by splitting the intermediate dimension m=3 into m1=1 and m2=2.
    # The multiplication A (2x3) * B (3x6) -> C (2x6) is split into:
    # C1 = A1 * B1 (where A1 is 2x1, B1 is 1x6)
    # C2 = A2 * B2 (where A2 is 2x2, B2 is 2x6)
    #
    # Rank of M_{2,1,6} is 2*1*6 = 12 (since one dimension is 1, it's just element-wise scaling, no Strassen benefit).
    # Actually, for M_{n,1,p}, the rank is n*p. Here n=2, p=6, so rank 12.
    # Rank of M_{2,2,6} can be split p=6 into three blocks of 2: 3 * M_{2,2,2}.
    # Rank of M_{2,2,2} is 7 (Strassen).
    # So Rank of M_{2,2,6} is 3 * 7 = 21.
    # Total rank = 12 + 21 = 33.
    
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: M_{2,1,6}
    # A1 is rows 0:2 of A (indices 0, 1 in A's flattened 2x3, cols 0)
    # B1 is row 0 of B (indices 0:6 in B's flattened 3x6, row 0)
    # C1 is C (indices 0:12 in C's flattened 6x2, all cols)
    # Wait, the tensor indices are:
    # A is n x m -> index i*m + j
    # B is m x p -> index j*p + k
    # C is p x n -> index k*n + i
    #
    # For the sub-problem M_{2,1,6}:
    # A_sub is 2x1 (i in 0..1, j=0)
    # B_sub is 1x6 (j=0, k in 0..5)
    # C_sub is 6x2 (k in 0..5, i in 0..1)
    #
    # The schoolbook decomposition for M_{2,1,6} has rank 12.
    # For each i in 0..1, k in 0..5:
    #   U[i*1 + 0, col] = 1  (A index i*m + j -> i*3 + 0)
    #   V[0*6 + k, col] = 1  (B index j*p + k -> 0*6 + k)
    #   W[k*2 + i, col] = 1  (C index k*n + i -> k*2 + i)
    
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: M_{2,2,6}
    # A_sub is 2x2 (i in 0..1, j in 1..2)
    # B_sub is 2x6 (j in 1..2, k in 0..5)
    # C_sub is 6x2 (k in 0..5, i in 0..1)
    #
    # Split p=6 into three blocks of 2: k in {0,1}, {2,3}, {4,5}.
    # For each block, we use Strassen for M_{2,2,2}.
    # Strassen for M_{2,2,2} uses 7 multiplications.
    # The standard Strassen decomposition for 2x2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    #
    # We need to map these to the global indices.
    # For a specific block of B columns (k0, k0+1):
    # B_sub for this block is 2x2:
    # B[1, k0]   B[1, k0+1]
    # B[2, k0]   B[2, k0+1]
    # A_sub is 2x2:
    # A[0, 1]    A[0, 2]
    # A[1, 1]    A[1, 2]
    #
    # The 7 multiplications produce terms that contribute to C[0,0], C[0,1], C[1,0], C[1,1] 
    # (relative to the block, so global C[k0, 0], C[k0, 1], C[k0+1, 0], C[k0+1, 1]).
    
    # Strassen coefficients
    # m1: (A00+A11)(B00+B11)
    # m2: (A10+A11)B00
    # m3: A00(B01-B11)
    # m4: A11(B10-B00)
    # m5: (A00+A01)B11
    # m6: (A10-A00)(B00+B01)
    # m7: (A01-A11)(B10+B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # We will iterate over the 3 blocks of p.
    for block_idx in range(3):
        k0 = block_idx * 2
        k1 = k0 + 1
        
        # m1
        # A: (0,1) and (1,2) -> indices 0*3+1=1, 1*3+2=5
        # B: (1,k0) and (2,k1) -> indices 1*6+k0=6+k0, 2*6+k1=12+k1
        U[1, col] = 1.0
        U[5, col] = 1.0
        V[6 + k0, col] = 1.0
        V[12 + k1, col] = 1.0
        # C: (k0,0), (k1,1)
        W[k0 * 2 + 0, col] = 1.0
        W[k1 * 2 + 1, col] = 1.0
        col += 1
        
        # m2
        # A: (1,1) and (1,2) -> indices 1*3+1=4, 1*3+2=5
        # B: (1,k0) -> index 1*6+k0=6+k0
        U[4, col] = 1.0
        U[5, col] = 1.0
        V[6 + k0, col] = 1.0
        # C: (k0+1, 0)
        W[(k0 + 1) * 2 + 0, col] = 1.0
        col += 1
        
        # m3
        # A: (0,1) -> index 0*3+1=1
        # B: (1,k1) and (2,k1) -> indices 1*6+k1=6+k1, 2*6+k1=12+k1
        U[1, col] = 1.0
        V[6 + k1, col] = 1.0
        V[12 + k1, col] = -1.0
        # C: (k0, 1)
        W[k0 * 2 + 1, col] = 1.0
        col += 1
        
        # m4
        # A: (1,2) -> index 1*3+2=5
        # B: (2,k0) and (1,k0) -> indices 2*6+k0=12+k0, 1*6+k0=6+k0
        U[5, col] = 1.0
        V[12 + k0, col] = 1.0
        V[6 + k0, col] = -1.0
        # C: (k0, 0) and (k0+1, 0)
        W[k0 * 2 + 0, col] = 1.0
        W[(k0 + 1) * 2 + 0, col] = 1.0
        col += 1
        
        # m5
        # A: (0,1) and (0,2) -> indices 0*3+1=1, 0*3+2=2
        # B: (2,k1) -> index 2*6+k1=12+k1
        U[1, col] = 1.0
        U[2, col] = 1.0
        V[12 + k1, col] = 1.0
        # C: (k0, 0) and (k0, 1)
        W[k0 * 2 + 0, col] = -1.0
        W[k0 * 2 + 1, col] = 1.0
        col += 1
        
        # m6
        # A: (1,1) and (0,1) -> indices 1*3+1=4, 0*3+1=1
        # B: (1,k0) and (1,k1) -> indices 1*6+k0=6+k0, 1*6+k1=6+k1
        U[4, col] = 1.0
        U[1, col] = -1.0
        V[6 + k0, col] = 1.0
        V[6 + k1, col] = 1.0
        # C: (k0+1, 1)
        W[(k0 + 1) * 2 + 1, col] = 1.0
        col += 1
        
        # m7
        # A: (0,2) and (1,2) -> indices 0*3+2=2, 1*3+2=5
        # B: (2,k0) and (2,k1) -> indices 2*6+k0=12+k0, 2*6+k1=12+k1
        U[2, col] = 1.0
        U[5, col] = -1.0
        V[12 + k0, col] = 1.0
        V[12 + k1, col] = 1.0
        # C: (k0, 0)
        W[k0 * 2 + 0, col] = 1.0
        col += 1
        
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
