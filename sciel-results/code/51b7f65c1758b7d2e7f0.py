import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We will define the 4 Strassen blocks.
    # Each block uses 7 scalar multiplications.
    # Total rank = 4 * 7 = 28.
    
    # Helper to add a term to U, V, W
    # u_idx: index in U (row * m + col)
    # v_idx: index in V (row * p + col)
    # w_idx: index in W (col * n + row)  <-- Note: W[k*n+i, r] means row=k*n+i, col=r in standard matrix mult, 
    # but the task says W[k*n+i, r] is the factor. 
    # The decomposition is T = sum_r U[:,r] V[:,r] W[:,r]^T ? 
    # Standard tensor decomposition: T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # Here indices are:
    # U corresponds to A (n x m) -> index i in [0, n*m)
    # V corresponds to B (m x p) -> index j in [0, m*p)
    # W corresponds to C (n x p) -> index k in [0, n*p)
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the index for W is `idx = k*n + i`? 
    # Let's check the dimensions. C is n x p. 
    # Usually C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Tensor T_{i,j,k}.
    # U_{i,r}, V_{j,r}, W_{k,r}.
    # The prompt says W has shape (p*n, rank). 
    # If W is indexed by `k*n + i`, that looks like it's transposed or specific to the convention.
    # Let's look at the previous code's W indexing:
    # `w_idx = C_cols[0] * n + C_rows[0]`
    # C_rows is the row of C (i), C_cols is the col of C (k).
    # So w_idx = k * n + i.
    # This matches the prompt "W[k*n+i,r]".
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A rows 0,1; cols 0,1. B rows 0,1; cols 0,1. C rows 0,1; cols 0,1.
    # Strassen M1..M7
    # Base col for this block in U,V,W is 0..6
    
    # M1: (A00+A11)(B00+B11)
    U[0*4+0][0] += 1; U[1*4+1][0] += 1
    V[0*4+0][0] += 1; V[1*4+1][0] += 1
    W[0*2+0][0] += 1 # C00 (k=0, i=0) -> 0*2+0
    W[1*2+1][0] += 1 # C11 (k=1, i=1) -> 1*2+1
    
    # M2: (A10+A11)B00
    U[1*4+0][1] += 1; U[1*4+1][1] += 1
    V[0*4+0][1] += 1
    W[0*2+1][1] += 1 # C10 (k=0, i=1) -> 0*2+1
    W[1*2+1][1] -= 1 # C11 (k=1, i=1) -> 1*2+1
    
    # M3: A00(B01-B11)
    U[0*4+0][2] += 1
    V[0*4+1][2] += 1; V[1*4+1][2] -= 1
    W[1*2+0][2] += 1 # C01 (k=1, i=0) -> 1*2+0
    W[1*2+1][2] += 1 # C11 (k=1, i=1) -> 1*2+1
    
    # M4: A11(B10-B00)
    U[1*4+1][3] += 1
    V[1*4+0][3] += 1; V[0*4+0][3] -= 1
    W[0*2+0][3] += 1 # C00
    W[0*2+1][3] += 1 # C10
    
    # M5: (A00+A01)B11
    U[0*4+0][4] += 1; U[0*4+1][4] += 1
    V[1*4+1][4] += 1
    W[0*2+0][4] -= 1 # C00
    W[1*2+0][4] += 1 # C01
    
    # M6: (A10-A00)(B00+B01)
    U[1*4+0][5] += 1; U[0*4+0][5] -= 1
    V[0*4+0][5] += 1; V[0*4+1][5] += 1
    W[1*2+1][5] += 1 # C11
    
    # M7: (A01-A11)(B10+B11)
    U[0*4+1][6] += 1; U[1*4+1][6] -= 1
    V[1*4+0][6] += 1; V[1*4+1][6] += 1
    W[0*2+0][6] += 1 # C00
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A cols 2,3. B rows 2,3. C cols 0,1.
    # Base col 7..13
    # M1: (A02+A13)(B20+B31)
    U[0*4+2][7] += 1; U[1*4+3][7] += 1
    V[2*4+0][7] += 1; V[3*4+1][7] += 1
    W[0*2+0][7] += 1 # C00
    W[1*2+1][7] += 1 # C11
    
    # M2: (A12+A13)B20
    U[1*4+2][8] += 1; U[1*4+3][8] += 1
    V[2*4+0][8] += 1
    W[0*2+1][8] += 1 # C10
    W[1*2+1][8] -= 1 # C11
    
    # M3: A02(B21-B31)
    U[0*4+2][9] += 1
    V[2*4+1][9] += 1; V[3*4+1][9] -= 1
    W[1*2+0][9] += 1 # C01
    W[1*2+1][9] += 1 # C11
    
    # M4: A13(B30-B20)
    U[1*4+3][10] += 1
    V[3*4+0][10] += 1; V[2*4+0][10] -= 1
    W[0*2+0][10] += 1 # C00
    W[0*2+1][10] += 1 # C10
    
    # M5: (A02+A03)B31
    U[0*4+2][11] += 1; U[0*4+3][11] += 1
    V[3*4+1][11] += 1
    W[0*2+0][11] -= 1 # C00
    W[1*2+0][11] += 1 # C01
    
    # M6: (A12-A02)(B20+B21)
    U[1*4+2][12] += 1; U[0*4+2][12] -= 1
    V[2*4+0][12] += 1; V[2*4+1][12] += 1
    W[1*2+1][12] += 1 # C11
    
    # M7: (A03-A13)(B30+B31)
    U[0*4+3][13] += 1; U[1*4+3][13] -= 1
    V[3*4+0][13] += 1; V[3*4+1][13] += 1
    W[0*2+0][13] += 1 # C00

    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A cols 0,1. B cols 2,3. C cols 2,3.
    # Base col 14..20
    # M1: (A00+A11)(B02+B13)
    U[0*4+0][14] += 1; U[1*4+1][14] += 1
    V[0*4+2][14] += 1; V[1*4+3][14] += 1
    W[2*2+0][14] += 1 # C02 (k=2, i=0) -> 2*2+0=4
    W[3*2+1][14] += 1 # C13 (k=3, i=1) -> 3*2+1=7
    
    # M2: (A10+A11)B02
    U[1*4+0][15] += 1; U[1*4+1][15] += 1
    V[0*4+2][15] += 1
    W[2*2+1][15] += 1 # C12 (k=2, i=1) -> 2*2+1=5
    W[3*2+1][15] -= 1 # C13
    
    # M3: A00(B03-B13)
    U[0*4+0][16] += 1
    V[0*4+3][16] += 1; V[1*4+3][16] -= 1
    W[3*2+0][16] += 1 # C03 (k=3, i=0) -> 3*2+0=6
    W[3*2+1][16] += 1 # C13
    
    # M4: A11(B12-B02)
    U[1*4+1][17] += 1
    V[1*4+2][17] += 1; V[0*4+2][17] -= 1
    W[2*2+0][17] += 1 # C02
    W[2*2+1][17] += 1 # C12
    
    # M5: (A00+A01)B13
    U[0*4+0][18] += 1; U[0*4+1][18] += 1
    V[1*4+3][18] += 1
    W[2*2+0][18] -= 1 # C02
    W[3*2+0][18] += 1 # C03
    
    # M6: (A10-A00)(B02+B03)
    U[1*4+0][19] += 1; U[0*4+0][19] -= 1
    V[0*4+2][19] += 1; V[0*4+3][19] += 1
    W[3*2+1][19] += 1 # C13
    
    # M7: (A01-A11)(B12+B13)
    U[0*4+1][20] += 1; U[1*4+1][20] -= 1
    V[1*4+2][20] += 1; V[1*4+3][20] += 1
    W[2*2+0][20] += 1 # C02

    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A cols 2,3. B rows 2,3; cols 2,3. C cols 2,3.
    # Base col 21..27
    # M1: (A02+A13)(B22+B33)
    U[0*4+2][21] += 1; U[1*4+3][21] += 1
    V[2*4+2][21] += 1; V[3*4+3][21] += 1
    W[2*2+0][21] += 1 # C02
    W[3*2+1][21] += 1 # C13
    
    # M2: (A12+A13)B22
    U[1*4+2][22] += 1; U[1*4+3][22] += 1
    V[2*4+2][22] += 1
    W[2*2+1][22] += 1 # C12
    W[3*2+1][22] -= 1 # C13
    
    # M3: A02(B23-B33)
    U[0*4+2][23] += 1
    V[2*4+3][23] += 1; V[3*4+3][23] -= 1
    W[3*2+0][23] += 1 # C03
    W[3*2+1][23] += 1 # C13
    
    # M4: A13(B32-B22)
    U[1*4+3][24] += 1
    V[3*4+2][24] += 1; V[2*4+2][24] -= 1
    W[2*2+0][24] += 1 # C02
    W[2*2+1][24] += 1 # C12
    
    # M5: (A02+A03)B33
    U[0*4+2][25] += 1; U[0*4+3][25] += 1
    V[3*4+3][25] += 1
    W[2*2+0][25] -= 1 # C02
    W[3*2+0][25] += 1 # C03
    
    # M6: (A12-A02)(B22+B23)
    U[1*4+2][26] += 1; U[0*4+2][26] -= 1
    V[2*4+2][26] += 1; V[2*4+3][26] += 1
    W[3*2+1][26] += 1 # C13
    
    # M7: (A03-A13)(B32+B33)
    U[0*4+3][27] += 1; U[1*4+3][27] -= 1
    V[3*4+2][27] += 1; V[3*4+3][27] += 1
    W[2*2+0][27] += 1 # C02

    return {"U": U, "V": V, "W": W}
