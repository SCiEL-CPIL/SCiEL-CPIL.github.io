import numpy as np

DIMENSIONS = [3, 3, 4]

def construct_3x3x2():
    # Returns U (9x17), V (6x17), W (6x17)
    # U index: i*3 + j (0..8)
    # V index: j*2 + k (0..5)
    # W index: k*3 + i (0..5) -- Wait, trace convention W[k*n+i, r]
    # n=3, p=2. W is p*n = 2*3 = 6.
    # Index in W: k*3 + i.
    
    U = np.zeros((9, 17))
    V = np.zeros((6, 17))
    W = np.zeros((6, 17))
    
    # Strassen part for i,j in {0,1}, k in {0,1}
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map these to U, V, W.
    # U col c: coefficients for A_ij
    # V col c: coefficients for B_jk
    # W col c: coefficients for C_ik (output)
    
    # Let's define the linear forms for A and B for each M.
    # And the resulting contribution to C.
    
    # M1: A: (1,0,0,1,0,0,0,0,0) -> indices 0, 3 (00, 11)
    #     B: (1,0,1,0,0,0) -> indices 0, 2 (00, 10? No, B is 3x2. j*2+k.
    #     B indices: j=0,k=0 -> 0. j=1,k=0 -> 2.
    #     Wait, B is 3x2. j in 0..2, k in 0..1.
    #     B_{00} is index 0. B_{10} is index 2.
    #     So B1 = B_{00} + B_{10}? No, Strassen uses B_{11}, B_{22} etc.
    #     In 2x2 case:
    #     A = [[A00, A01], [A10, A11]]
    #     B = [[B00, B01], [B10, B11]]
    #     Here B is 3x2. The 2x2 block is B[0:2, 0:2].
    #     So B_{00}, B_{01}, B_{10}, B_{11}.
    #     Indices in V (size 6):
    #     j=0,k=0 -> 0
    #     j=0,k=1 -> 1
    #     j=1,k=0 -> 2
    #     j=1,k=1 -> 3
    #     j=2,k=0 -> 4
    #     j=2,k=1 -> 5
    
    # Strassen M1: (A00+A11)(B00+B11)
    # U: A00 (idx 0), A11 (idx 4? No, A is 3x3. i*3+j.
    # A00 -> 0. A11 -> 1*3+1 = 4.
    # V: B00 (idx 0), B11 (idx 3).
    # W: Contribution to C00, C01, C10, C11.
    # C indices: k*3+i.
    # C00 (k=0,i=0) -> 0
    # C01 (k=0,i=1) -> 1
    # C10 (k=1,i=0) -> 3
    # C11 (k=1,i=1) -> 4
    # M1 contributes to C00, C11 with +1, C01, C10 with -1?
    # Standard Strassen:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # So for M1:
    # W[0] += 1 (C00)
    # W[1] += 0? No, C01 is M3+M5. M1 is not in C01.
    # W[3] += 0? C10 is M2+M4. M1 is not in C10.
    # W[4] += 1 (C11)
    
    # Let's verify the Strassen formulas.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # This looks correct.
    
    # I will code this up.
    
    # Then add 10 schoolbook terms.
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Use 34 rank solution
    U1, V1, W1 = construct_3x3x2()
    # Shift indices for second block
    # V indices for second block: j*2 + (k+2)
    # W indices for second block: (k+2)*3 + i
    
    U = np.zeros((9, 34))
    V = np.zeros((12, 34)) # m*p = 3*4 = 12
    W = np.zeros((12, 34)) # p*n = 4*3 = 12
    
    U[:, 0:17] = U1
    V[0:6, 0:17] = V1
    W[0:6, 0:17] = W1
    
    # Second block
    U[:, 17:34] = U1
    # V shift: k -> k+2. Index j*2 + k.
    # New index j*2 + (k+2) = j*2 + k + 2.
    V[2:8, 17:34] = V1 # Wait, V1 is 6x17.
    # V1 indices 0..5 correspond to j=0..2, k=0..1.
    # We want to place them at j=0..2, k=2..3.
    # Index = j*4 + k? No, V is m*p = 3*4 = 12.
    # Index = j*p + k = j*4 + k.
    # So for k=2,3, indices are j*4+2, j*4+3.
    # V1 index j*2+k.
    # We need to map V1[j*2+k] to V[j*4+k+2].
    
    for j in range(3):
        for k in range(2):
            V[j*4 + k + 2, 17:34] = V1[j*2 + k, :]
            
    # W shift: k -> k+2. Index k*3 + i.
    # New index (k+2)*3 + i = k*3 + i + 6.
    # W1 index k*3 + i.
    W[6:12, 17:34] = W1
    
    return {"U": U, "V": V, "W": W}
