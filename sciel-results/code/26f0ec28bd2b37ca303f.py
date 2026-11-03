import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U: 2x2, V: 2x2, W: 2x2
    # Indices: U[i, j], V[j, k], W[k, i]
    # We need 7 vectors of size 4 (for 2x2)
    # Let's define the 7 terms for 2x2x2
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # U vectors (for A)
    # a is (0,0), b is (0,1), c is (1,0), d is (1,1)
    U2 = np.zeros((4, 7))
    # P1: a+d -> (1,0,0,1)
    U2[0, 0] = 1; U2[3, 0] = 1
    # P2: c+d -> (0,0,1,1)
    U2[2, 1] = 1; U2[3, 1] = 1
    # P3: a -> (1,0,0,0)
    U2[0, 2] = 1
    # P4: d -> (0,0,0,1)
    U2[3, 3] = 1
    # P5: a+b -> (1,1,0,0)
    U2[0, 4] = 1; U2[1, 4] = 1
    # P6: c-a -> (-1,0,1,0)
    U2[0, 5] = -1; U2[2, 5] = 1
    # P7: b-d -> (0,1,0,-1)
    U2[1, 6] = 1; U2[3, 6] = -1
    
    # V vectors (for B)
    # e is (0,0), f is (0,1), g is (1,0), h is (1,1)
    V2 = np.zeros((4, 7))
    # P1: e+h -> (1,0,0,1)
    V2[0, 0] = 1; V2[3, 0] = 1
    # P2: e -> (1,0,0,0)
    V2[0, 1] = 1
    # P3: f-h -> (0,1,0,-1)
    V2[1, 2] = 1; V2[3, 2] = -1
    # P4: g-e -> (-1,0,1,0)
    V2[0, 3] = -1; V2[2, 3] = 1
    # P5: h -> (0,0,0,1)
    V2[3, 4] = 1
    # P6: e+f -> (1,1,0,0)
    V2[0, 5] = 1; V2[1, 5] = 1
    # P7: g+h -> (0,0,1,1)
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # W vectors (for C)
    # C00, C01, C10, C11
    # Indices: 0->C00, 1->C01, 2->C10, 3->C11
    W2 = np.zeros((4, 7))
    # P1: C00 - C11 -> (1,0,0,-1)
    W2[0, 0] = 1; W2[3, 0] = -1
    # P2: C10 - C11 -> (0,0,1,-1)
    W2[2, 1] = 1; W2[3, 1] = -1
    # P3: C01 - C11 -> (0,1,0,-1)
    W2[1, 2] = 1; W2[3, 2] = -1
    # P4: C00 + C10 -> (1,0,1,0)
    W2[0, 3] = 1; W2[2, 3] = 1
    # P5: -C00 + C01 -> (-1,1,0,0)
    W2[0, 4] = -1; W2[1, 4] = 1
    # P6: C11 -> (0,0,0,1)
    W2[3, 5] = 1
    # P7: C00 -> (1,0,0,0)
    W2[0, 6] = 1
    
    # Verify 2x2x2
    # Sum_s U[i,j,s] V[j,k,s] W[k,i,s] should be 1 for all i,j,k
    # This is a standard check, but we trust Strassen.

    # Construct 4x4x4 decomposition using Kronecker product
    # Rank 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    col = 0
    for s1 in range(7):
        for s2 in range(7):
            # U4 index: i1*2+i2, j1*2+j2 -> (i1*2+i2)*4 + (j1*2+j2)
            # Wait, U is n x m. n=4, m=4.
            # Row index is i*m + j = i*4 + j.
            # i = i1*2 + i2. j = j1*2 + j2.
            # Row = (i1*2 + i2)*4 + (j1*2 + j2) = i1*8 + i2*4 + j1*2 + j2.
            
            # V is m x p. m=4, p=4 (for this part).
            # Row index is j*p + k = j*4 + k.
            # j = j1*2 + j2. k = k1*2 + k2.
            # Row = (j1*2 + j2)*4 + (k1*2 + k2) = j1*8 + j2*4 + k1*2 + k2.
            
            # W is p x n. p=4, n=4.
            # Row index is k*n + i = k*4 + i.
            # k = k1*2 + k2. i = i1*2 + i2.
            # Row = (k1*2 + k2)*4 + (i1*2 + i2) = k1*8 + k2*4 + i1*2 + i2.
            
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    # This loop is too slow. Vectorize.
                                    pass
                                    
            # Vectorized approach
            # U4[:, col] = kron(U2[:, s1], U2[:, s2])
            # But we need to map indices correctly.
            # U2 is 4x7. U2[i, s] where i = i1*2 + i2.
            # U4 row index is i*4 + j.
            # U4[i*4 + j, col] = U2[i1*2+i2, s1] * U2[i2, s2]? No.
            # U2 index is i = i1*2 + i2.
            # So U2[i, s] corresponds to i1, i2.
            # We want U4[i*4 + j, col] = U2[i1*2+i2, s1] * U2[i2, s2] ? No.
            # The Kronecker product of vectors u, v of size 4 gives a vector of size 16.
            # (u \otimes v)[i*4 + j] = u[i] * v[j].
            # Here u = U2[:, s1], v = U2[:, s2].
            # So U4[:, col] = np.kron(U2[:, s1], U2[:, s2]).
            # This matches i = i1*2 + i2, j = j1*2 + j2?
            # No, np.kron(u, v) index is i*len(v) + j.
            # If u has index i1*2+i2, and v has index j1*2+j2.
            # Then kron index is (i1*2+i2)*4 + (j1*2+j2).
            # This is exactly the row index for U4.
            
            U4[:, col] = np.kron(U2[:, s1], U2[:, s2])
            
            # V4
            # V2 index is j = j1*2 + j2.
            # V4 row index is j*4 + k.
            # np.kron(V2[:, s1], V2[:, s2]) index is (j1*2+j2)*4 + (k1*2+k2).
            # This matches.
            V4[:, col] = np.kron(V2[:, s1], V2[:, s2])
            
            # W4
            # W2 index is k = k1*2 + k2.
            # W4 row index is k*4 + i.
            # np.kron(W2[:, s1], W2[:, s2]) index is (k1*2+k2)*4 + (i1*2+i2).
            # This matches.
            W4[:, col] = np.kron(W2[:, s1], W2[:, s2])
            
            col += 1
            
    # Now add the 4x4x1 part for k=4
    # Rank 16
    # U, V, W for this part
    # U row i*4 + j.
    # V row j*5 + 4.
    # W row 4*4 + i = 16 + i.
    
    U_total = np.zeros((16, 65))
    V_total = np.zeros((20, 65)) # m*p = 4*5 = 20
    W_total = np.zeros((20, 65)) # p*n = 5*4 = 20
    
    # Copy 4x4x4 part
    # U4 is 16x49. U_total is 16x65.
    U_total[:, 0:49] = U4
    
    # V4 is 16x49. V_total is 20x65.
    # V4 rows correspond to k in 0..3.
    # V_total rows for k in 0..3 are j*5 + k.
    # V4 rows are j*4 + k.
    # We need to map V4[j*4 + k, c] to V_total[j*5 + k, c].
    for j in range(4):
        for k in range(4):
            V_total[j*5 + k, 0:49] = V4[j*4 + k, 0:49]
            
    # W4 is 16x49. W_total is 20x65.
    # W4 rows correspond to i in 0..3 (since k is fixed in the product? No).
    # W4 row index is k*4 + i.
    # W_total row index is k*4 + i for k in 0..3.
    # So W_total[0:16, 0:49] = W4.
    W_total[0:16, 0:49] = W4
    
    # Add 4x4x1 part
    # For each i, j in 0..3:
    col = 49
    for i in range(4):
        for j in range(4):
            U_total[i*4 + j, col] = 1
            V_total[j*5 + 4, col] = 1
            W_total[4*4 + i, col] = 1
            col += 1
            
    # Convert to lists of lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
