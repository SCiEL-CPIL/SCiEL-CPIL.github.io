import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create 2x2x2 Strassen decomposition
    # Returns U, V, W of shape (4, 7)
    def strassen_2x2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = 1
        
        # M3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-f)
        U[3, 3] = 1
        V[2, 3] = 1; V[1, 3] = -1
        W[1, 3] = 1; W[3, 3] = 1
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[2, 4] = 1; W[3, 4] = -1
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[3, 6] = 1
        
        return U, V, W

    # Helper to create 4x4x4 decomposition using Strassen
    # Returns U, V, W of shape (16, 49)
    def strassen_4x4x4():
        U_sub, V_sub, W_sub = strassen_2x2x2()
        
        U = np.zeros((16, 49), dtype=int)
        V = np.zeros((16, 49), dtype=int)
        W = np.zeros((16, 49), dtype=int)
        
        # Block indices for 4x4 matrix
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        
        # Outer Strassen terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B12)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map these to the sub-decompositions.
        # Each term is a product of sums of blocks.
        # E.g. M1 = (A11+A22)(B11+B22)
        # This is not a single tensor product. It's a sum of 4 products.
        # (A11+A22) is a sum of two 2x2 matrices.
        # The term in the tensor decomposition corresponds to the bilinear form.
        # The rank-49 decomposition is obtained by taking the Kronecker product of the two Strassen decompositions?
        # No, it's the tensor product of the decompositions.
        
        # Let's use the property: Rank(M_{4,4,4}) <= Rank(M_{2,2,2}) * Rank(M_{2,2,2}) = 49.
        # The decomposition is given by the Kronecker product of the index sets.
        
        # U_4x4x4[i1*2+i2, j1*2+j2, r1*7+r2] = U_2x2x2[i1, j1, r1] * U_2x2x2[i2, j2, r2]?
        # No, the indices are mixed.
        
        # Let's construct it explicitly.
        # U has rows indexed by (i, j) where i,j in 0..3.
        # i = i1*2 + i2, j = j1*2 + j2.
        # U[i, j, r] where r = r1*7 + r2.
        # U[i1*2+i2, j1*2+j2, r1*7+r2] = U_sub[i1, j1, r1] * U_sub[i2, j2, r2]
        
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        i = i1*2 + i2
                        j = j1*2 + j2
                        for r1 in range(7):
                            for r2 in range(7):
                                r = r1*7 + r2
                                U[i*4+j, r] = U_sub[i1, j1, r1] * U_sub[i2, j2, r2]
                                V[i*4+j, r] = V_sub[i1, j1, r1] * V_sub[i2, j2, r2]
                                # W index: k*4+i.
                                # k = k1*2 + k2.
                                # W[k1*2+k2, i1*2+i2, r] = W_sub[k1, i1, r1] * W_sub[k2, i2, r2]
                                # But W_sub is indexed by (k, i) in 0..1.
                                # W_sub[k, i, r].
                                # So W[k1*2+k2, i1*2+i2, r1*7+r2] = W_sub[k1, i1, r1] * W_sub[k2, i2, r2]
                                # Wait, W_sub shape is (4, 7). Index k*2+i.
                                for k1 in range(2):
                                    for k2 in range(2):
                                        k = k1*2 + k2
                                        w_idx = k*4 + i
                                        U_val = U_sub[i1, j1, r1] * U_sub[i2, j2, r2]
                                        V_val = V_sub[i1, j1, r1] * V_sub[i2, j2, r2]
                                        W_val = W_sub[k1*2+i1, r1] * W_sub[k2*2+i2, r2]
                                        # This is not right. The Kronecker product structure is:
                                        # T = T1 \otimes T2.
                                        # T_{a1,a2, b1,b2, c1,c2} = T1_{a1,b1,c1} T2_{a2,b2,c2}.
                                        # Here a=(i,j), b=(j,k), c=(k,i).
                                        # a1=(i1,j1), a2=(i2,j2).
                                        # b1=(j1,k1), b2=(j2,k2).
                                        # c1=(k1,i1), c2=(k2,i2).
                                        # So U[i1*2+i2, j1*2+j2, r] = U1[i1,j1,r1] U2[i2,j2,r2]
                                        # V[j1*2+j2, k1*2+k2, r] = V1[j1,k1,r1] V2[j2,k2,r2]
                                        # W[k1*2+k2, i1*2+i2, r] = W1[k1,i1,r1] W2[k2,i2,r2]
                                        
                                        # We need to fill U, V, W.
                                        # U row: i1*2+i2, j1*2+j2.
                                        # V row: j1*2+j2, k1*2+k2.
                                        # W row: k1*2+k2, i1*2+i2.
                                        
                                        # Note: The loop over k1,k2 is inside, but W depends on k.
                                        # We should loop over all indices.
        
        # Let's redo the filling logic cleanly.
        U = np.zeros((16, 49), dtype=int)
        V = np.zeros((16, 49), dtype=int)
        W = np.zeros((16, 49), dtype=int)
        
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        for k1 in range(2):
                            for k2 in range(2):
                                for r1 in range(7):
                                    for r2 in range(7):
                                        r = r1*7 + r2
                                        u_val = U_sub[i1*2+j1, r1] * U_sub[i2*2+j2, r2]
                                        v_val = V_sub[j1*2+k1, r1] * V_sub[j2*2+k2, r2]
                                        w_val = W_sub[k1*2+i1, r1] * W_sub[k2*2+i2, r2]
                                        
                                        u_idx = (i1*2+i2)*4 + (j1*2+j2)
                                        v_idx = (j1*2+j2)*4 + (k1*2+k2)
                                        w_idx = (k1*2+k2)*4 + (i1*2+i2)
                                        
                                        U[u_idx, r] += u_val
                                        V[v_idx, r] += v_val
                                        W[w_idx, r] += w_val
                                        
        return U, V, W

    # Helper for 4x4x1
    def schoolbook_4x4x1():
        U = np.zeros((16, 16), dtype=int)
        V = np.zeros((4, 16), dtype=int)
        W = np.zeros((4, 16), dtype=int)
        
        for i in range(4):
            for j in range(4):
                r = i*4 + j
                U[i*4+j, r] = 1
                V[j, r] = 1
                W[i, r] = 1
        return U, V, W

    U1, V1, W1 = strassen_4x4x4()
    U2, V2, W2 = strassen_4x4x4()
    U3, V3, W3 = schoolbook_4x4x1()
    
    # Combine
    # U: (16, 114)
    U = np.hstack([U1, U2, U3])
    
    # V: (36, 114)
    # V1 corresponds to k in 0..3. V1 shape (16, 49).
    # V2 corresponds to k in 4..7.
    # V3 corresponds to k in 8.
    
    V = np.zeros((36, 114), dtype=int)
    # V1
    for j in range(4):
        for k in range(4):
            V[j*9+k, :49] = V1[j*4+k, :]
            
    # V2
    for j in range(4):
        for k in range(4):
            V[j*9+(k+4), 49:98] = V2[j*4+k, :]
            
    # V3
    for j in range(4):
        V[j*9+8, 98:114] = V3[j, :]
        
    # W: (36, 114)
    W = np.zeros((36, 114), dtype=int)
    # W1
    for k in range(4):
        for i in range(4):
            W[k*4+i, :49] = W1[k*4+i, :]
            
    # W2
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W2[k*4+i, :]
            
    # W3
    for i in range(4):
        W[8*4+i, 98:114] = W3[i, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
