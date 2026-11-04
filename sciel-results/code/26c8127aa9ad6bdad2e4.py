import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We use the decomposition M_{2,4,8} <= M_{2,2,4} x M_{1,2,2}
    # M_{2,2,4} <= M_{2,2,2} x M_{1,1,2}
    # So M_{2,4,8} <= M_{2,2,2} x M_{1,1,2} x M_{1,2,2}
    # Rank = 7 * 2 * 4 = 56
    
    # 1. Construct M_{2,2,2} (Strassen, rank 7)
    # n1=2, m1=2, p1=2
    n1, m1, p1 = 2, 2, 2
    r1 = 7
    U1 = [[0]*r1 for _ in range(n1*m1)]
    V1 = [[0]*r1 for _ in range(m1*p1)]
    W1 = [[0]*r1 for _ in range(p1*n1)]
    
    # Strassen terms
    # M1 = (a+d)(e+h)
    # a: A[0,0] idx 0*2+0=0. d: A[1,1] idx 1*2+1=3.
    # e: B[0,0] idx 0*2+0=0. h: B[1,1] idx 1*2+1=3.
    # C11: C[0,0] idx 0*2+0=0.
    # We need to add contributions to W.
    
    # Helper to add a term U_idx * V_idx to W_idx with coeff
    def add_term(u_idx, v_idx, w_idx, coeff):
        # This is for a single rank-1 tensor in the decomposition of M_{2,2,2}
        # But Strassen is a sum of 7 rank-1 terms? No, Strassen is 7 products.
        # Each product is a rank-1 tensor in the "multiplication" sense?
        # Yes, each M_k is a product of two linear forms.
        # So we can just list the 7 rank-1 tensors.
        pass

    # The 7 rank-1 tensors for Strassen:
    # 1. (a+d)(e+h) -> C11, C22
    # U: a(0), d(3). V: e(0), h(3).
    # This is not a single rank-1 tensor for U and V.
    # U is a vector, V is a vector.
    # For M1, U1 = [1, 0, 0, 1] (a+d). V1 = [1, 0, 0, 1] (e+h).
    # W1 must produce C11 and C22.
    # C11 += M1. C22 += M1.
    # W1 indices: C11 is idx 0 (k=0, i=0 -> 0*2+0=0). C22 is idx 3 (k=1, i=1 -> 1*2+1=3).
    # So W1 = [1, 0, 0, 1].
    
    # Let's define the 7 vectors
    U1_vecs = []
    V1_vecs = []
    W1_vecs = []
    
    # 1. (a+d)(e+h)
    U1_vecs.append([1, 0, 0, 1]) # a+d
    V1_vecs.append([1, 0, 0, 1]) # e+h
    W1_vecs.append([1, 0, 0, 1]) # C11 + C22
    
    # 2. (c+d)e
    U1_vecs.append([0, 0, 1, 1]) # c+d
    V1_vecs.append([1, 0, 0, 0]) # e
    W1_vecs.append([0, 0, 1, 0]) # C21
    
    # 3. a(f-h)
    U1_vecs.append([1, 0, 0, 0]) # a
    V1_vecs.append([0, 1, 0, -1]) # f-h
    W1_vecs.append([0, 1, 0, 0]) # C12
    
    # 4. d(g-e)
    U1_vecs.append([0, 0, 0, 1]) # d
    V1_vecs.append([0, -1, 1, 0]) # g-e
    W1_vecs.append([0, 0, 1, 0]) # C21
    
    # 5. (a+b)h
    U1_vecs.append([1, 1, 0, 0]) # a+b
    V1_vecs.append([0, 0, 0, 1]) # h
    W1_vecs.append([0, 1, 0, 0]) # C12
    
    # 6. (c-a)(e+f)
    U1_vecs.append([-1, 0, 1, 0]) # c-a
    V1_vecs.append([1, 1, 0, 0]) # e+f
    W1_vecs.append([0, 0, 0, 1]) # C22
    
    # 7. (b-d)(g+h)
    U1_vecs.append([0, 1, 0, -1]) # b-d
    V1_vecs.append([0, 0, 1, 1]) # g+h
    W1_vecs.append([1, 0, 0, 0]) # C11
    
    # Convert to matrices
    U1 = np.array(U1_vecs).T # 4x7
    V1 = np.array(V1_vecs).T # 4x7
    W1 = np.array(W1_vecs).T # 4x7
    
    # 2. Construct M_{1,1,2} (Schoolbook, rank 2)
    # n2=1, m2=1, p2=2
    n2, m2, p2 = 1, 1, 2
    r2 = 2
    U2 = [[0]*r2 for _ in range(n2*m2)]
    V2 = [[0]*r2 for _ in range(m2*p2)]
    W2 = [[0]*r2 for _ in range(p2*n2)]
    
    # Term 1: A[0,0] B[0,0] -> C[0,0]
    U2[0][0] = 1
    V2[0][0] = 1
    W2[0][0] = 1
    
    # Term 2: A[0,0] B[0,1] -> C[0,1]
    U2[0][1] = 1
    V2[1][1] = 1
    W2[1][1] = 1
    
    U2 = np.array(U2).T # 1x2
    V2 = np.array(V2).T # 2x2
    W2 = np.array(W2).T # 2x2
    
    # 3. Construct M_{1,2,2} (Schoolbook, rank 4)
    # n3=1, m3=2, p3=2
    n3, m3, p3 = 1, 2, 2
    r3 = 4
    U3 = [[0]*r3 for _ in range(n3*m3)]
    V3 = [[0]*r3 for _ in range(m3*p3)]
    W3 = [[0]*r3 for _ in range(p3*n3)]
    
    # Term 1: A[0,0] B[0,0] -> C[0,0]
    U3[0][0] = 1
    V3[0][0] = 1
    W3[0][0] = 1
    
    # Term 2: A[0,1] B[1,0] -> C[0,0]
    U3[1][1] = 1
    V3[2][1] = 1
    W3[0][1] = 1
    
    # Term 3: A[0,0] B[0,1] -> C[0,1]
    U3[0][2] = 1
    V3[1][2] = 1
    W3[1][2] = 1
    
    # Term 4: A[0,1] B[1,1] -> C[0,1]
    U3[1][3] = 1
    V3[3][3] = 1
    W3[1][3] = 1
    
    U3 = np.array(U3).T # 2x4
    V3 = np.array(V3).T # 4x4
    W3 = np.array(W3).T # 2x4
    
    # Combine
    # Total rank R = 7 * 2 * 4 = 56
    R = 56
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Mapping indices
    # i = i1 * n2 + i2? No.
    # The tensor product structure:
    # T(i, j, k) = T1(i1, j1, k1) * T2(i2, j2, k2) * T3(i3, j3, k3)
    # i = i1 * (n2*n3) + i2 * n3 + i3
    # j = j1 * (m2*m3) + j2 * m3 + j3
    # k = k1 * (p2*p3) + k2 * p3 + k3
    
    # n1=2, n2=1, n3=1. n=2.
    # i1 in 0..1. i2=0. i3=0.
    # i = i1 * 1 + 0 + 0 = i1.
    
    # m1=2, m2=1, m3=2. m=4.
    # j1 in 0..1. j2 in 0..0. j3 in 0..1.
    # j = j1 * (1*2) + j2 * 2 + j3 = 2*j1 + j3.
    
    # p1=2, p2=2, p3=2. p=8.
    # k1 in 0..1. k2 in 0..1. k3 in 0..1.
    # k = k1 * (2*2) + k2 * 2 + k3 = 4*k1 + 2*k2 + k3.
    
    # Wait, the W index in the task is k*n + i.
    # My W1, W2, W3 are defined with local k, i.
    # Local k for W1 is k1. Local i for W1 is i1.
    # Local k for W2 is k2. Local i for W2 is i2.
    # Local k for W3 is k3. Local i for W3 is i3.
    
    # We need to map the local W indices to the global W index.
    # Global W index = k_global * n + i_global.
    # k_global = 4*k1 + 2*k2 + k3.
    # i_global = i1. (since n2=n3=1)
    
    # Let's iterate over the rank factors
    for r1 in range(7):
        for r2 in range(2):
            for r3 in range(4):
                col = r1 * 8 + r2 * 4 + r3
                # U
                # U_global[i, col] = U1[i1, r1] * U2[i2, r2] * U3[i3, r3]
                # i1 = i. i2=0. i3=0.
                # U1[i, r1] * U2[0, r2] * U3[0, r3]
                # Note: U3 has 2 rows (i3=0,1). But n3=1, so i3 must be 0?
                # Wait, n3=1. So i3 is always 0.
                # But U3 is for M_{1,2,2}. n3=1, m3=2.
                # U3 index is i3*m3 + j3? No, U3 index is i3*m3 + j3?
                # In M_{n,m,p}, U index is i*m + j.
                # For M_{1,2,2}, U index is i3*2 + j3.
                # But i3 is fixed to 0?
                # No, the tensor product combines the "i" dimensions.
                # The "i" dimension of the total tensor is the product of "i" dimensions of factors.
                # n_total = n1 * n2 * n3 = 2 * 1 * 1 = 2.
                # So i_total = i1 * (n2*n3) + i2 * n3 + i3.
                # i1 in 0..1, i2=0, i3=0.
                # So i_total = i1.
                # However, U3 depends on j3 as well?
                # No, U3 is a matrix of size (n3*m3) x r3.
                # The index of U3 is (i3, j3).
                # In the tensor product, the U index is (i1, i2, i3) and (j1, j2, j3)?
                # No. The U tensor corresponds to the A matrix.
                # A has indices (i, j).
                # In the product, A_total[i, j] = A1[i1, j1] * A2[i2, j2] * A3[i3, j3].
                # So U_total[i_total, j_total, col] = U1[i1, j1, r1] * U2[i2, j2, r2] * U3[i3, j3, r3].
                # But our U matrices are flattened.
                # U1 index: i1*m1 + j1.
                # U2 index: i2*m2 + j2.
                # U3 index: i3*m3 + j3.
                
                # So for a given col, we need to fill U_total[i_total, col].
                # i_total = i1 * (n2*n3) + i2 * n3 + i3.
                # j_total = j1 * (m2*m3) + j2 * m3 + j3.
                
                # We iterate over all i1, j1, i2, j2, i3, j3.
                for i1 in range(n1):
                    for j1 in range(m1):
                        for i2 in range(n2):
                            for j2 in range(m2):
                                for i3 in range(n3):
                                    for j3 in range(m3):
                                        i_tot = i1 * (n2 * n3) + i2 * n3 + i3
                                        j_tot = j1 * (m2 * m3) + j2 * m3 + j3
                                        u_idx = i_tot * m + j_tot
                                        
                                        val = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2] * U3[i3*m3+j3, r3]
                                        U[u_idx, col] += val
                                        
                                        # V
                                        # V index: j_total * p_total + k_total? No.
                                        # V corresponds to B. B indices (j, k).
                                        # j_total is same as above.
                                        # k_total = k1 * (p2*p3) + k2 * p3 + k3.
                                        # V1 index: j1*p1 + k1.
                                        # V2 index: j2*p2 + k2.
                                        # V3 index: j3*p3 + k3.
                                        
                                        for k1 in range(p1):
                                            for k2 in range(p2):
                                                for k3 in range(p3):
                                                    k_tot = k1 * (p2 * p3) + k2 * p3 + k3
                                                    v_idx = j_tot * p + k_tot
                                                    
                                                    val_v = V1[j1*p1+k1, r1] * V2[j2*p2+k2, r2] * V3[j3*p3+k3, r3]
                                                    V[v_idx, col] += val_v
                                                    
                                                    # W
                                                    # W index: k_tot * n + i_tot.
                                                    # W1 index: k1*n1 + i1.
                                                    # W2 index: k2*n2 + i2.
                                                    # W3 index: k3*n3 + i3.
                                                    
                                                    w_idx = k_tot * n + i_tot
                                                    val_w = W1[k1*n1+i1, r1] * W2[k2*n2+i2, r2] * W3[k3*n3+i3, r3]
                                                    W[w_idx, col] += val_w
                                                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
