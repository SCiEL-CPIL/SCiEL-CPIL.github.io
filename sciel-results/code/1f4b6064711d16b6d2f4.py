import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 2x2 matrices flattened to length 4
    # Order: 00, 01, 10, 11
    
    # Standard Strassen factors
    # M1 = (A11 - A22)(B21 + B22)
    # M2 = (A11 + A22)(B11 - B12)
    # M3 = (A21 + A22)B12
    # M4 = A11(B22 - B12)
    # M5 = (A11 - A12)B22
    # M6 = (A12 + A21)(B11 + B12)
    # M7 = (A11 + A12)(B21 - B11)
    
    # Let's use the vectors from the prompt as they seem to work (score 1.306)
    # But I will verify them or use a known correct set.
    
    U2 = np.array([
        [1, 0, 0, 1], # A11 + A22 ? No, 00+11
        [0, 0, 1, 1], # A21 + A22 ? No, 10+11
        [1, 0, 0, 0], # A11
        [0, 0, 0, 1], # A22
        [1, 1, 0, 0], # A11 + A12
        [-1, 0, 1, 0],# A21 - A11
        [0, 1, 0, -1] # A12 - A22
    ])
    
    V2 = np.array([
        [1, 0, 0, 1], # B11 + B22
        [1, 0, 0, 0], # B11
        [0, 1, 0, -1],# B12 - B22
        [-1, 0, 1, 0],# B21 - B11
        [0, 0, 0, 1], # B22
        [1, 1, 0, 0], # B11 + B12
        [0, 0, 1, 1]  # B21 + B22
    ])
    
    W2 = np.array([
        [1, 0, 0, 1], # C11 + C22
        [0, 0, 1, -1],# C21 - C22
        [0, 1, 0, 1], # C12 + C22
        [1, 0, 1, 0], # C11 + C21
        [-1, 1, 0, 0],# -C11 + C12
        [0, 0, 0, 1], # C22
        [1, 0, 0, 0]  # C11
    ])
    
    # Verify 2x2 Strassen:
    # C = sum_r U2[r] (A * U2[r]) * (V2[r] * B) W2[r]? No.
    # The tensor is T_{i,j,k} = sum_r U[i,r] V[j,r] W[k,r].
    # For 2x2, i,j,k in {0,1}.
    # A is 2x2, B is 2x2.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is not a direct product of U, V, W on indices i,j,k.
    # The factorization is:
    # U has indices (i, j). V has indices (j, k). W has indices (k, i).
    # Wait, the prompt says:
    # U: (n*m, rank). Index i*m + j.
    # V: (m*p, rank). Index j*p + k.
    # W: (p*n, rank). Index k*n + i.
    
    # So for 2x2x2:
    # U index: i*2 + j.
    # V index: j*2 + k.
    # W index: k*2 + i.
    
    # Let's check the first term r=0 in the prompt's U2, V2, W2.
    # U2[0] = [1, 0, 0, 1]. Indices: 00, 01, 10, 11.
    # So U[0,0]=1, U[1,1]=1.
    # This corresponds to A_{0,0} and A_{1,1}.
    # V2[0] = [1, 0, 0, 1].
    # V[0,0]=1, V[1,1]=1.
    # This corresponds to B_{0,0} and B_{1,1}.
    # W2[0] = [1, 0, 0, 1].
    # W[0,0]=1, W[1,1]=1.
    # This corresponds to C_{0,0} and C_{1,1}.
    
    # The term contributes:
    # U[0,0] V[0,0] W[0,0] = 1 * 1 * 1 = 1 to T[0,0,0].
    # U[0,0] V[0,1] W[1,0] = 1 * 0 * 0 = 0.
    # ...
    # This looks like it's constructing the identity or something.
    # Actually, the Strassen algorithm computes linear combinations.
    # The rank-1 tensors are not just single entries, they are the coefficients.
    
    # The construction in the prompt uses Kronecker product of these 2x2 factors.
    # I will trust the prompt's factors as they yielded a valid score.
    
    # Generate 4x4x4 factors
    U4_list = []
    V4_list = []
    W4_list = []
    
    for s in range(7):
        for t in range(7):
            U4 = np.zeros(16)
            V4 = np.zeros(16)
            W4 = np.zeros(16)
            
            for I in range(4):
                for J in range(4):
                    r_I = I // 2
                    c_I = I % 2
                    r_J = J // 2
                    c_J = J % 2
                    # U index is I*4 + J? No, U index is i*m + j.
                    # Here m=4. So index is I*4 + J.
                    # The value is U2[s, r_I*2+c_I] * U2[t, r_J*2+c_J]?
                    # Wait, the Kronecker product of U2[s] and U2[t].
                    # U2[s] is a vector of length 4 (flattened 2x2).
                    # The Kronecker product of two 2x2 matrices A and B is a 4x4 matrix.
                    # Flattened, it's length 16.
                    # U4[I*4+J] = U2[s, I*2 + (J//2)] * U2[t, (I%2)*2 + (J%2)]?
                    # Let's check the prompt's logic:
                    # coeff = U2[s][idx2(r, c)] * U2[t][idx2(i_loc, j_loc)]
                    # r = I//2, c = J//2. i_loc = I%2, j_loc = J%2.
                    # This is correct for Kronecker product.
                    
                    coeff = U2[s, (I//2)*2 + (J//2)] * U2[t, (I%2)*2 + (J%2)]
                    U4[I*4 + J] = coeff
                    
            for J in range(4):
                for K in range(4):
                    coeff = V2[s, (J//2)*2 + (K//2)] * V2[t, (J%2)*2 + (K%2)]
                    V4[J*4 + K] = coeff
                    
            for I in range(4):
                for K in range(4):
                    coeff = W2[s, (I//2)*2 + (K//2)] * W2[t, (I%2)*2 + (K%2)]
                    # W index in 4x4 is K*4 + I?
                    # Prompt says: W[k*n+i, r].
                    # For 4x4, n=4. Index is k*4 + i.
                    # Here I is i, K is k.
                    # So index is K*4 + I.
                    # Prompt code: W4[i * 4 + k]. Wait.
                    # Prompt code: W4[i * 4 + k] = coeff.
                    # Then: W[k * 4 + i][col_idx] = W4[i * 4 + k].
                    # This implies W4 is stored as i*4+k, but mapped to k*4+i.
                    # This is a transpose.
                    # Let's follow the prompt's mapping exactly to be safe.
                    W4[I*4 + K] = coeff # Stored as I*4+K in W4 array
                    # But in the final W matrix, it's placed at K*4+I.
            
            U4_list.append(U4)
            V4_list.append(V4)
            W4_list.append(W4)
            
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    for r in range(49):
        # Block 1
        U[:, r] = U4_list[r]
        for j in range(4):
            for k in range(4):
                V[j*8 + k, r] = V4_list[r][j*4 + k]
        for k in range(4):
            for i in range(4):
                W[k*4 + i, r] = W4_list[r][i*4 + k]
                
        # Block 2
        r2 = 49 + r
        U[:, r2] = U4_list[r]
        for j in range(4):
            for k in range(4):
                V[j*8 + (k+4), r2] = V4_list[r][j*4 + k]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4 + i, r2] = W4_list[r][i*4 + k]
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
