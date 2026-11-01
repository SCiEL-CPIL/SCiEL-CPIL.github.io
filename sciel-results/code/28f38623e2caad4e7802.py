import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    # A is n x m (2x2)
    # B is m x p (2x11)
    # C is n x p (2x11)
    
    num_a = n * m # 4
    num_b = m * p # 22
    num_c = n * p # 22
    
    # We will collect the terms
    # Each term is (U_vec, V_vec, W_vec)
    terms = []
    
    # Helper to add a Strassen block for columns k1, k2 (0-indexed)
    def add_strassen(k1, k2):
        # B indices:
        # B[0, k1] is index k1
        # B[0, k2] is index k2
        # B[1, k1] is index p + k1
        # B[1, k2] is index p + k2
        
        b11_idx = k1
        b12_idx = k2
        b21_idx = p + k1
        b22_idx = p + k2
        
        # C indices:
        # C[0, k1] is index k1
        # C[0, k2] is index k2
        # C[1, k1] is index p + k1
        # C[1, k2] is index p + k2
        
        c11_idx = k1
        c12_idx = k2
        c21_idx = p + k1
        c22_idx = p + k2
        
        # A indices:
        # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        
        a11 = 0
        a12 = 1
        a21 = 2
        a22 = 3
        
        # M1 = (A11 + A22)(B11 + B22)
        u = [0]*num_a; v = [0]*num_b; w = [0]*num_c
        u[a11] = 1; u[a22] = 1
        v[b11_idx] = 1; v[b22_idx] = 1
        # C11 += M1, C22 += M1
        w[c11_idx] = 1; w[c22_idx] = 1
        terms.append((u, v, w))
        
        # M2 = (A21 + A22)B11
        u = [0]*num_a; v = [0]*num_b; w = [0]*num_c
        u[a21] = 1; u[a22] = 1
        v[b11_idx] = 1
        # C21 += M2, C22 -= M2
        w[c21_idx] = 1; w[c22_idx] = -1
        terms.append((u, v, w))
        
        # M3 = A11(B12 - B22)
        u = [0]*num_a; v = [0]*num_b; w = [0]*num_c
        u[a11] = 1
        v[b12_idx] = 1; v[b22_idx] = -1
        # C12 += M3, C22 += M3
        w[c12_idx] = 1; w[c22_idx] = 1
        terms.append((u, v, w))
        
        # M4 = A22(B21 - B11)
        u = [0]*num_a; v = [0]*num_b; w = [0]*num_c
        u[a22] = 1
        v[b21_idx] = 1; v[b11_idx] = -1
        # C11 += M4, C21 += M4
        w[c11_idx] = 1; w[c21_idx] = 1
        terms.append((u, v, w))
        
        # M5 = (A11 + A12)B22
        u = [0]*num_a; v = [0]*num_b; w = [0]*num_c
        u[a11] = 1; u[a12] = 1
        v[b22_idx] = 1
        # C12 += M5, C11 -= M5
        w[c12_idx] = 1; w[c11_idx] = -1
        terms.append((u, v, w))
        
        # M6 = (A21 - A11)(B11 + B12)
        u = [0]*num_a; v = [0]*num_b; w = [0]*num_c
        u[a21] = 1; u[a11] = -1
        v[b11_idx] = 1; v[b12_idx] = 1
        # C22 += M6
        w[c22_idx] = 1
        terms.append((u, v, w))
        
        # M7 = (A12 - A22)(B21 + B22)
        u = [0]*num_a; v = [0]*num_b; w = [0]*num_c
        u[a12] = 1; u[a22] = -1
        v[b21_idx] = 1; v[b22_idx] = 1
        # C11 += M7
        w[c11_idx] = 1
        terms.append((u, v, w))

    # Helper to add a Schoolbook block for column k
    def add_schoolbook(k):
        b1_idx = k
        b2_idx = p + k
        c1_idx = k
        c2_idx = p + k
        
        a11 = 0; a12 = 1; a21 = 2; a22 = 3
        
        # C1 = A11 B1 + A12 B2
        # Term 1: A11 B1
        u = [0]*num_a; v = [0]*num_b; w = [0]*num_c
        u[a11] = 1
        v[b1_idx] = 1
        w[c1_idx] = 1
        terms.append((u, v, w))
        
        # Term 2: A12 B2
        u = [0]*num_a; v = [0]*num_b; w = [0]*num_c
        u[a12] = 1
        v[b2_idx] = 1
        w[c1_idx] = 1
        terms.append((u, v, w))
        
        # Term 3: A21 B1
        u = [0]*num_a; v = [0]*num_b; w = [0]*num_c
        u[a21] = 1
        v[b1_idx] = 1
        w[c2_idx] = 1
        terms.append((u, v, w))
        
        # Term 4: A22 B2
        u = [0]*num_a; v = [0]*num_b; w = [0]*num_c
        u[a22] = 1
        v[b2_idx] = 1
        w[c2_idx] = 1
        terms.append((u, v, w))

    # Apply blocks
    # Cols 0,1 -> Strassen
    add_strassen(0, 1)
    # Cols 2,3 -> Strassen
    add_strassen(2, 3)
    # Cols 4,5 -> Strassen
    add_strassen(4, 5)
    # Cols 6,7 -> Strassen
    add_strassen(6, 7)
    # Cols 8,9 -> Strassen
    add_strassen(8, 9)
    # Col 10 -> Schoolbook
    add_schoolbook(10)
    
    rank = len(terms)
    
    U = np.zeros((num_a, rank))
    V = np.zeros((num_b, rank))
    W = np.zeros((num_c, rank))
    
    for r in range(rank):
        u, v, w = terms[r]
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
