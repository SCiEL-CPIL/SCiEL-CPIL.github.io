import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 135.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    def check_time():
        return time.time() - t0 < DEADLINE

    def validate(U, V, W, r):
        """Vectorized validation using NumPy."""
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                u_row = Ua[i*m + j, :]
                for k in range(p):
                    v_row = Va[j*p + k, :]
                    w_row = Wa[k*n + i, :]
                    tot = np.dot(u_row, v_row * w_row)
                    if tot != 1:
                        return False
        return True

    # Strassen 2x2x2 Decomposition
    # n=2, m=2, p=2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices:
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    def get_strassen_2x2x2():
        # Rank 7
        r = 7
        U = np.zeros((4, r), dtype=np.int64)
        V = np.zeros((4, r), dtype=np.int64)
        W = np.zeros((4, r), dtype=np.int64)
        
        # Strassen terms for 2x2:
        # M1 = (A11 + A22) (B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11) (B11 + B12)
        # M7 = (A12 - A22) (B21 + B22)
        
        # U corresponds to A (2x2)
        # V corresponds to B (2x2)
        # W corresponds to C (2x2)
        
        # A indices: 0:11, 1:12, 2:21, 3:22
        # B indices: 0:11, 1:12, 2:21, 3:22
        # C indices: 0:11, 1:12, 2:21, 3:22
        
        # Term 1: (A11+A22)(B11+B22) -> C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = -1; W[2, 0] = -1; W[3, 0] = 1
        
        # Term 2: (A21+A22)B11 -> C11 -= M2, C21 += M2, C22 += M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[0, 1] = -1; W[2, 1] = 1; W[3, 1] = 1
        
        # Term 3: A11(B12-B22) -> C12 += M3, C22 -= M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = -1
        
        # Term 4: A22(B21-B11) -> C21 += M4, C11 -= M4, C22 -= M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = -1; W[3, 3] = -1
        
        # Term 5: (A11+A12)B22 -> C11 += M5, C22 += M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[3, 4] = 1
        
        # Term 6: (A21-A11)(B11+B12) -> C12 += M6, C21 += M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1, 5] = 1; W[2, 5] = 1
        
        # Term 7: (A12-A22)(B21+B22) -> C12 -= M7, C21 -= M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = -1; W[2, 6] = -1
        
        return U, V, W

    def kron_factors(U1, V1, W1, U2, V2, W2):
        """
        Compute Kronecker product of two tensor decompositions.
        T1: n1 x m1 x p1, T2: n2 x m2 x p2.
        Result: (n1*n2) x (m1*m2) x (p1*p2).
        U1 shape: (n1*m1, r1), V1 shape: (m1*p1, r1), W1 shape: (p1*n1, r1)
        U2 shape: (n2*m2, r2), V2 shape: (m2*p2, r2), W2 shape: (p2*n2, r2)
        """
        n1, m1, p1 = U1.shape[0] // U1.shape[0], 0, 0 # Need dimensions
        # Infer dimensions from shapes
        # U1: (n1*m1, r1)
        # We need n1, m1, p1 to be passed or inferred.
        # Let's assume standard ordering:
        # Index in U1: i1*m1 + j1
        # Index in V1: j1*p1 + k1
        # Index in W1: k1*n1 + i1
        
        # This function needs n1, m1, p1, n2, m2, p2
        pass

    def kron_factors_dims(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 * r2
        
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        U = np.zeros((n*m, r), dtype=np.int64)
        V = np.zeros((m*p, r), dtype=np.int64)
        W = np.zeros((p*n, r), dtype=np.int64)
        
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1 * r2 + c2
                
                # U
                # i = i1*n2 + i2, j = j1*m2 + j2
                # U1 index: i1*m1 + j1
                # U2 index: i2*m2 + j2
                # U index: i*m + j = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
                
                u1_col = U1[:, c1] # (n1*m1,)
                u2_col = U2[:, c2] # (n2*m2,)
                
                # Reshape u1 to (n1, m1), u2 to (n2, m2)
                u1_mat = u1_col.reshape(n1, m1)
                u2_mat = u2_col.reshape(n2, m2)
                
                # Kronecker product of matrices
                u_prod = np.kron(u1_mat, u2_mat) # (n1*n2, m1*m2)
                
                # Flatten to (n*m,)
                # Row index in u_prod: i1*n2 + i2?
                # np.kron(A, B) where A is (n1, m1) and B is (n2, m2)
                # Result is (n1*n2, m1*m2).
                # Element (i, j) is A[i1, j1] * B[i2, j2] where i = i1*n2 + i2, j = j1*m2 + j2.
                # This matches our index definition.
                
                U[:, c] = u_prod.flatten()
                
                # V
                # j = j1*m2 + j2, k = k1*p2 + k2
                # V1 index: j1*p1 + k1
                # V2 index: j2*p2 + k2
                # V index: j*p + k = (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
                
                v1_col = V1[:, c1] # (m1*p1,)
                v2_col = V2[:, c2] # (m2*p2,)
                
                v1_mat = v1_col.reshape(m1, p1)
                v2_mat = v2_col.reshape(m2, p2)
                
                v_prod = np.kron(v1_mat, v2_mat) # (m1*m2, p1*p2)
                
                V[:, c] = v_prod.flatten()
                
                # W
                # k = k1*p2 + k2, i = i1*n2 + i2
                # W1 index: k1*n1 + i1
                # W2 index: k2*n2 + i2
                # W index: k*n + i = (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
                
                w1_col = W1[:, c1] # (p1*n1,)
                w2_col = W2[:, c2] # (p2*n2,)
                
                w1_mat = w1_col.reshape(p1, n1)
                w2_mat = w2_col.reshape(p2, n2)
                
                w_prod = np.kron(w1_mat, w2_mat) # (p1*p2, n1*n2)
                
                W[:, c] = w_prod.flatten()
                
        return U, V, W

    def get_trivial_1x1x1():
        # n=1, m=1, p=1, rank 1
        U = np.ones((1, 1), dtype=np.int64)
        V = np.ones((1, 1), dtype=np.int64)
        W = np.ones((1, 1), dtype=np.int64)
        return U, V, W

    def get_4x4x1():
        # n=4, m=4, p=1, rank 16
        n, m, p = 4, 4, 1
        r = n*m
        U = np.zeros((n*m, r), dtype=np.int64)
        V = np.zeros((m*p, r), dtype=np.int64)
        W = np.zeros((p*n, r), dtype=np.int64)
        
        for i in range(n):
            for j in range(m):
                c = i*m + j
                U[i*m+j, c] = 1
                V[j*p+0, c] = 1
                W[0*n+i, c] = 1
        return U, V, W

    def concat_decompositions(U1, V1, W1, U2, V2, W2):
        """
        Concatenate two decompositions that cover disjoint k-ranges.
        Assumes U1, V1, W1 cover k in [0, p1) and U2, V2, W2 cover k in [p1, p1+p2).
        The resulting decomposition covers [0, p1+p2).
        """
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        return U, V, W

    # Step 1: Get 2x2x2 Strassen
    U2, V2, W2 = get_strassen_2x2x2()
    
    # Step 2: Get 4x4x4 via Kronecker
    U4, V4, W4 = kron_factors_dims(U2, V2, W2, 2, 2, 2, U2, V2, W2, 2, 2, 2)
    
    # Step 3: Get 4x4x1
    U41, V41, W41 = get_4x4x1()
    
    # Step 4: Get 4x4x5 by concatenating 4x4x4 and 4x4x1
    # Note: 4x4x4 covers k=0..3. 4x4x1 covers k=0..0.
    # We need to shift the k-index of the second part.
    # However, the factors are just matrices. The "shift" is implicit in how we use them?
    # No, the factors V and W have indices depending on k.
    # V shape (m*p, r). For 4x4x4, p=4. For 4x4x1, p=1.
    # When concatenating to form 4x4x5 (p=5), we need to place the V and W entries correctly.
    # V4 has non-zeros for k=0..3. V41 has non-zeros for k=0.
    # In the 4x4x5 tensor, the second part corresponds to k=4.
    # So we need to shift V41 and W41 to k=4.
    
    def shift_k(V, W, p_new, k_offset, n, m, p_old):
        """
        Shift the k-index of V and W by k_offset.
        V shape (m*p_old, r). New shape (m*p_new, r).
        W shape (p_old*n, r). New shape (p_new*n, r).
        """
        r = V.shape[1]
        V_new = np.zeros((m*p_new, r), dtype=np.int64)
        W_new = np.zeros((p_new*n, r), dtype=np.int64)
        
        # V: index j*p + k. New index j*p_new + (k + k_offset).
        for j in range(m):
            for k in range(p_old):
                for c in range(r):
                    val = V[j*p_old + k, c]
                    if val != 0:
                        new_k = k + k_offset
                        V_new[j*p_new + new_k, c] = val
        
        # W: index k*n + i. New index (k + k_offset)*n + i.
        for k in range(p_old):
            for i in range(n):
                for c in range(r):
                    val = W[k*n + i, c]
                    if val != 0:
                        new_k = k + k_offset
                        W_new[new_k*n + i, c] = val
                        
        return V_new, W_new

    # 4x4x4 covers k=0..3.
    # 4x4x1 covers k=0. We want it to cover k=4.
    V41_shifted, W41_shifted = shift_k(V41, W41, 5, 4, 4, 4, 1)
    
    # U is not shifted because it doesn't depend on k.
    U45 = np.hstack([U4, U41])
    V45 = np.hstack([V4, V41_shifted])
    W45 = np.hstack([W4, W41_shifted])
    
    # Step 5: Get 4x4x9 by concatenating 4x4x4 and 4x4x5
    # 4x4x4 covers k=0..3.
    # 4x4x5 covers k=0..4. We want it to cover k=4..8.
    # So we shift 4x4x5 by 4.
    
    V45_shifted, W45_shifted = shift_k(V45, W45, 9, 4, 4, 4, 5)
    
    U49 = np.hstack([U4, U45])
    V49 = np.hstack([V4, V45_shifted])
    W49 = np.hstack([W4, W45_shifted])
    
    r_final = U49.shape[1]
    
    # Validate
    if not validate(U49.tolist(), V49.tolist(), W49.tolist(), r_final):
        # Fallback to schoolbook
        r_fb = n * m * p
        U_fb = np.zeros((nm, r_fb), dtype=np.int64)
        V_fb = np.zeros((mp, r_fb), dtype=np.int64)
        W_fb = np.zeros((pn, r_fb), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, c] = 1
                    V_fb[j*p+k, c] = 1
                    W_fb[k*n+i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}
