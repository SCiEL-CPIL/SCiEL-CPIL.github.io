import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Global sizes
    U_size = n * m  # 9
    V_size = m * p  # 15
    W_size = p * n  # 15
    
    # We will build lists of (u_idx, v_idx, w_idx, coeff)
    # But the output format is matrices U, V, W of shape (size, rank)
    # So we can just append columns to U, V, W.
    
    U = []
    V = []
    W = []
    
    def add_tensor(u_vec, v_vec, w_vec):
        """
        u_vec: list of length n*m with 0/1/-1
        v_vec: list of length m*p with 0/1/-1
        w_vec: list of length p*n with 0/1/-1
        """
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    def zeros(size):
        return [0] * size
        
    # Helper to create a vector with a single 1 at index
    def unit_vec(size, idx, val=1):
        v = zeros(size)
        v[idx] = val
        return v
        
    # Helper to create a vector with 1s at a range of indices
    def range_vec(size, start, end, val=1):
        v = zeros(size)
        for i in range(start, end):
            v[i] = val
        return v

    # Block 1: A[0:2, 0:2] * B[0:2, :] -> C[0:2, :]
    # A indices: i in 0,1; j in 0,1. U idx = i*3 + j
    # B indices: j in 0,1; k in 0..4. V idx = j*5 + k
    # C indices: k in 0..4; i in 0,1. W idx = k*3 + i
    
    # Sub-block 1a: 2x2x2 (cols 0,1)
    # Strassen
    # M1 = (A00+A11)(B00+B11)
    u = zeros(9); u[0] = 1; u[3] = 1 # A00, A11
    v = zeros(15); v[0] = 1; v[6] = 1 # B00, B11 (j=0,k=0 -> 0; j=1,k=1 -> 1*5+1=6)
    w = zeros(15); w[0] = 1; w[5] = 1 # C00, C11 (k=0,i=0 -> 0; k=1,i=1 -> 1*3+1=4? No. W idx = k*3+i. C[0,0] -> 0*3+0=0. C[1,1] -> 1*3+1=4.)
    # Wait, W idx = k*n + i.
    # C[0,0] -> k=0, i=0 -> 0.
    # C[1,1] -> k=1, i=1 -> 1*3+1=4.
    # My previous W idx calculation: w[0]=1, w[4]=1.
    w = zeros(15); w[0] = 1; w[4] = 1
    add_tensor(u, v, w)
    
    # M2 = (A10+A11)B00
    u = zeros(9); u[2] = 1; u[3] = 1 # A10 (1*3+0=3? No. i=1,j=0 -> 3. i=1,j=1 -> 4.)
    # A[i,j] -> i*3+j.
    # A10 -> 1*3+0 = 3.
    # A11 -> 1*3+1 = 4.
    u = zeros(9); u[3] = 1; u[4] = 1
    v = zeros(15); v[0] = 1 # B00
    w = zeros(15); w[3] = 1 # C10 (k=0, i=1 -> 0*3+1=1? No. W idx = k*n+i. C[1,0] -> k=0, i=1 -> 1.)
    # C[1,0] is row 1, col 0.
    # W[k*n+i]. k=0, i=1 -> 1.
    w = zeros(15); w[1] = 1
    add_tensor(u, v, w)
    
    # M3 = A00(B10-B11)
    u = zeros(9); u[0] = 1 # A00
    v = zeros(15); v[5] = 1; v[6] = -1 # B10 (j=1,k=0 -> 5), B11 (j=1,k=1 -> 6)
    w = zeros(15); w[2] = 1 # C01 (k=1, i=0 -> 1*3+0=3? No. W idx = k*n+i. C[0,1] -> k=1, i=0 -> 3.)
    w = zeros(15); w[3] = 1
    add_tensor(u, v, w)
    
    # M4 = A11(B01-B00)
    u = zeros(9); u[4] = 1 # A11
    v = zeros(15); v[1] = 1; v[0] = -1 # B01 (j=0,k=1 -> 1), B00 (j=0,k=0 -> 0)
    w = zeros(15); w[4] = 1 # C11 (k=1, i=1 -> 1*3+1=4)
    add_tensor(u, v, w)
    
    # M5 = (A00+A01)B11
    u = zeros(9); u[0] = 1; u[1] = 1 # A00, A01
    v = zeros(15); v[6] = 1 # B11
    w = zeros(15); w[3] = 1 # C01
    add_tensor(u, v, w)
    
    # M6 = (A10-A00)(B00+B10)
    u = zeros(9); u[3] = 1; u[0] = -1 # A10, A00
    v = zeros(15); v[0] = 1; v[5] = 1 # B00, B10
    w = zeros(15); w[1] = 1 # C10
    add_tensor(u, v, w)
    
    # M7 = (A01-A11)(B11+B01)
    u = zeros(9); u[1] = 1; u[4] = -1 # A01, A11
    v = zeros(15); v[6] = 1; v[1] = 1 # B11, B01
    w = zeros(15); w[4] = 1 # C11
    add_tensor(u, v, w)
    
    # Sub-block 1b: 2x2x2 (cols 2,3)
    # B cols 2,3. V idx: j*5 + k.
    # B02 (j=0,k=2) -> 2. B03 (j=0,k=3) -> 3.
    # B12 (j=1,k=2) -> 7. B13 (j=1,k=3) -> 8.
    # C cols 2,3. W idx: k*3 + i.
    # C02 (k=2,i=0) -> 6. C03 (k=3,i=0) -> 9.
    # C12 (k=2,i=1) -> 7. C13 (k=3,i=1) -> 10.
    
    # M1
    u = zeros(9); u[0] = 1; u[4] = 1
    v = zeros(15); v[2] = 1; v[8] = 1
    w = zeros(15); w[6] = 1; w[10] = 1
    add_tensor(u, v, w)
    
    # M2
    u = zeros(9); u[3] = 1; u[4] = 1
    v = zeros(15); v[2] = 1
    w = zeros(15); w[7] = 1
    add_tensor(u, v, w)
    
    # M3
    u = zeros(9); u[0] = 1
    v = zeros(15); v[7] = 1; v[8] = -1
    w = zeros(15); w[9] = 1
    add_tensor(u, v, w)
    
    # M4
    u = zeros(9); u[4] = 1
    v = zeros(15); v[3] = 1; v[2] = -1
    w = zeros(15); w[10] = 1
    add_tensor(u, v, w)
    
    # M5
    u = zeros(9); u[0] = 1; u[1] = 1
    v = zeros(15); v[8] = 1
    w = zeros(15); w[9] = 1
    add_tensor(u, v, w)
    
    # M6
    u = zeros(9); u[3] = 1; u[0] = -1
    v = zeros(15); v[2] = 1; v[7] = 1
    w = zeros(15); w[7] = 1
    add_tensor(u, v, w)
    
    # M7
    u = zeros(9); u[1] = 1; u[4] = -1
    v = zeros(15); v[8] = 1; v[3] = 1
    w = zeros(15); w[10] = 1
    add_tensor(u, v, w)
    
    # Sub-block 1c: 2x2x1 (col 4)
    # B col 4. V idx: B04 (j=0,k=4) -> 4. B14 (j=1,k=4) -> 9.
    # C col 4. W idx: C04 (k=4,i=0) -> 12. C14 (k=4,i=1) -> 13.
    
    # Schoolbook
    # C04 = A00 B04 + A01 B14
    u = zeros(9); u[0] = 1
    v = zeros(15); v[4] = 1
    w = zeros(15); w[12] = 1
    add_tensor(u, v, w)
    
    u = zeros(9); u[1] = 1
    v = zeros(15); v[9] = 1
    w = zeros(15); w[12] = 1
    add_tensor(u, v, w)
    
    # C14 = A10 B04 + A11 B14
    u = zeros(9); u[3] = 1
    v = zeros(15); v[4] = 1
    w = zeros(15); w[13] = 1
    add_tensor(u, v, w)
    
    u = zeros(9); u[4] = 1
    v = zeros(15); v[9] = 1
    w = zeros(15); w[13] = 1
    add_tensor(u, v, w)
    
    # Block 2: A[0:2, 2] * B[2, :] -> C[0:2, :]
    # A indices: i in 0,1; j=2. U idx = i*3+2.
    # B indices: j=2; k in 0..4. V idx = 2*5+k = 10+k.
    # C indices: k in 0..4; i in 0,1. W idx = k*3+i.
    
    # 10 terms
    for i in range(2):
        for k in range(5):
            u = zeros(9); u[i*3+2] = 1
            v = zeros(15); v[10+k] = 1
            w = zeros(15); w[k*3+i] = 1
            add_tensor(u, v, w)
            
    # Block 3: A[2, 0:2] * B[0:2, :] -> C[2, :]
    # A indices: i=2; j in 0,1. U idx = 2*3+j = 6+j.
    # B indices: j in 0,1; k in 0..4. V idx = j*5+k.
    # C indices: k in 0..4; i=2. W idx = k*3+2.
    
    # 10 terms
    for j in range(2):
        for k in range(5):
            u = zeros(9); u[6+j] = 1
            v = zeros(15); v[j*5+k] = 1
            w = zeros(15); w[k*3+2] = 1
            add_tensor(u, v, w)
            
    # Block 4: A[2, 2] * B[2, :] -> C[2, :]
    # A indices: i=2; j=2. U idx = 2*3+2 = 8.
    # B indices: j=2; k in 0..4. V idx = 10+k.
    # C indices: k in 0..4; i=2. W idx = k*3+2.
    
    # 5 terms
    for k in range(5):
        u = zeros(9); u[8] = 1
        v = zeros(15); v[10+k] = 1
        w = zeros(15); w[k*3+2] = 1
        add_tensor(u, v, w)
        
    # Convert lists to matrices
    rank = len(U)
    U_mat = np.array(U).T
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
