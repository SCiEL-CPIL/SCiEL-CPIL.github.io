import numpy as np

def construct():
    n, m, p = 2, 5, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_tensor(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_term(i, j, k, coeff=1):
        """
        Adds a rank-1 tensor term: A[i,j] * B[j,k] * C[i,k]
        u_vec has 1 at i*m+j
        v_vec has 1 at j*p+k
        w_vec has 1 at k*n+i
        """
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i * m + j] = coeff
        v[j * p + k] = 1
        w[k * n + i] = 1
        add_tensor(u, v, w)
        
    def add_strassen_2x2x2(base_i, base_j, base_k):
        """
        Adds the 7 terms for 2x2x2 Strassen decomposition.
        A indices: i in {base_i, base_i+1}, j in {base_j, base_j+1}
        B indices: j in {base_j, base_j+1}, k in {base_k, base_k+1}
        C indices: i in {base_i, base_i+1}, k in {base_k, base_k+1}
        """
        # M1 = (a00+a11)(b00+b11)
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[base_i * m + base_j] = 1
        u[(base_i+1) * m + (base_j+1)] = 1
        v[base_j * p + base_k] = 1
        v[(base_j+1) * p + (base_k+1)] = 1
        # C00 += M1, C11 += M1
        w[base_k * n + base_i] = 1
        w[(base_k+1) * n + (base_i+1)] = 1
        add_tensor(u, v, w)
        
        # M2 = (a10+a11)b00
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[(base_i+1) * m + base_j] = 1
        u[(base_i+1) * m + (base_j+1)] = 1
        v[base_j * p + base_k] = 1
        # C10 += M2, C11 -= M2
        w[base_k * n + (base_i+1)] = 1
        w[(base_k+1) * n + (base_i+1)] = -1
        add_tensor(u, v, w)
        
        # M3 = a00(b10-b11)
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[base_i * m + base_j] = 1
        v[(base_j+1) * p + base_k] = 1
        v[(base_j+1) * p + (base_k+1)] = -1
        # C01 += M3, C11 += M3
        w[(base_k+1) * n + base_i] = 1
        w[(base_k+1) * n + (base_i+1)] = 1
        add_tensor(u, v, w)
        
        # M4 = a11(b01+b11)
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[(base_i+1) * m + (base_j+1)] = 1
        v[base_j * p + (base_k+1)] = 1
        v[(base_j+1) * p + (base_k+1)] = 1
        # C00 += M4, C01 += M4
        w[base_k * n + base_i] = 1
        w[(base_k+1) * n + base_i] = 1
        add_tensor(u, v, w)
        
        # M5 = (a00-a10)b11
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[base_i * m + base_j] = 1
        u[(base_i+1) * m + base_j] = -1
        v[(base_j+1) * p + (base_k+1)] = 1
        # C00 -= M5, C10 += M5
        w[base_k * n + base_i] = -1
        w[base_k * n + (base_i+1)] = 1
        add_tensor(u, v, w)
        
        # M6 = (a01+a10)(b00-b10)
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[base_i * m + (base_j+1)] = 1
        u[(base_i+1) * m + base_j] = 1
        v[base_j * p + base_k] = 1
        v[(base_j+1) * p + base_k] = -1
        # C11 += M6
        w[(base_k+1) * n + (base_i+1)] = 1
        add_tensor(u, v, w)
        
        # M7 = (a01-a11)(b10+b11)
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        u[base_i * m + (base_j+1)] = 1
        u[(base_i+1) * m + (base_j+1)] = -1
        v[(base_j+1) * p + base_k] = 1
        v[(base_j+1) * p + (base_k+1)] = 1
        # C01 -= M7, C11 -= M7
        w[(base_k+1) * n + base_i] = -1
        w[(base_k+1) * n + (base_i+1)] = -1
        add_tensor(u, v, w)

    # Block 1: j=0, 2x1x6
    for k in range(6):
        add_term(0, 0, k)
        add_term(1, 0, k)
        
    # Block 2: j=1,2, 2x2x6
    # Split k into 0 and 1,2
    # k=0: 2x2x1
    add_term(0, 1, 0)
    add_term(0, 2, 0)
    add_term(1, 1, 0)
    add_term(1, 2, 0)
    
    # k=1,2: 2x2x2 Strassen
    add_strassen_2x2x2(0, 1, 1)
    
    # Block 3: j=3,4, 2x2x6
    # Split k into 0 and 1,2
    # k=0: 2x2x1
    add_term(0, 3, 0)
    add_term(0, 4, 0)
    add_term(1, 3, 0)
    add_term(1, 4, 0)
    
    # k=1,2: 2x2x2 Strassen
    add_strassen_2x2x2(0, 3, 1)
    
    # Wait, I only covered k=0,1,2. What about k=3,4,5?
    # The split was p=6 into 1+2+2+1? No.
    # My plan was:
    # Block 2 (j=1,2) covers k=0..5.
    # I split k into 0 (size 1) and 1,2 (size 2). That's only 3.
    # I need to cover k=3,4,5 as well.
    # So for Block 2, I should have two Strassen blocks?
    # R(2,2,6) = 2 * R(2,2,3).
    # R(2,2,3) = R(2,2,1) + R(2,2,2) = 4 + 7 = 11.
    # So for k=0,1,2, I used 11.
    # For k=3,4,5, I need another 11.
    
    # Let's add the second half for Block 2
    # k=3: 2x2x1
    add_term(0, 1, 3)
    add_term(0, 2, 3)
    add_term(1, 1, 3)
    add_term(1, 2, 3)
    
    # k=4,5: 2x2x2 Strassen
    add_strassen_2x2x2(0, 1, 4)
    
    # And for Block 3
    # k=3: 2x2x1
    add_term(0, 3, 3)
    add_term(0, 4, 3)
    add_term(1, 3, 3)
    add_term(1, 4, 3)
    
    # k=4,5: 2x2x2 Strassen
    add_strassen_2x2x2(0, 3, 4)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
