import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 110
    
    # We will build the decomposition using a hybrid of Strassen and schoolbook/low-rank tricks.
    # Total rank estimated at 55.
    
    # Helper to create zero matrices
    def make_zeros(rows, cols):
        return [[0.0] * cols for _ in range(rows)]
    
    # We will accumulate factors into lists of columns, then combine.
    # U_cols: list of vectors of length n*m
    # V_cols: list of vectors of length m*p
    # W_cols: list of vectors of length p*n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_factor(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen 2x2x2 factors
    # U indices: 00, 01, 10, 11
    # V indices: 00, 01, 10, 11
    # W indices: 00, 01, 10, 11
    # 
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B10)
    # M7: (A01-A11)(B11+B01)
    
    # We define the factors for a local 2x2x2 block.
    # Local U index (i,j) -> global U index (base_i + i, base_j + j)
    # Local V index (j,k) -> global V index (base_j + j, base_k + k)
    # Local W index (k,i) -> global W index (base_k + k, base_i + i)
    
    def add_strassen_2x2x2(base_i, base_j, base_k):
        # 7 factors
        # 1
        u = [0]*4; v = [0]*4; w = [0]*4
        u[0]=1; u[3]=1
        v[0]=1; v[3]=1
        w[0]=1; w[3]=1
        # Map to global
        gu = [0]*(n*m); gv = [0]*(m*p); gw = [0]*(p*n)
        gu[base_i*m + base_j + 0] = u[0] # (0,0)
        gu[base_i*m + base_j + 2] = u[3] # (1,1)
        gv[base_j*p + base_k + 0] = v[0] # (0,0)
        gv[base_j*p + base_k + 3] = v[3] # (1,1)
        gw[base_k*n + base_i + 0] = w[0] # (0,0)
        gw[base_k*n + base_i + 2] = w[3] # (1,1)
        add_factor(gu, gv, gw)
        
        # 2
        u = [0]*4; v = [0]*4; w = [0]*4
        u[2]=1; u[3]=1
        v[0]=1
        w[0]=1; w[2]=1
        gu = [0]*(n*m); gv = [0]*(m*p); gw = [0]*(p*n)
        gu[base_i*m + base_j + 2] = u[2] # (1,0)
        gu[base_i*m + base_j + 3] = u[3] # (1,1)
        gv[base_j*p + base_k + 0] = v[0] # (0,0)
        gw[base_k*n + base_i + 0] = w[0] # (0,0)
        gw[base_k*n + base_i + 2] = w[2] # (1,0)
        add_factor(gu, gv, gw)
        
        # 3
        u = [0]*4; v = [0]*4; w = [0]*4
        u[0]=1
        v[2]=1; v[3]=-1
        w[1]=-1; w[3]=1
        gu = [0]*(n*m); gv = [0]*(m*p); gw = [0]*(p*n)
        gu[base_i*m + base_j + 0] = u[0] # (0,0)
        gv[base_j*p + base_k + 2] = v[2] # (1,0)
        gv[base_j*p + base_k + 3] = v[3] # (1,1)
        gw[base_k*n + base_i + 1] = w[1] # (0,1)
        gw[base_k*n + base_i + 3] = w[3] # (1,1)
        add_factor(gu, gv, gw)
        
        # 4
        u = [0]*4; v = [0]*4; w = [0]*4
        u[3]=1
        v[1]=1; v[0]=-1
        w[1]=1; w[2]=1
        gu = [0]*(n*m); gv = [0]*(m*p); gw = [0]*(p*n)
        gu[base_i*m + base_j + 3] = u[3] # (1,1)
        gv[base_j*p + base_k + 1] = v[1] # (0,1)
        gv[base_j*p + base_k + 0] = v[0] # (0,0)
        gw[base_k*n + base_i + 1] = w[1] # (0,1)
        gw[base_k*n + base_i + 2] = w[2] # (1,0)
        add_factor(gu, gv, gw)
        
        # 5
        u = [0]*4; v = [0]*4; w = [0]*4
        u[0]=1; u[1]=1
        v[3]=1
        w[1]=-1; w[3]=1
        gu = [0]*(n*m); gv = [0]*(m*p); gw = [0]*(p*n)
        gu[base_i*m + base_j + 0] = u[0] # (0,0)
        gu[base_i*m + base_j + 1] = u[1] # (0,1)
        gv[base_j*p + base_k + 3] = v[3] # (1,1)
        gw[base_k*n + base_i + 1] = w[1] # (0,1)
        gw[base_k*n + base_i + 3] = w[3] # (1,1)
        add_factor(gu, gv, gw)
        
        # 6
        u = [0]*4; v = [0]*4; w = [0]*4
        u[2]=1; u[0]=-1
        v[0]=1; v[2]=1
        w[2]=1; w[3]=1
        gu = [0]*(n*m); gv = [0]*(m*p); gw = [0]*(p*n)
        gu[base_i*m + base_j + 2] = u[2] # (1,0)
        gu[base_i*m + base_j + 0] = u[0] # (0,0)
        gv[base_j*p + base_k + 0] = v[0] # (0,0)
        gv[base_j*p + base_k + 2] = v[2] # (1,0)
        gw[base_k*n + base_i + 2] = w[2] # (1,0)
        gw[base_k*n + base_i + 3] = w[3] # (1,1)
        add_factor(gu, gv, gw)
        
        # 7
        u = [0]*4; v = [0]*4; w = [0]*4
        u[1]=1; u[3]=-1
        v[3]=1; v[1]=1
        w[2]=-1; w[3]=1
        gu = [0]*(n*m); gv = [0]*(m*p); gw = [0]*(p*n)
        gu[base_i*m + base_j + 1] = u[1] # (0,1)
        gu[base_i*m + base_j + 3] = u[3] # (1,1)
        gv[base_j*p + base_k + 3] = v[3] # (1,1)
        gv[base_j*p + base_k + 1] = v[1] # (0,1)
        gw[base_k*n + base_i + 2] = w[2] # (1,0)
        gw[base_k*n + base_i + 3] = w[3] # (1,1)
        add_factor(gu, gv, gw)
        
    # Helper for 1x2xK (Rank 1)
    def add_rank1_1x2xK(base_i, base_j, base_k, K):
        # A is 1x2 (rows base_i, cols base_j, base_j+1)
        # B is 2xK (rows base_j, base_j+1; cols base_k .. base_k+K-1)
        # C is 1xK (rows base_i; cols base_k .. base_k+K-1)
        # Tensor T[i, j, k] = 1 for i=base_i, j in {base_j, base_j+1}, k in range
        # Rank 1 factor:
        # U[i, c] = 1
        # V[j, c] = 1
        # W[k, c] = 1
        
        gu = [0]*(n*m); gv = [0]*(m*p); gw = [0]*(p*n)
        
        gu[base_i*m + base_j] = 1
        gu[base_i*m + base_j + 1] = 1
        
        for jj in range(2):
            for kk in range(K):
                gv[(base_j+jj)*p + base_k + kk] = 1
                
        for kk in range(K):
            gw[(base_k+kk)*n + base_i] = 1
            
        add_factor(gu, gv, gw)
        
    # Helper for 2x1xK (Rank 1) - Not needed for our split, but good to have
    # Helper for 2x2x1 (Rank 4)
    def add_2x2x1(base_i, base_j, base_k):
        # Schoolbook for 2x2x1
        # 4 factors
        for ii in range(2):
            for jj in range(2):
                gu = [0]*(n*m); gv = [0]*(m*p); gw = [0]*(p*n)
                gu[(base_i+ii)*m + base_j+jj] = 1
                gv[(base_j+jj)*p + base_k] = 1
                gw[base_k*n + base_i+ii] = 1
                add_factor(gu, gv, gw)
                
    # Construction
    # Split m=5 into 2, 2, 1
    # Block 1: A[:, 0:2] * B[0:2, :]
    # A1 is 3x2 (rows 0-2, cols 0-1). B1 is 2x7 (rows 0-1, cols 0-6).
    # Split A1 into 2x2 (rows 0-1) and 1x2 (row 2).
    # Split B1 into 2x4 (cols 0-3) and 2x3 (cols 4-6).
    
    # Sub-block 1.1: A[0:2, 0:2] * B[0:2, 0:4] (2x2x4)
    # Split B into 2x2 (cols 0-1) and 2x2 (cols 2-3)
    add_strassen_2x2x2(0, 0, 0) # A[0:2,0:2] * B[0:2,0:2]
    add_strassen_2x2x2(0, 0, 2) # A[0:2,0:2] * B[0:2,2:4]
    
    # Sub-block 1.2: A[0:2, 0:2] * B[0:2, 4:7] (2x2x3)
    # Split B into 2x2 (cols 4-5) and 2x1 (col 6)
    add_strassen_2x2x2(0, 0, 4) # A[0:2,0:2] * B[0:2,4:6]
    add_2x2x1(0, 0, 6)          # A[0:2,0:2] * B[0:2,6:7]
    
    # Sub-block 1.3: A[2:3, 0:2] * B[0:2, 0:4] (1x2x4)
    add_rank1_1x2xK(2, 0, 0, 4)
    
    # Sub-block 1.4: A[2:3, 0:2] * B[0:2, 4:7] (1x2x3)
    add_rank1_1x2xK(2, 0, 4, 3)
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    # A2 is 3x2 (rows 0-2, cols 2-3). B2 is 2x7 (rows 2-3, cols 0-6).
    
    # Sub-block 2.1: A[0:2, 2:4] * B[2:4, 0:4] (2x2x4)
    add_strassen_2x2x2(0, 2, 0)
    add_strassen_2x2x2(0, 2, 2)
    
    # Sub-block 2.2: A[0:2, 2:4] * B[2:4, 4:7] (2x2x3)
    add_strassen_2x2x2(0, 2, 4)
    add_2x2x1(0, 2, 6)
    
    # Sub-block 2.3: A[2:3, 2:4] * B[2:4, 0:4] (1x2x4)
    add_rank1_1x2xK(2, 2, 0, 4)
    
    # Sub-block 2.4: A[2:3, 2:4] * B[2:4, 4:7] (1x2x3)
    add_rank1_1x2xK(2, 2, 4, 3)
    
    # Block 3: A[:, 4:5] * B[4:5, :]
    # A3 is 3x1 (row 0-2, col 4). B3 is 1x7 (row 4, col 0-6).
    # This is 3x1x7.
    # Rank 1?
    # T[i, 4, k] = 1.
    # U[i, c] = 1 for i in 0..2.
    # V[4, c] = 1.
    # W[k, c] = 1 for k in 0..6.
    
    gu = [0]*(n*m); gv = [0]*(m*p); gw = [0]*(p*n)
    for i in range(3):
        gu[i*m + 4] = 1
    gv[4*p + 0] = 1 # V index (4, 0)? No, V index is (j, k). j=4. k=0..6.
    # Wait, V shape is (m*p, rank). Index is j*p + k.
    # B3 is row 4 of B. So j=4.
    # C3 is col k of C. So k=0..6.
    # W index is (k, i).
    
    for k in range(7):
        gv[4*p + k] = 1
        gw[k*n + 0] = 1 # W index (k, i). i=0?
        # Wait, W index is (k, i).
        # C[i, k].
        # My W mapping: gw[k*n + i].
        # So for each k, and i in 0..2, gw[k*n + i] should be 1?
        # Let's check the tensor T[i, 4, k] = 1.
        # Factor: U[i, c] * V[4, c] * W[k, c].
        # If U[i, c]=1, V[4, c]=1, W[k, c]=1, then product is 1.
        # So W[k, c] must be 1 for all k.
        # And U[i, c] must be 1 for all i.
        
    # Correcting Block 3 factor
    # Remove previous incorrect partial adds if any (none yet for block 3)
    # Add correct factor
    gu = [0]*(n*m); gv = [0]*(m*p); gw = [0]*(p*n)
    for i in range(3):
        gu[i*m + 4] = 1
    for k in range(7):
        gv[4*p + k] = 1
        for i in range(3):
            gw[k*n + i] = 1
    add_factor(gu, gv, gw)
    
    # Assemble final matrices
    rank = len(U_cols)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_vec = U_cols[c]
        v_vec = V_cols[c]
        w_vec = W_cols[c]
        for i in range(n * m):
            if u_vec[i] != 0:
                U[i][c] = u_vec[i]
        for j in range(m * p):
            if v_vec[j] != 0:
                V[j][c] = v_vec[j]
        for k in range(p * n):
            if w_vec[k] != 0:
                W[k][c] = w_vec[k]
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
