import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=3, m=4, p=6
    # U: 12 x R
    # V: 24 x R
    # W: 18 x R
    
    # We will build the decomposition dynamically.
    # Start with empty lists for columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_val, v_val, w_val):
        U_cols.append(u_val)
        V_cols.append(v_val)
        W_cols.append(w_val)
        
    def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        # A is 2x2 at (a_row_start, a_col_start)
        # B is 2x2 at (b_row_start, b_col_start)
        # C is 2x2 at (c_row_start, c_col_start)
        # Indices in U: a_row_start + i, a_col_start + j
        # Indices in V: b_row_start + i, b_col_start + j
        # Indices in W: c_col_start + j, c_row_start + i (Trace convention W[k*n+i, r] where C[i,j] = sum U[i,k]V[k,j] -> W[j*n+i]? 
        # Task says: W[k*n+i, r] for C[i, k]? 
        # "output factor follows the official trace convention W[k*n+i,r]"
        # Standard: C[i,j] = sum_k U[i,k] V[k,j].
        # Tensor T_{ijk} = U_{ik} V_{kj} W_{jk}?
        # Usually W_{j n + i} corresponds to C_{i,j}.
        # Let's assume W index is j*n + i.
        
        # Strassen P1..P7
        # P1 = (a11+a22)(b11+b22)
        # P2 = (a21+a22)b11
        # P3 = a11(b12-b22)
        # P4 = a22(b21-b11)
        # P5 = (a11+a12)b22
        # P6 = (a21-a11)(b11+b12)
        # P7 = (a12-a22)(b21+b22)
        
        # M11 = P1 - P2 + P4 - P7
        # M12 = P3 + P4
        # M21 = P1 + P2
        # M22 = P3 - P5 + P6 - P7
        
        # We add 7 terms.
        
        # Term 1: P1
        # U: a11+a22 -> (1,0) + (0,1) relative to A
        # V: b11+b22 -> (1,0) + (0,1) relative to B
        # W: M11 coeff 1, M21 coeff 1
        # W indices: M11 is (c_row_start, c_col_start) -> W[c_col_start*n + c_row_start]
        #            M21 is (c_row_start+1, c_col_start) -> W[c_col_start*n + c_row_start+1]
        
        u = [0]*12
        v = [0]*24
        w = [0]*18
        
        u[a_row_start + 0 + 0*m] += 1 # a11
        u[a_row_start + 1 + 0*m] += 1 # a22
        # Wait, U index is i*m + j.
        # A is at (a_row_start, a_col_start).
        # A[i,j] is at U[a_row_start+i, a_col_start+j]? No, U is (n*m, rank).
        # U row index is i*m + j.
        
        # Let's correct U index:
        # A[i,j] corresponds to U[(a_row_start+i)*m + (a_col_start+j)]
        
        u = [0]*12
        v = [0]*24
        w = [0]*18
        
        # P1
        u[(a_row_start+0)*m + (a_col_start+0)] += 1
        u[(a_row_start+1)*m + (a_col_start+1)] += 1
        
        v[(b_row_start+0)*p + (b_col_start+0)] += 1
        v[(b_row_start+1)*p + (b_col_start+1)] += 1
        
        w[(c_col_start+0)*n + (c_row_start+0)] += 1 # M11
        w[(c_col_start+0)*n + (c_row_start+1)] += 1 # M21
        
        add_term(u, v, w)
        
        # P2
        u = [0]*12
        v = [0]*24
        w = [0]*18
        
        u[(a_row_start+1)*m + (a_col_start+0)] += 1 # a21
        u[(a_row_start+1)*m + (a_col_start+1)] += 1 # a22
        
        v[(b_row_start+0)*p + (b_col_start+0)] += 1 # b11
        
        w[(c_col_start+0)*n + (c_row_start+0)] += -1 # M11
        w[(c_col_start+0)*n + (c_row_start+1)] += 1 # M21
        
        add_term(u, v, w)
        
        # P3
        u = [0]*12
        v = [0]*24
        w = [0]*18
        
        u[(a_row_start+0)*m + (a_col_start+0)] += 1 # a11
        
        v[(b_row_start+0)*p + (b_col_start+1)] += 1 # b12
        v[(b_row_start+1)*p + (b_col_start+1)] += -1 # b22
        
        w[(c_col_start+1)*n + (c_row_start+0)] += 1 # M12
        w[(c_col_start+1)*n + (c_row_start+1)] += 1 # M22
        
        add_term(u, v, w)
        
        # P4
        u = [0]*12
        v = [0]*24
        w = [0]*18
        
        u[(a_row_start+1)*m + (a_col_start+1)] += 1 # a22
        
        v[(b_row_start+1)*p + (b_col_start+0)] += 1 # b21
        v[(b_row_start+0)*p + (b_col_start+0)] += -1 # b11
        
        w[(c_col_start+0)*n + (c_row_start+0)] += 1 # M11
        w[(c_col_start+1)*n + (c_row_start+0)] += 1 # M12
        
        add_term(u, v, w)
        
        # P5
        u = [0]*12
        v = [0]*24
        w = [0]*18
        
        u[(a_row_start+0)*m + (a_col_start+0)] += 1 # a11
        u[(a_row_start+0)*m + (a_col_start+1)] += 1 # a12
        
        v[(b_row_start+1)*p + (b_col_start+1)] += 1 # b22
        
        w[(c_col_start+1)*n + (c_row_start+1)] += -1 # M22
        
        add_term(u, v, w)
        
        # P6
        u = [0]*12
        v = [0]*24
        w = [0]*18
        
        u[(a_row_start+1)*m + (a_col_start+0)] += 1 # a21
        u[(a_row_start+0)*m + (a_col_start+0)] += -1 # a11
        
        v[(b_row_start+0)*p + (b_col_start+0)] += 1 # b11
        v[(b_row_start+0)*p + (b_col_start+1)] += 1 # b12
        
        w[(c_col_start+1)*n + (c_row_start+1)] += 1 # M22
        
        add_term(u, v, w)
        
        # P7
        u = [0]*12
        v = [0]*24
        w = [0]*18
        
        u[(a_row_start+0)*m + (a_col_start+1)] += 1 # a12
        u[(a_row_start+1)*m + (a_col_start+1)] += -1 # a22
        
        v[(b_row_start+1)*p + (b_col_start+0)] += 1 # b21
        v[(b_row_start+1)*p + (b_col_start+1)] += 1 # b22
        
        w[(c_col_start+0)*n + (c_row_start+0)] += -1 # M11
        w[(c_col_start+1)*n + (c_row_start+1)] += -1 # M22
        
        add_term(u, v, w)

    def add_dot_product(a_row, a_col_start, b_row_start, b_col, c_row, c_col):
        # A is 1 x 4 (row a_row, cols a_col_start..a_col_start+3)
        # B is 4 x 1 (rows b_row_start..b_row_start+3, col b_col)
        # C is 1 x 1 (row c_row, col c_col)
        # C = sum_{k=0}^3 A[a_row, a_col_start+k] * B[b_row_start+k, b_col]
        
        for k in range(4):
            u = [0]*12
            v = [0]*24
            w = [0]*18
            
            u[a_row*m + (a_col_start+k)] += 1
            v[(b_row_start+k)*p + b_col] += 1
            w[c_col*n + c_row] += 1
            
            add_term(u, v, w)

    # Block 1: A_top (0:2, 0:4) * B_left (0:4, 0:4) -> C (0:2, 0:4)
    # A11 (0:2, 0:2), A12 (0:2, 2:4)
    # B11 (0:2, 0:2), B12 (0:2, 2:4)
    # B21 (2:4, 0:2), B22 (2:4, 2:4)
    
    # C[0:2, 0:2] = A11*B11 + A12*B21
    add_strassen_block(0, 0, 0, 0, 0, 0) # A11*B11
    add_strassen_block(0, 2, 2, 0, 0, 0) # A12*B21
    
    # C[0:2, 2:4] = A11*B12 + A12*B22
    add_strassen_block(0, 0, 0, 2, 0, 2) # A11*B12
    add_strassen_block(0, 2, 2, 2, 0, 2) # A12*B22

    # Block 2: A_top (0:2, 0:4) * B_right (0:4, 4:6) -> C (0:2, 4:6)
    # B31 (0:2, 4:6), B32 (2:4, 4:6)
    
    # C[0:2, 4:6] = A11*B31 + A12*B32
    add_strassen_block(0, 0, 0, 4, 0, 4) # A11*B31
    add_strassen_block(0, 2, 2, 4, 0, 4) # A12*B32

    # Block 3: A_bot (2, 0:4) * B_left (0:4, 0:4) -> C (2, 0:4)
    for j in range(4):
        add_dot_product(2, 0, 0, j, 2, j)

    # Block 4: A_bot (2, 0:4) * B_right (0:4, 4:6) -> C (2, 4:6)
    for j in range(4, 6):
        add_dot_product(2, 0, 0, j, 2, j)

    # Convert lists of columns to matrices
    rank = len(U_cols)
    U = [[0]*rank for _ in range(12)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(18)]
    
    for r in range(rank):
        for i in range(12):
            U[i][r] = U_cols[r][i]
        for i in range(24):
            V[i][r] = V_cols[r][i]
        for i in range(18):
            W[i][r] = W_cols[r][i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
