"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # U: n*m = 6 rows, indexed i*m + j
    # V: m*p = 30 rows, indexed j*p + k
    # W: p*n = 20 rows, indexed k*n + i
    # C = A * B where A is n x m (2x3), B is m x p (3x10), C is n x p (2x10)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # W[k*n+i] corresponds to C[i,k]

    rank = 0
    U_list = []
    V_list = []
    W_list = []

    def add_term(u_indices, v_indices, w_indices):
        nonlocal rank
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for idx, val in u_indices:
            u[idx] = val
        for idx, val in v_indices:
            v[idx] = val
        for idx, val in w_indices:
            w[idx] = val
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        rank += 1

    # Block 1: First 2 columns of B (k=0,1)
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B1 = [[b00, b01], [b10, b11], [b20, b21]]
    # C1 = A * B1 (2x2)
    # Partition A into A1 (2x2: a00,a01,a10,a11) and A2 (2x1: a02,a12)
    # Partition B1 into B11 (2x2: b00,b01,b10,b11) and B12 (1x2: b20,b21)
    # C1 = A1*B11 + A2*B12
    
    # Strassen for A1*B11 (rank 7):
    # A1 = [[a00, a01], [a10, a11]]
    # B11 = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # C11[0,0] = M1 - M5 + M7
    # C11[0,1] = M3 + M5
    # C11[1,0] = M2 + M4
    # C11[1,1] = M1 - M2 + M6
    
    # A2*B12 (schoolbook, rank 4):
    # C12[0,0] = a02*b20
    # C12[0,1] = a02*b21
    # C12[1,0] = a12*b20
    # C12[1,1] = a12*b21

    # Strassen terms:
    # M1 = (a00+a11)*(b00+b11)
    # U: a00 at i=0,j=0 -> idx 0; a11 at i=1,j=1 -> idx 3
    # V: b00 at j=0,k=0 -> idx 0; b11 at j=1,k=1 -> idx 11
    # W: C[0,0] at k=0,i=0 -> idx 0; C[1,1] at k=1,i=1 -> idx 1*2+1=3
    add_term([(0,1),(3,1)], [(0,1),(11,1)], [(0,1),(3,1)])
    
    # M2 = (a10+a11)*b00
    # U: a10 at i=1,j=0 -> idx 2; a11 at i=1,j=1 -> idx 3
    # V: b00 at j=0,k=0 -> idx 0
    # W: C[1,0] at k=0,i=1 -> idx 0*2+1=1; C[1,1] at k=1,i=1 -> idx 3
    add_term([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)])
    
    # M3 = a00*(b01-b11)
    # U: a00 at i=0,j=0 -> idx 0
    # V: b01 at j=0,k=1 -> idx 1; b11 at j=1,k=1 -> idx 11
    # W: C[0,1] at k=1,i=0 -> idx 1*2+0=2
    add_term([(0,1)], [(1,1),(11,-1)], [(2,1)])
    
    # M4 = a11*(b10-b00)
    # U: a11 at i=1,j=1 -> idx 3
    # V: b10 at j=1,k=0 -> idx 10; b00 at j=0,k=0 -> idx 0
    # W: C[1,0] at k=0,i=1 -> idx 1
    add_term([(3,1)], [(10,1),(0,-1)], [(1,1)])
    
    # M5 = (a00+a01)*b11
    # U: a00 at i=0,j=0 -> idx 0; a01 at i=0,j=1 -> idx 1
    # V: b11 at j=1,k=1 -> idx 11
    # W: C[0,0] at k=0,i=0 -> idx 0; C[0,1] at k=1,i=0 -> idx 2
    add_term([(0,1),(1,1)], [(11,1)], [(0,-1),(2,1)])
    
    # M6 = (a10-a00)*(b00+b01)
    # U: a10 at i=1,j=0 -> idx 2; a00 at i=0,j=0 -> idx 0
    # V: b00 at j=0,k=0 -> idx 0; b01 at j=0,k=1 -> idx 1
    # W: C[1,1] at k=1,i=1 -> idx 3
    add_term([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)])
    
    # M7 = (a01-a11)*(b10+b11)
    # U: a01 at i=0,j=1 -> idx 1; a11 at i=1,j=1 -> idx 3
    # V: b10 at j=1,k=0 -> idx 10; b11 at j=1,k=1 -> idx 11
    # W: C[0,0] at k=0,i=0 -> idx 0
    add_term([(1,1),(3,-1)], [(10,1),(11,1)], [(0,1)])

    # Schoolbook for A2*B12 (4 terms):
    # a02*b20: U idx 5 (i=0,j=2), V idx 20 (j=2,k=0), W idx 0 (k=0,i=0)
    add_term([(5,1)], [(20,1)], [(0,1)])
    # a02*b21: U idx 5, V idx 21 (j=2,k=1), W idx 2 (k=1,i=0)
    add_term([(5,1)], [(21,1)], [(2,1)])
    # a12*b20: U idx 6 (i=1,j=2), V idx 20, W idx 1 (k=0,i=1)
    add_term([(6,1)], [(20,1)], [(1,1)])
    # a12*b21: U idx 6, V idx 21, W idx 3 (k=1,i=1)
    add_term([(6,1)], [(21,1)], [(3,1)])

    # Block 2-5: Remaining 8 columns of B (k=2..9), schoolbook
    # For each i in [0,1], j in [0,1,2], k in [2..9]:
    # U[i*m+j], V[j*p+k], W[k*n+i]
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
