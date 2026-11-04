"""Matrix multiplication 6x6x6 - Strassen block decomposition."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-structured Strassen decomposition for 6x6 matrix multiplication.
    Partition 6x6 into 3x3 grid of 2x2 blocks.
    For each (i,j,k) block triple, compute A_{ik} * B_{kj} using Strassen (rank 7).
    Total rank = 27 * 7 = 189.
    
    The output W follows convention W[k*n+i, r] where (k,i) is the (row,col) of C
    and r is the rank index. So W is indexed as W[col_C * n + row_C, rank].
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block size 2, 3 blocks per dimension
    bs = 2
    nb = 3  # number of blocks per dimension
    
    total_rank = 27 * 7  # 189
    
    # U: (n*m, rank) = (36, 189)
    # V: (m*p, rank) = (36, 189)  
    # W: (p*n, rank) = (36, 189)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A coefficients for each Strassen product (index by local position in 2x2: 0=a,1=b,2=c,3=d)
    strassen_A = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]
    
    # B coefficients (0=e, 1=f, 2=g, 3=h)
    strassen_B = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]
    
    # Output coefficients: for each Strassen product t, which C entries get what coefficient
    # C local indices: 0=C11, 1=C12, 2=C21, 3=C22
    output_coeffs = [
        [1, 0, 0, 1],   # M1: C11+=1, C22+=1
        [0, 0, 1, -1],  # M2: C21+=1, C22-=1
        [0, 1, 0, 1],   # M3: C12+=1, C22+=1
        [1, 0, 1, 0],   # M4: C11+=1, C21+=1
        [-1, 1, 0, 0],  # M5: C11-=1, C12+=1
        [0, 0, 0, 1],   # M6: C22+=1
        [1, 0, 0, 0],   # M7: C11+=1
    ]
    
    # For each block triple (i, j, k) and Strassen product t:
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for t in range(7):
                    # A block A_{ik}: rows i*bs..i*bs+1, cols k*bs..k*bs+1
                    # For local (r, c) in 2x2 block:
                    #   global row = i*bs + r, global col = k*bs + c
                    #   U index = (i*bs+r) * m + (k*bs+c)
                    for r in range(bs):
                        for c in range(bs):
                            a_local = r * bs + c  # 0,1,2,3
                            a_coeff = strassen_A[t][a_local]
                            if a_coeff != 0:
                                u_idx = (i * bs + r) * m + (k * bs + c)
                                U[u_idx][col] = a_coeff
                    
                    # B block B_{kj}: rows k*bs..k*bs+1, cols j*bs..j*bs+1
                    # For local (r, c) in 2x2 block:
                    #   global row = k*bs + r, global col = j*bs + c
                    #   V index = (k*bs+r) * p + (j*bs+c)
                    for r in range(bs):
                        for c in range(bs):
                            b_local = r * bs + c
                            b_coeff = strassen_B[t][b_local]
                            if b_coeff != 0:
                                v_idx = (k * bs + r) * p + (j * bs + c)
                                V[v_idx][col] = b_coeff
                    
                    # C block C_{ij}: rows i*bs..i*bs+1, cols j*bs..j*bs+1
                    # For local (r, c) in 2x2 block:
                    #   global row of C = i*bs + r, global col of C = j*bs + c
                    #   W index = col_C * n + row_C = (j*bs+c) * n + (i*bs+r)
                    for r in range(bs):
                        for c in range(bs):
                            c_local = r * bs + c
                            c_coeff = output_coeffs[t][c_local]
                            if c_coeff != 0:
                                w_idx = (j * bs + c) * n + (i * bs + r)
                                W[w_idx][col] = c_coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
