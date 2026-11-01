def construct():
    """
    Rank-42 exact decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Decomposition: C = A[:,0:2]@B[0:2,:] + A[:,2:4]@B[2:4,:]
    Each 2x2 @ 2x6 term splits into three 2x2 @ 2x2 multiplications.
    Each 2x2 @ 2x2 uses Strassen (rank 7).
    Total rank = 2 * 3 * 7 = 42.
    
    Interface:
    U: (n*m, rank) = (8, 42)
    V: (m*p, rank) = (24, 42)
    W: (p*n, rank) = (12, 42)
    
    Indexing:
    U[i*m + j, r] for A[i,j]
    V[j*p + k, r] for B[j,k]
    W[k*n + i, r] for C[i,k]
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 @ 2x2: A = [[a0,a1],[a2,a3]], B = [[b0,b1],[b2,b3]]
    # Products:
    #   M1 = (a0+a3)*(b0+b3)
    #   M2 = (a2+a3)*b0
    #   M3 = a0*(b1-b3)
    #   M4 = a3*(b2-b0)
    #   M5 = (a0+a1)*b3
    #   M6 = (a2-a0)*(b0+b1)
    #   M7 = (a1-a3)*(b2+b3)
    # Output:
    #   c0 = M1 + M4 - M5 + M7
    #   c1 = M3 + M5
    #   c2 = M2 + M4
    #   c3 = M1 - M2 + M3 + M6
    
    # Each entry: list of (local_index, sign)
    # Local U/V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # M1
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
        # M3
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
        # M4
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (2, 1)]},
        # M5
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
        # M6
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]},
    ]
    
    col = 0
    for block in range(2):       # 0: A cols 0-1, B rows 0-1; 1: A cols 2-3, B rows 2-3
        for sub in range(3):     # 0: B cols 0-1; 1: B cols 2-3; 2: B cols 4-5
            for s_idx in range(7):
                term = strassen_terms[s_idx]
                
                # U: A[i][block*2 + l] -> U[i*m + block*2 + l, col]
                for local_u, coeff in term['U']:
                    a_row = local_u // 2
                    a_col = local_u % 2
                    u_idx = a_row * m + (2 * block + a_col)
                    U[u_idx][col] += coeff
                
                # V: B[block*2 + l][sub*2 + c] -> V[(block*2+l)*p + sub*2+c, col]
                for local_v, coeff in term['V']:
                    b_row = local_v // 2
                    b_col = local_v % 2
                    v_idx = (2 * block + b_row) * p + (2 * sub + b_col)
                    V[v_idx][col] += coeff
                
                # W: C[i][sub*2 + c] -> W[(sub*2+c)*n + i, col]
                for local_w, coeff in term['W']:
                    c_row = local_w // 2
                    c_col = local_w % 2
                    w_idx = (2 * sub + c_col) * n + c_row
                    W[w_idx][col] += coeff
                
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
