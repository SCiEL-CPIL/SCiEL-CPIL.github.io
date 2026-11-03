"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen 2x2*2x2 factors: 7 factors
    # Each factor: (U_entries, V_entries, W_entries)
    # U: left matrix 2x2, indexed (row, col) -> local index row*2+col
    # V: right matrix 2x2, indexed (col, out) -> local index col*2+out  
    # W: output 2x2, indexed (out, row) for trace convention -> local index out*2+row
    # 
    # Actually for the sub-problem: left is 2x2 (rows i=0,1; cols j=0,1)
    # right is 2x2 (cols j=0,1; outs k=0,1)
    # output is 2x2 (rows i=0,1; outs k=0,1)
    # W index in global: k*n + i
    
    # Strassen factors as (list of (local_u_idx, coeff), list of (local_v_idx, coeff), list of (local_w_idx, coeff))
    # local_u_idx: row*2+col for left 2x2
    # local_v_idx: col*2+out for right 2x2
    # local_w_idx: out*2+row for output 2x2 (trace convention: W[k*n+i])
    
    strassen_factors = [
        # M1 = (a11+a22)(b11+b22)
        {
            'U': [(0, 1), (3, 1)],   # a11=idx0, a22=idx3
            'V': [(0, 1), (3, 1)],   # b11=idx0, b22=idx3
            'W': [(0, 1), (3, 1)]    # c11=idx0, c22=idx3
        },
        # M2 = (a21+a22)b11
        {
            'U': [(2, 1), (3, 1)],   # a21=idx2, a22=idx3
            'V': [(0, 1)],           # b11=idx0
            'W': [(2, 1)]            # c21=idx2
        },
        # M3 = a11(b12-b22)
        {
            'U': [(0, 1)],           # a11=idx0
            'V': [(1, 1), (3, -1)],  # b12=idx1, b22=idx3
            'W': [(1, 1)]            # c12=idx1
        },
        # M4 = a22(b21-b11)
        {
            'U': [(3, 1)],           # a22=idx3
            'V': [(2, 1), (0, -1)],  # b21=idx2, b11=idx0
            'W': [(0, 1), (2, 1)]    # c11=idx0, c21=idx2
        },
        # M5 = (a11+a12)b22
        {
            'U': [(0, 1), (1, 1)],   # a11=idx0, a12=idx1
            'V': [(3, 1)],           # b22=idx3
            'W': [(0, -1), (1, 1)]   # c11=idx0, c12=idx1
        },
        # M6 = (a21-a11)(b11+b12)
        {
            'U': [(2, 1), (0, -1)],  # a21=idx2, a11=idx0
            'V': [(0, 1), (1, 1)],   # b11=idx0, b12=idx1
            'W': [(3, 1)]            # c22=idx3
        },
        # M7 = (a12-a22)(b21+b22)
        {
            'U': [(1, 1), (3, -1)],  # a12=idx1, a22=idx3
            'V': [(2, 1), (3, 1)],   # b21=idx2, b22=idx3
            'W': [(0, 1)]            # c11=idx0
        },
    ]
    
    rank = 8 * 7  # 8 sub-problems, 7 Strassen factors each
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for j_group in range(2):       # two halves of m=4
        for k_group in range(4):   # four quarters of p=8
            for s in range(7):     # 7 Strassen factors
                factor = strassen_factors[s]
                
                # U entries: global row = i*m + j, where i=0,1 and j=j_group*2+{0,1}
                for (local_u, coeff) in factor['U']:
                    u_row = local_u // 2  # i
                    u_col = local_u % 2   # j within group
                    global_u = u_row * m + (j_group * 2 + u_col)
                    U[global_u][col] = coeff
                
                # V entries: global row = j*p + k, where j=j_group*2+{0,1} and k=k_group*2+{0,1}
                for (local_v, coeff) in factor['V']:
                    v_col = local_v // 2  # j within group
                    v_out = local_v % 2   # k within group
                    global_v = (j_group * 2 + v_col) * p + (k_group * 2 + v_out)
                    V[global_v][col] = coeff
                
                # W entries: global index = k*n + i, where k=k_group*2+{0,1} and i=0,1
                for (local_w, coeff) in factor['W']:
                    w_out = local_w // 2  # k within group
                    w_row = local_w % 2   # i
                    global_w = (k_group * 2 + w_out) * n + w_row
                    W[global_w][col] = coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
