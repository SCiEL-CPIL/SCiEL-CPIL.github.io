"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    
    A (4x4) split into 2x2 blocks: A[rb, cb] for rb in {0,1}, cb in {0,1}
    B (4x6) split into 2x2 blocks: B[rb, cb] for rb in {0,1}, cb in {0,1,2}
    C (4x6) split into 2x2 blocks: C[rb, cb] for rb in {0,1}, cb in {0,1,2}
    
    C[rb, cb] = A[rb,0]*B[0,cb] + A[rb,1]*B[1,cb]
    Each 2x2*2x2 product uses Strassen (7 mults), so 14 per output block.
    6 output blocks * 14 = 84 total rank.
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_term(offset_i, offset_j, offset_k):
        """
        Add Strassen decomposition of X_block * Y_block where:
        X_block is 2x2 block of A at rows offset_i..offset_i+1, cols offset_j..offset_j+1
        Y_block is 2x2 block of B at rows offset_j..offset_j+1, cols offset_k..offset_k+1
        Output contributes to C at rows offset_i..offset_i+1, cols offset_k..offset_k+1
        
        Strassen for X*Y (X,Y are 2x2):
        m1 = (x00+x11)(y00+y11)
        m2 = (x10+x11)y00
        m3 = x00(y01-y11)
        m4 = x11(y10-y00)
        m5 = (x00+x01)y11
        m6 = (x10-x00)(y00+y01)
        m7 = (x01-x11)(y10+y11)
        
        C00 = m1 + m4 - m5 + m7
        C01 = m3 + m5
        C10 = m2 + m4
        C11 = m1 - m2 + m3 + m6
        """
        i0, j0, k0 = offset_i, offset_j, offset_k
        
        # Define the 7 Strassen multiplications as (U_dict, V_dict, W_dict)
        # U_dict: {(i,j): coeff} for A elements
        # V_dict: {(j,k): coeff} for B elements  
        # W_dict: {(i,k): coeff} for C output contribution
        
        strassen_mults = [
            # m1 = (x00+x11)(y00+y11)
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            # m2 = (x10+x11)y00
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            # m3 = x00(y01-y11)
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            # m4 = x11(y10-y00)
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            # m5 = (x00+x01)y11
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            # m6 = (x10-x00)(y00+y01)
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            # m7 = (x01-x11)(y10+y11)
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        
        for U_coeffs, V_coeffs, W_coeffs in strassen_mults:
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            for (i, j), coeff in U_coeffs.items():
                U_col[i * m + j] = coeff
            
            for (j, k), coeff in V_coeffs.items():
                V_col[j * p + k] = coeff
            
            for (i, k), coeff in W_coeffs.items():
                W_col[k * n + i] = coeff
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    # For each output block (rb, cb):
    # C[rb, cb] = A[rb, 0]*B[0, cb] + A[rb, 1]*B[1, cb]
    for rb in range(2):
        for cb in range(3):
            # Term 1: A[rb, 0] * B[0, cb]
            add_strassen_term(rb * 2, 0, cb * 2)
            # Term 2: A[rb, 1] * B[1, cb]
            add_strassen_term(rb * 2, 2, cb * 2)
    
    rank = len(U_cols)
    
    # Build final matrices: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
