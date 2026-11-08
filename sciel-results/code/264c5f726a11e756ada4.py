"""Matrix multiplication 4,4,8 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses a recursive Strassen-like decomposition:
    - Split B (4x8) into two 4x4 halves
    - Apply rank-49 Strassen decomposition for 4x4*4x4 to each half
    - Total rank: 98
    
    Returns dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (16, rank)
    - V has shape (m*p, rank) = (32, rank)  
    - W has shape (p*n, rank) = (32, rank)
    
    Using the W[k*n+i, r] trace convention.
    """
    n, m, p = 4, 4, 8
    
    # Schoolbook decomposition as fallback (rank = n*m*p = 128)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}
    
    start_time = time.time()
    
    # Try to build Strassen decomposition
    try:
        # Strassen 2x2x2 terms: 7 terms
        # Each term: (U_entries, V_entries, W_entries)
        # U_entries: dict {(i,j): coeff} for A (2x2)
        # V_entries: dict {(j,k): coeff} for B (2x2)
        # W_entries: dict {(k,i): coeff} for C (2x2), using k*n+i convention where n=2
        
        s2_terms = [
            # M1 = (A11+A22)(B11+B22)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2 = (A21+A22)B11
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
            # M3 = A11(B21-B22)
            ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4 = A22(B12-B11)
            ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            # M5 = (A11+A12)B22
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M6 = (A21-A11)(B11+B12)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7 = (A12-A22)(B21+B22)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        # Build 49-term decomposition for 4x4*4x4
        # Split 4x4 into 2x2 blocks: [[A00, A01], [A10, A11]]
        # Strassen at block level:
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        # M2 = (A10+A11)B00 -> C10 - C11
        # M3 = A00(B10-B11) -> C01 + C11
        # M4 = A11(B01-B00) -> C00 + C10
        # M5 = (A00+A01)B11 -> -C00 + C01
        # M6 = (A10-A00)(B00+B01) -> C11
        # M7 = (A01-A11)(B10+B11) -> C00
        
        big_terms = [
            # (A_blocks, B_blocks, C_blocks)
            # Each block spec: list of ((block_row, block_col), coeff)
            # A_blocks: for A matrix blocks
            # B_blocks: for B matrix blocks
            # C_blocks: for C output blocks
            ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
            ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
            ([((0,0), 1)], [((1,0), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
            ([((1,1), 1)], [((0,1), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
            ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
            ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
            ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ]
        
        terms_49 = []
        for a_blocks, b_blocks, c_blocks in big_terms:
            for u2, v2, w2 in s2_terms:
                U_dict = {}
                V_dict = {}
                W_dict = {}
                
                # Build U for 4x4 A matrix
                for (br, bc), ac in a_blocks:
                    for (i, j), uc in u2.items():
                        gi, gj = br * 2 + i, bc * 2 + j
                        key = (gi, gj)
                        U_dict[key] = U_dict.get(key, 0) + ac * uc
                
                # Build V for 4x4 B matrix
                for (br, bc), bc_coeff in b_blocks:
                    for (j, k), vc in v2.items():
                        gi, gj = br * 2 + j, bc * 2 + k
                        key = (gi, gj)
                        V_dict[key] = V_dict.get(key, 0) + bc_coeff * vc
                
                # Build W for 4x4 C matrix using W[k*n+i] convention where n=4
                for (br, bc), cc in c_blocks:
                    for (k_sub, i_sub), wc in w2.items():
                        global_row = br * 2 + i_sub
                        global_col = bc * 2 + k_sub
                        key = (global_col, global_row)  # (k, i) in W convention
                        W_dict[key] = W_dict.get(key, 0) + cc * wc
                
                terms_49.append((U_dict, V_dict, W_dict))
        
        # Build 4x4 * 4x8 decomposition
        # Split B into B1 (cols 0-3) and B2 (cols 4-7)
        all_terms = []
        
        # For B1:
        for U_dict, V_dict, W_dict in terms_49:
            new_V = {}
            for (j, k), coeff in V_dict.items():
                new_V[(j, k)] = coeff  # B1 columns 0-3
            all_terms.append((U_dict, new_V, W_dict))
        
        # For B2:
        for U_dict, V_dict, W_dict in terms_49:
            new_V = {}
            for (j, k), coeff in V_dict.items():
                new_V[(j, k + 4)] = coeff  # B2 columns 4-7
            new_W = {}
            for (k, i), coeff in W_dict.items():
                new_W[(k + 4, i)] = coeff  # C columns 4-7
            all_terms.append((U_dict, new_V, new_W))
        
        rank = len(all_terms)  # Should be 98
        
        # Build U, V, W matrices
        U = [[0] * rank for _ in range(n * m)]  # (16, rank)
        V = [[0] * rank for _ in range(m * p)]  # (32, rank)
        W = [[0] * rank for _ in range(p * n)]  # (32, rank)
        
        for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
            for (i, j), coeff in U_dict.items():
                if coeff != 0:
                    U[i * m + j][c] = coeff
            for (j, k), coeff in V_dict.items():
                if coeff != 0:
                    V[j * p + k][c] = coeff
            for (k, i), coeff in W_dict.items():
                if coeff != 0:
                    W[k * n + i][c] = coeff
        
        return {'U': U, 'V': V, 'W': W}
    
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END

