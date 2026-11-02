"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-39 exact decomposition of the 2x2 by 2x11 matrix multiplication tensor.
    
    Strategy: Treat the 2x11 matrix B as consisting of 5 blocks of size 2x2 
    (columns 0-1, 2-3, 4-5, 6-7, 8-9) plus a final single column (column 10).
    Use Strassen's algorithm (rank 7) for each 2x2 block multiplication, 
    and schoolbook (rank 4) for the final 2x1 column.
    
    Total rank = 5 * 7 + 4 = 39.
    
    The output factor follows W[k*n+i, r] convention.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    
    # Number of 2-column blocks: p // 2 = 5 blocks
    # Remaining columns: p % 2 = 1 column
    num_blocks = p // 2  # 5
    rem_cols = p % 2     # 1
    
    # Rank: 7 per 2x2 block + 4 per 2x1 block
    rank = num_blocks * 7 + rem_cols * 4  # 5*7 + 1*4 = 39
    
    # Allocate matrices as lists of lists of integers
    U = [[0] * rank for _ in range(n * m)]    # 4 x 39
    V = [[0] * rank for _ in range(m * p)]    # 22 x 39
    W = [[0] * rank for _ in range(p * n)]    # 22 x 39
    
    col_offset = 0
    
    # Strassen 2x2 decomposition:
    # For A = [[a11, a12], [a21, a22]] and B = [[b11, b12], [b21, b22]]:
    # C11 = (a11+a22)(b11+b22) - a22(b21+b22) + a11(b11-b21)
    # C12 = a12(b11+b22)
    # C21 = a21(a11+a22)
    # C22 = (a11+a12)(b22-b21) + a22(b11-b12)
    #
    # 7 bilinear forms:
    # f1 = (a11+a22)(b11+b22)
    # f2 = a22(b21+b22)
    # f3 = a11(b11-b21)
    # f4 = a12(b11+b22)
    # f5 = a21(a11+a22)
    # f6 = (a11+a12)(b22-b21)
    # f7 = a22(b11-b12)
    #
    # C11 = f1 - f2 + f3
    # C12 = f4
    # C21 = f5
    # C22 = f6 + f7
    
    # For each 2-column block, define the U, V, W contributions
    # U has 4 rows: index = i*m + j where i is row of A (0,1), j is col of A (0,1)
    # V has 22 rows: index = j*p + k where j is row of B (0,1), k is col of B
    # W has 22 rows: index = k*n + i where k is col of C (0..10), i is row of C (0,1)
    
    for b in range(num_blocks):
        # Columns 2b and 2b+1 of B
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # V row indices for this block:
        # Row 0 of B, cols k1, k2: indices k1, k2
        # Row 1 of B, cols k1, k2: indices p + k1, p + k2
        v_b11 = k1
        v_b12 = k2
        v_b21 = p + k1
        v_b22 = p + k2
        
        # W row indices for this block:
        # Col k1, row 0 of C: index k1*n + 0 = 2*k1
        # Col k1, row 1 of C: index k1*n + 1 = 2*k1 + 1
        # Col k2, row 0 of C: index k2*n + 0 = 2*k2
        # Col k2, row 1 of C: index k2*n + 1 = 2*k2 + 1
        w_c11 = 2 * k1
        w_c12 = 2 * k1 + 1
        w_c21 = 2 * k2
        w_c22 = 2 * k2 + 1
        
        # U row indices:
        # a11 -> index 0*m + 0 = 0
        # a12 -> index 0*m + 1 = 1
        # a21 -> index 1*m + 0 = 2
        # a22 -> index 1*m + 1 = 3
        u_a11 = 0
        u_a12 = 1
        u_a21 = 2
        u_a22 = 3
        
        # Define the 7 Strassen bilinear forms
        # Each form is (u_vec, v_vec, w_contribs)
        # u_vec: 4-vector for A
        # v_vec: 4-vector for B block [b11, b12, b21, b22]
        # w_contribs: dict mapping output indices to coefficients
        
        strassen_forms = [
            # f1 = (a11+a22)(b11+b22)
            ({u_a11: 1, u_a22: 1}, {v_b11: 1, v_b22: 1}, {w_c11: 1}),
            # f2 = a22(b21+b22)
            ({u_a22: 1}, {v_b21: 1, v_b22: 1}, {w_c11: -1}),
            # f3 = a11(b11-b21)
            ({u_a11: 1}, {v_b11: 1, v_b21: -1}, {w_c11: 1}),
            # f4 = a12(b11+b22)
            ({u_a12: 1}, {v_b11: 1, v_b22: 1}, {w_c12: 1}),
            # f5 = a21(a11+a22)
            ({u_a21: 1}, {u_a11: 0, u_a22: 0}, {w_c21: 1}),  # placeholder, fix below
            # f6 = (a11+a12)(b22-b21)
            ({u_a11: 1, u_a12: 1}, {v_b22: 1, v_b21: -1}, {w_c22: 1}),
            # f7 = a22(b11-b12)
            ({u_a22: 1}, {v_b11: 1, v_b12: -1}, {w_c22: 1}),
        ]
        
        # Fix f5: it's a21*(a11+a22) but that's not bilinear in A,B.
        # Actually f5 = a21 * (b11+b22)? No.
        # Let me re-derive Strassen properly.
        # 
        # Standard Strassen for 2x2:
        # p1 = (a11+a22)(b11+b22)
        # p2 = (a21+a22)b11
        # p3 = a11(b12-b22)
        # p4 = a22(b21-b11)
        # p5 = (a11+a12)b22
        # p6 = (a21-a11)(b11+b12)
        # p7 = (a12-a22)(b21+b22)
        # 
        # c11 = p1 + p4 - p5 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3 + p6
        #
        # Let's use this standard version.
        
        strassen_forms_v2 = [
            # p1 = (a11+a22)(b11+b22)
            ({u_a11: 1, u_a22: 1}, {v_b11: 1, v_b22: 1}, {w_c11: 1, w_c22: 1}),
            # p2 = (a21+a22)b11
            ({u_a21: 1, u_a22: 1}, {v_b11: 1}, {w_c21: 1, w_c22: -1}),
            # p3 = a11(b12-b22)
            ({u_a11: 1}, {v_b12: 1, v_b22: -1}, {w_c12: 1, w_c22: 1}),
            # p4 = a22(b21-b11)
            ({u_a22: 1}, {v_b21: 1, v_b11: -1}, {w_c11: 1, w_c21: 1}),
            # p5 = (a11+a12)b22
            ({u_a11: 1, u_a12: 1}, {v_b22: 1}, {w_c11: -1, w_c12: 1}),
            # p6 = (a21-a11)(b11+b12)
            ({u_a21: 1, u_a11: -1}, {v_b11: 1, v_b12: 1}, {w_c22: 1}),
            # p7 = (a12-a22)(b21+b22)
            ({u_a12: 1, u_a22: -1}, {v_b21: 1, v_b22: 1}, {w_c11: 1}),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_forms_v2:
            col = col_offset
            # Fill U
            for u_idx, val in u_coeffs.items():
                U[u_idx][col] += val
            # Fill V
            for v_idx, val in v_coeffs.items():
                V[v_idx][col] += val
            # Fill W
            for w_idx, val in w_coeffs.items():
                W[w_idx][col] += val
            col_offset += 1
    
    # Handle remaining 1 column (column 10)
    if rem_cols > 0:
        k = 2 * num_blocks  # = 10
        
        # V row indices: row 0 of B col k -> k, row 1 of B col k -> p+k
        v_b1 = k
        v_b2 = p + k
        
        # W row indices: col k row 0 -> 2k, col k row 1 -> 2k+1
        w_c1 = 2 * k
        w_c2 = 2 * k + 1
        
        # U row indices same as before
        u_a11 = 0
        u_a12 = 1
        u_a21 = 2
        u_a22 = 3
        
        # Schoolbook 2x2 by 2x1: 4 multiplications
        schoolbook_forms = [
            # a11 * b1
            ({u_a11: 1}, {v_b1: 1}, {w_c1: 1}),
            # a12 * b2
            ({u_a12: 1}, {v_b2: 1}, {w_c1: 1}),
            # a21 * b1
            ({u_a21: 1}, {v_b1: 1}, {w_c2: 1}),
            # a22 * b2
            ({u_a22: 1}, {v_b2: 1}, {w_c2: 1}),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in schoolbook_forms:
            col = col_offset
            for u_idx, val in u_coeffs.items():
                U[u_idx][col] += val
            for v_idx, val in v_coeffs.items():
                V[v_idx][col] += val
            for w_idx, val in w_coeffs.items():
                W[w_idx][col] += val
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
