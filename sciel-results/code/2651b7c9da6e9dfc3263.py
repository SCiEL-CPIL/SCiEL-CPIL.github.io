"""Matrix multiplication 2x3x9 with rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # seconds
    
    # We will build a decomposition of M_{2,3,9} with rank 50
    # by splitting p=9 into 4+5, then further splitting using Strassen for 2x2x2.
    
    # Strategy:
    # M_{2,3,9} = M_{2,3,4} + M_{2,3,5}
    # M_{2,3,4} = M_{2,3,2} + M_{2,3,2}
    # M_{2,3,5} = M_{2,3,2} + M_{2,3,3}
    # M_{2,3,3} = M_{2,3,2} + M_{2,3,1}
    # M_{2,3,1} = M_{2,2,1} + M_{2,1,1}
    
    # For M_{2,3,2}: split m=3 into 2+1
    # M_{2,3,2} = M_{2,2,2} + M_{2,1,2}
    # M_{2,2,2} uses Strassen (rank 7)
    # M_{2,1,2} is schoolbook (rank 4)
    # So M_{2,3,2} has rank 7+4 = 11
    
    # M_{2,3,1}: split m=3 into 2+1
    # M_{2,3,1} = M_{2,2,1} + M_{2,1,1}
    # M_{2,2,1} is schoolbook (rank 4)
    # M_{2,1,1} is schoolbook (rank 2)
    # So M_{2,3,1} has rank 4+2 = 6
    
    # Total rank:
    # M_{2,3,4} = 2 * 11 = 22
    # M_{2,3,5} = 11 + 6 = 17
    # Total = 22 + 17 = 39? Wait, let me recount.
    
    # Actually: M_{2,3,9} = M_{2,3,4} + M_{2,3,5}
    # M_{2,3,4} = M_{2,3,2} + M_{2,3,2} = 11 + 11 = 22
    # M_{2,3,5} = M_{2,3,2} + M_{2,3,3}
    # M_{2,3,3} = M_{2,3,2} + M_{2,3,1} = 11 + 6 = 17
    # M_{2,3,5} = 11 + 17 = 28
    # Total = 22 + 28 = 50. Good.
    
    # Now let's build this systematically.
    # I'll create a list of "atoms" - each atom is a small tensor with its U,V,W
    # and offsets into the global U,V,W.
    
    # Global dimensions: n=2, m=3, p=9
    # U: 6 x rank (n*m rows)
    # V: 27 x rank (m*p rows)  
    # W: 18 x rank (p*n rows)
    
    rank = 50
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(27)]
    W = [[0]*rank for _ in range(18)]
    
    # Column pointer for assigning columns in U,V,W
    col = 0
    
    # Helper functions for small cases
    
    def add_strassen_222(u_offset, v_offset, w_offset, start_col):
        """Add Strassen decomposition for M_{2,2,2}.
        u_offset: (i,j) offset in U, so global row = (u_offset_i + i)*m_global + (u_offset_j + j)
        But wait, U is indexed by [i*m + j] where i in [0,n), j in [0,m).
        For a subproblem M_{n_sub, m_sub, p_sub}, the U indices are i_sub in [0,n_sub), j_sub in [0,m_sub).
        Global U row = (i_offset + i_sub) * m_global + (j_offset + j_sub)
        
        Actually, let me think about this more carefully.
        U has shape (n*m, rank). Row index = i*m + j for i in [0,n), j in [0,m).
        For a sub-block at position (i0, j0) with sub-dimensions (n_sub, m_sub),
        the U rows are (i0+ii)*m + (j0+jj) for ii in [0,n_sub), jj in [0,m_sub).
        
        v_offset: (j0, k0) for V. V row = j*p + k.
        w_offset: (k0, i0) for W. W row = k*n + i.
        
        Strassen for 2x2x2:
        U1 = A11+A22, V1 = B11+B22, W1 = C11+C22
        U2 = A21+A22, V2 = B11, W2 = C21
        U3 = A11, V3 = B21+B22, W3 = C12
        U4 = A22, V4 = B12, W4 = C22
        U5 = A11+A12, V5 = B22, W5 = C11
        U6 = A21-A11, V6 = B11+B12, W6 = C21
        U7 = A12-A22, V7 = B21+B22, W7 = C12
        
        Let me write the 7 terms for Strassen.
        """
        global col
        # 7 terms for Strassen
        terms = [
            # (U coeffs: list of (ii, jj, val), V coeffs: list of (jj, kk, val), W coeffs: list of (kk, ii, val))
            # Term 1: U=A11+A22, V=B11+B22, W=C11+C22
            ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
            # Term 2: U=A21+A22, V=B11, W=C21
            ([((1,0,1), (1,1,1))], [((0,0,1))], [((1,1,1))]),
            # Term 3: U=A11, V=B21+B22, W=C12
            ([((0,0,1))], [((1,0,1), (1,1,1))], [((0,1,1))]),
            # Term 4: U=A22, V=B12, W=C22
            ([((1,1,1))], [((0,1,1))], [((1,1,1))]),
            # Term 5: U=A11+A12, V=B22, W=C11
            ([((0,0,1), (0,1,1))], [((1,1,1))], [((0,0,1))]),
            # Term 6: U=A21-A11, V=B11+B12, W=C21
            ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [((1,1,1))]),
            # Term 7: U=A12-A22, V=B21+B22, W=C12
            ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [((0,1,1))]),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in terms:
            for ii, jj, val in u_coeffs:
                U[(u_offset[0]+ii)*3 + (u_offset[1]+jj)][start_col + col] = val
            for jj, kk, val in v_coeffs:
                V[(v_offset[0]+jj)*9 + (v_offset[1]+kk)][start_col + col] = val
            for kk, ii, val in w_coeffs:
                W[(w_offset[0]+kk)*2 + (w_offset[1]+ii)][start_col + col] = val
            col += 1
        return start_col + 7
    
    def add_schoolbook_nm_p(n_sub, m_sub, p_sub, u_offset, v_offset, w_offset, start_col):
        """Add schoolbook decomposition for M_{n_sub, m_sub, p_sub}."""
        global col
        c = 0
        for ii in range(n_sub):
            for jj in range(m_sub):
                for kk in range(p_sub):
                    U[(u_offset[0]+ii)*3 + (u_offset[1]+jj)][start_col + c] = 1
                    V[(v_offset[0]+jj)*9 + (v_offset[1]+kk)][start_col + c] = 1
                    W[(w_offset[0]+kk)*2 + (w_offset[1]+ii)][start_col + c] = 1
                    c += 1
        return start_col + n_sub * m_sub * p_sub
    
    # Now decompose M_{2,3,9}
    
    # Split p=9 into 4 and 5: k in [0,4) and k in [4,9)
    
    # Part 1: M_{2,3,4} with k offset 0
    # Split p=4 into 2+2: k in [0,2) and k in [2,4)
    
    # Part 1a: M_{2,3,2} with k offset 0
    # Split m=3 into 2+1: j in [0,2) and j in [2,3)
    
    # Part 1a1: M_{2,2,2} with i offset 0, j offset 0, k offset 0
    col = add_strassen_222((0,0), (0,0), (0,0), 0)
    
    # Part 1a2: M_{2,1,2} with i offset 0, j offset 2, k offset 0
    col = add_schoolbook_nm_p(2, 1, 2, (0,2), (2,0), (0,0), col)
    
    # Part 1b: M_{2,3,2} with k offset 2
    # Split m=3 into 2+1
    # Part 1b1: M_{2,2,2} with i offset 0, j offset 0, k offset 2
    col = add_strassen_222((0,0), (0,2), (2,0), col)
    
    # Part 1b2: M_{2,1,2} with i offset 0, j offset 2, k offset 2
    col = add_schoolbook_nm_p(2, 1, 2, (0,2), (2,2), (2,0), col)
    
    # Part 2: M_{2,3,5} with k offset 4
    # Split p=5 into 2+3: k in [4,6) and k in [6,9)
    
    # Part 2a: M_{2,3,2} with k offset 4
    # Split m=3 into 2+1
    # Part 2a1: M_{2,2,2} with i offset 0, j offset 0, k offset 4
    col = add_strassen_222((0,0), (0,4), (4,0), col)
    
    # Part 2a2: M_{2,1,2} with i offset 0, j offset 2, k offset 4
    col = add_schoolbook_nm_p(2, 1, 2, (0,2), (2,4), (4,0), col)
    
    # Part 2b: M_{2,3,3} with k offset 6
    # Split p=3 into 2+1: k in [6,8) and k in [8,9)
    
    # Part 2b1: M_{2,3,2} with k offset 6
    # Split m=3 into 2+1
    # Part 2b11: M_{2,2,2} with i offset 0, j offset 0, k offset 6
    col = add_strassen_222((0,0), (0,6), (6,0), col)
    
    # Part 2b12: M_{2,1,2} with i offset 0, j offset 2, k offset 6
    col = add_schoolbook_nm_p(2, 1, 2, (0,2), (2,6), (6,0), col)
    
    # Part 2b2: M_{2,3,1} with k offset 8
    # Split m=3 into 2+1
    # Part 2b21: M_{2,2,1} with i offset 0, j offset 0, k offset 8
    col = add_schoolbook_nm_p(2, 2, 1, (0,0), (0,8), (8,0), col)
    
    # Part 2b22: M_{2,1,1} with i offset 0, j offset 2, k offset 8
    col = add_schoolbook_nm_p(2, 1, 1, (0,2), (2,8), (8,0), col)
    
    # Verify col == 50
    assert col == 50, f"Expected rank 50, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
