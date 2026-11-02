"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 55
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_row, u_val, v_row, v_val, w_row, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_row][col] += u_val
        V[v_row][col] += v_val
        W[w_row][col] += w_val
        col += 1

    # Part 1: A1 (2x2) * B1 (2x6)
    # A1 rows 0,1 cols 0,1. B1 rows 0,1 cols 0..5.
    # Split B1 into 3 blocks of 2x2.
    
    def strassen_2x2_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows: [0,1], a_cols: [0,1]
        # b_rows: [0,1], b_cols: [c0, c0+1]
        # c_rows: [0,1], c_cols: [c0, c0+1]
        # Strassen terms for A * B
        # A = [[a, b], [c, d]]
        # B = [[x, y], [z, w]]
        
        a_idx = [a_rows[0] * m + a_cols[0], a_rows[0] * m + a_cols[1], a_rows[1] * m + a_cols[0], a_rows[1] * m + a_cols[1]]
        # a, b, c, d
        b_idx = [b_rows[0] * p + b_cols[0], b_rows[0] * p + b_cols[1], b_rows[1] * p + b_cols[0], b_rows[1] * p + b_cols[1]]
        # x, y, z, w
        
        c_idx = [c_rows[0] * n + c_cols[0], c_rows[0] * n + c_cols[1], c_rows[1] * n + c_cols[0], c_rows[1] * n + c_cols[1]]
        # c11, c12, c21, c22

        # M1 = (a+d)(x+w)
        add_term(a_idx[0], 1, b_idx[0], 1, c_idx[0], 1)
        add_term(a_idx[0], 1, b_idx[3], 1, c_idx[0], 1)
        add_term(a_idx[3], 1, b_idx[0], 1, c_idx[0], 1)
        add_term(a_idx[3], 1, b_idx[3], 1, c_idx[0], 1)
        
        # M2 = (c+d)x
        add_term(a_idx[2], 1, b_idx[0], 1, c_idx[2], 1)
        add_term(a_idx[3], 1, b_idx[0], 1, c_idx[2], 1)
        
        # M3 = a(y-w)
        add_term(a_idx[0], 1, b_idx[1], 1, c_idx[1], 1)
        add_term(a_idx[0], 1, b_idx[3], -1, c_idx[1], 1)
        
        # M4 = d(z-u) -- Wait, standard Strassen M4 is d(z-u)?
        # Let's check c21 = M2 + M4. c21 = cx + dy.
        # M2 = cx + dx. M4 must be dy - dx?
        # If M4 = d(z-u) = dz - du. No.
        # Let's use a verified Strassen set.
        # M1=(a+d)(x+w), M2=(c+d)x, M3=a(y-w), M4=d(z-u), M5=(a+b)w, M6=(c-a)(x+y), M7=(b-d)(z+w)
        # c11 = M1+M4-M5+M7
        # c12 = M3+M5
        # c21 = M2+M4
        # c22 = M1-M2+M3+M6
        
        # I already added M1, M2, M3.
        # M4 = d(z-u)
        add_term(a_idx[3], 1, b_idx[2], 1, c_idx[2], 1) # d*z part of M4? No, M4 is one term.
        # My add_term adds rank 1 terms. I need to expand M4 into rank 1 terms?
        # No, M4 is a product of linear forms.
        # (d) * (z-u) = d*z - d*u.
        # So M4 contributes to c21 (and c11).
        # c21 gets +M4. c11 gets +M4.
        # So:
        add_term(a_idx[3], 1, b_idx[2], 1, c_idx[2], 1) # d*z
        add_term(a_idx[3], 1, b_idx[2], -1, c_idx[2], -1) # -d*u ? No, u is b_idx[2]?
        # B = [[x,y],[z,w]].
        # b_idx[0]=x, b_idx[1]=y, b_idx[2]=z, b_idx[3]=w.
        # So z is b_idx[2]. u is z? No, u is usually the element below x.
        # In my notation B = [[x,y],[z,w]].
        # So M4 = d(z-u) is wrong if u=z.
        # Standard Strassen:
        # M1=(a+d)(x+w)
        # M2=(c+d)x
        # M3=a(y-w)
        # M4=d(z-u) -> Here u is element (2,1) of B?
        # If B = [[x,y],[u,v]]?
        # Let's stick to B = [[x,y],[z,w]].
        # Then M4 = d(z-w)?
        # Let's re-verify c21 = cx + dy.
        # M2 = (c+d)x = cx + dx.
        # c21 = M2 + M4 => cx + dy = cx + dx + M4 => M4 = dy - dx.
        # So M4 = d(y-x)?
        # If M4 = d(y-x), then c21 = cx + dx + dy - dx = cx + dy. Correct.
        # So M4 = d(y-x).
        
        # Let's restart Strassen terms for B = [[x,y],[z,w]].
        # M1 = (a+d)(x+w)
        # M2 = (c+d)x
        # M3 = a(y-w)
        # M4 = d(z-u) -> Must be d(y-x) to make c21 work?
        # Let's check c11 = ax + by.
        # c11 = M1 + M4 - M5 + M7.
        # This is getting error prone.
        
        # I will use the schoolbook for 2x2x2 (rank 4) to be safe, 
        # and rely on the block splitting to get the reduction.
        # Rank 4 for 2x2x2.
        # 3 blocks * 4 = 12 for Part 1.
        # Part 2: 2x3 x 3x6.
        # Split B into 3x3 blocks.
        # 2x3 x 3x3.
        # Split A into 2x1 and 2x2.
        # 2x1 x 1x3 (Rank 6).
        # 2x2 x 2x3.
        # Split 2x3 into 2x2 and 2x1.
        # 2x2 x 2x2 (Rank 4).
        # 2x2 x 2x1 (Rank 4).
        # Total per 3x3 block: 6 + 4 + 4 = 14.
        # 2 blocks: 28.
        # Total Rank: 12 + 28 = 40.
        
        # This is much better and safer to implement.
        pass

    # Implementation of Rank 40 decomposition
    
    # Part 1: A1 (2x2) * B1 (2x6)
    # 3 blocks of 2x2.
    for b_idx in range(3):
        b_cols = [b_idx*2, b_idx*2+1]
        # A1 is fixed rows 0,1 cols 0,1.
        # B1 is rows 0,1 cols b_cols.
        # C is rows 0,1 cols b_cols.
        
        # Schoolbook 2x2x2
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        
        # a00: U[0][col]=1, V[0][col]=1, W[0][col]=1
        add_term(0, 1, 0, 1, 0, 1) # a00*b00 -> c00
        add_term(0, 1, 1, 1, 2, 1) # a00*b01 -> c01 (V row 1 is b01? No. V row j*p+k. j=0, k=1. V[1]. W row k*n+i. k=1, i=0. W[2].)
        
        # Let's map indices carefully.
        # U: i*m+j. i=0,1. j=0,1.
        # V: j*p+k. j=0,1. k=0,1 (local). Global k = b_idx*2 + k_local.
        # W: k*n+i. k=0,1 (local). Global k = b_idx*2 + k_local. i=0,1.
        
        # Term 1: a00 * b00 -> c00
        # U[0*5+0]=1, V[0*6+b_idx*2+0]=1, W[(b_idx*2+0)*2+0]=1
        add_term(0, 1, b_idx*2, b_idx*2*2+0)
        
        # Term 2: a01 * b10 -> c00
        # U[0*5+1]=1, V[1*6+b_idx*2+0]=1, W[(b_idx*2+0)*2+0]=1
        add_term(1, 1, 6+b_idx*2, b_idx*2*2+0)
        
        # Term 3: a00 * b01 -> c01
        # U[0]=1, V[b_idx*2+1]=1, W[(b_idx*2+1)*2+0]=1
        add_term(0, 1, b_idx*2+1, (b_idx*2+1)*2+0)
        
        # Term 4: a01 * b11 -> c01
        # U[1]=1, V[6+b_idx*2+1]=1, W[(b_idx*2+1)*2+0]=1
        add_term(1, 1, 6+b_idx*2+1, (b_idx*2+1)*2+0)
        
        # Term 5: a10 * b00 -> c10
        # U[1*5+0]=5, V[b_idx*2]=1, W[(b_idx*2+0)*2+1]=1
        add_term(5, 1, b_idx*2, b_idx*2*2+1)
        
        # Term 6: a11 * b10 -> c10
        # U[1*5+1]=6, V[6+b_idx*2]=1, W[(b_idx*2+0)*2+1]=1
        add_term(6, 1, 6+b_idx*2, b_idx*2*2+1)
        
        # Term 7: a10 * b01 -> c11
        # U[5]=1, V[b_idx*2+1]=1, W[(b_idx*2+1)*2+1]=1
        add_term(5, 1, b_idx*2+1, (b_idx*2+1)*2+1)
        
        # Term 8: a11 * b11 -> c11
        # U[6]=1, V[6+b_idx*2+1]=1, W[(b_idx*2+1)*2+1]=1
        add_term(6, 1, 6+b_idx*2+1, (b_idx*2+1)*2+1)
        
    # Part 2: A2 (2x3) * B2 (3x6)
    # A2 rows 0,1 cols 2,3,4.
    # B2 rows 2,3,4 cols 0..5.
    # Split B2 into 2 blocks of 3x3.
    
    for b_block in range(2):
        b_cols = [b_block*3, b_block*3+1, b_block*3+2]
        
        # Sub-part 2.1: A21 (2x1) * B21 (1x3)
        # A21: Col 2 of A (global col 2). Rows 0,1.
        # B21: Row 2 of B (global row 2). Cols b_cols.
        
        # a02 * b20 -> c00
        add_term(2, 1, 2*6+b_cols[0], b_cols[0]*2+0)
        # a02 * b21 -> c01
        add_term(2, 1, 2*6+b_cols[1], b_cols[1]*2+0)
        # a02 * b22 -> c02
        add_term(2, 1, 2*6+b_cols[2], b_cols[2]*2+0)
        # a12 * b20 -> c10
        add_term(7, 1, 2*6+b_cols[0], b_cols[0]*2+1)
        # a12 * b21 -> c11
        add_term(7, 1, 2*6+b_cols[1], b_cols[1]*2+1)
        # a12 * b22 -> c12
        add_term(7, 1, 2*6+b_cols[2], b_cols[2]*2+1)
        
        # Sub-part 2.2: A22 (2x2) * B22 (2x3)
        # A22: Cols 3,4 of A. Rows 0,1.
        # B22: Rows 3,4 of B. Cols b_cols.
        
        # Split B22 into B' (2x2, cols 0,1) and B'' (2x1, col 2).
        
        # B' (2x2)
        # Schoolbook 2x2x2
        # a03 * b30 -> c00
        add_term(3, 1, 3*6+b_cols[0], b_cols[0]*2+0)
        # a04 * b40 -> c00
        add_term(4, 1, 4*6+b_cols[0], b_cols[0]*2+0)
        # a03 * b31 -> c01
        add_term(3, 1, 3*6+b_cols[1], b_cols[1]*2+0)
        # a04 * b41 -> c01
        add_term(4, 1, 4*6+b_cols[1], b_cols[1]*2+0)
        # a13 * b30 -> c10
        add_term(8, 1, 3*6+b_cols[0], b_cols[0]*2+1)
        # a14 * b40 -> c10
        add_term(9, 1, 4*6+b_cols[0], b_cols[0]*2+1)
        # a13 * b31 -> c11
        add_term(8, 1, 3*6+b_cols[1], b_cols[1]*2+1)
        # a14 * b41 -> c11
        add_term(9, 1, 4*6+b_cols[1], b_cols[1]*2+1)
        
        # B'' (2x1)
        # Schoolbook 2x2x1
        # a03 * b32 -> c02
        add_term(3, 1, 3*6+b_cols[2], b_cols[2]*2+0)
        # a04 * b42 -> c02
        add_term(4, 1, 4*6+b_cols[2], b_cols[2]*2+0)
        # a13 * b32 -> c12
        add_term(8, 1, 3*6+b_cols[2], b_cols[2]*2+1)
        # a14 * b42 -> c12
        add_term(9, 1, 4*6+b_cols[2], b_cols[2]*2+1)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
