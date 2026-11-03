"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 84 decomposition using Strassen's algorithm on 2x2 blocks
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen's 7 products for 2x2 matrices
    # P1 = (m00+m11)(n00+n11)
    # P2 = (m10+m11)n00
    # P3 = m00(n01-n11)
    # P4 = m11(n10-n00)
    # P5 = (m00+m01)n11
    # P6 = (m10-m00)(n00+n01)
    # P7 = (m01-m11)(n10+n11)
    
    # Results:
    # p00 = P1 + P4 - P5 + P7
    # p01 = P3 + P5
    # p10 = P2 + P4
    # p11 = P1 - P2 + P3 + P6
    
    # We have 12 such 2x2 multiplications
    # Let's define the mapping for each of the 12 multiplications
    # Each multiplication j has a base column offset j*7
    # The 7 Strassen products are at columns base+0 to base+6
    
    # For each multiplication j, we need to specify:
    # - Which 2x2 block of A is M
    # - Which 2x2 block of B is N
    # - Where the result P goes in C
    
    # A blocks:
    # A00: rows 0,1; cols 0,1
    # A01: rows 0,1; cols 2,3
    # A10: rows 2,3; cols 0,1
    # A11: rows 2,3; cols 2,3
    
    # B blocks (for each k in 0,1,2):
    # Bk00: rows 0,1; cols 2k, 2k+1
    # Bk10: rows 2,3; cols 2k, 2k+1
    
    # The 12 multiplications:
    # j=0: A00 * B000 -> C00 (top-left 2x2 of C0)
    # j=1: A01 * B010 -> C00 (added to top-left 2x2 of C0)
    # j=2: A10 * B000 -> C10 (top-left 2x2 of C1)
    # j=3: A11 * B010 -> C10 (added to top-left 2x2 of C1)
    # j=4: A00 * B100 -> C01 (top-left 2x2 of C1? No, C is 4x6, so C01 is cols 2,3 of top half)
    # Wait, C is 4x6.
    # C = [C00 C01 C02]
    #     [C10 C11 C12]
    # C00 is 2x2, C01 is 2x2, C02 is 2x2
    # C10 is 2x2, C11 is 2x2, C12 is 2x2
    
    # j=0: A00 * B000 -> C00
    # j=1: A01 * B010 -> C00
    # j=2: A10 * B000 -> C10
    # j=3: A11 * B010 -> C10
    # j=4: A00 * B100 -> C01
    # j=5: A01 * B110 -> C01
    # j=6: A10 * B100 -> C11
    # j=7: A11 * B110 -> C11
    # j=8: A00 * B200 -> C02
    # j=9: A01 * B210 -> C02
    # j=10: A10 * B200 -> C12
    # j=11: A11 * B210 -> C12
    
    # For each j, we define the M block and N block
    # M block: (row_offset, col_offset)
    # N block: (row_offset, col_offset)
    # Result block: (row_offset, col_offset)
    
    # A is 4x4, so row/col offsets are 0 or 2
    # B is 4x6, so row offsets are 0 or 2, col offsets are 0, 2, 4
    
    # Let's define the 12 cases
    cases = [
        # (M_row, M_col, N_row, N_col, C_row, C_col)
        (0, 0, 0, 0, 0, 0), # j=0: A00 * B000 -> C00
        (0, 2, 2, 0, 0, 0), # j=1: A01 * B010 -> C00
        (2, 0, 0, 0, 2, 0), # j=2: A10 * B000 -> C10
        (2, 2, 2, 0, 2, 0), # j=3: A11 * B010 -> C10
        (0, 0, 0, 2, 0, 2), # j=4: A00 * B100 -> C01
        (0, 2, 2, 2, 0, 2), # j=5: A01 * B110 -> C01
        (2, 0, 0, 2, 2, 2), # j=6: A10 * B100 -> C11
        (2, 2, 2, 2, 2, 2), # j=7: A11 * B110 -> C11
        (0, 0, 0, 4, 0, 4), # j=8: A00 * B200 -> C02
        (0, 2, 2, 4, 0, 4), # j=9: A01 * B210 -> C02
        (2, 0, 0, 4, 2, 4), # j=10: A10 * B200 -> C12
        (2, 2, 2, 4, 2, 4), # j=11: A11 * B210 -> C12
    ]
    
    for j, (mr, mc, nr, nc, cr, cc) in enumerate(cases):
        base = j * 7
        # Strassen products P1..P7 are at columns base..base+6
        # M is 2x2 at (mr, mc)
        # N is 2x2 at (nr, nc)
        # P is 2x2 at (cr, cc)
        
        # P1 = (m00+m11)(n00+n11)
        # m00 is at (mr, mc), m11 is at (mr+1, mc+1)
        # n00 is at (nr, nc), n11 is at (nr+1, nc+1)
        # U for P1:
        U[mr * m + mc][base + 0] = 1
        U[(mr+1) * m + (mc+1)][base + 0] = 1
        V[nr * p + nc][base + 0] = 1
        V[(nr+1) * p + (nc+1)][base + 0] = 1
        # W for P1:
        # p00 += P1, p11 += P1
        W[(nc) * n + (mr)][base + 0] = 1 # Wait, W index is k*n+i
        # C is 4x6, so C[r, c]. r in 0..3, c in 0..5.
        # W index is c*n + r? No, W[k*n+i, r] where k is col of B, i is row of A?
        # The task says: W[k*n+i,r]
        # Let's check the convention.
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # C[i, k] = sum_r U[i, r] V[k, r] W[r, ?]
        # Actually, the tensor is T[i, j, k] = C[i, k] where C = A B.
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # So W index is k*n + i.
        # For P1, it contributes to p00 and p11 of the 2x2 block at (cr, cc).
        # p00 is at C[cr, cc]
        # p11 is at C[cr+1, cc+1]
        W[cc * n + cr][base + 0] = 1
        W[(cc+1) * n + (cr+1)][base + 0] = 1
        
        # P2 = (m10+m11)n00
        U[(mr+1) * m + mc][base + 1] = 1
        U[(mr+1) * m + (mc+1)][base + 1] = 1
        V[nr * p + nc][base + 1] = 1
        # p10 += P2, p11 -= P2
        W[cc * n + (cr+1)][base + 1] = 1
        W[(cc+1) * n + (cr+1)][base + 1] = -1
        
        # P3 = m00(n01-n11)
        U[mr * m + mc][base + 2] = 1
        V[nr * p + (nc+1)][base + 2] = 1
        V[(nr+1) * p + (nc+1)][base + 2] = -1
        # p01 += P3, p11 += P3
        W[(cc+1) * n + cr][base + 2] = 1
        W[(cc+1) * n + (cr+1)][base + 2] = 1
        
        # P4 = m11(n10-n00)
        U[(mr+1) * m + (mc+1)][base + 3] = 1
        V[(nr+1) * p + nc][base + 3] = 1
        V[nr * p + nc][base + 3] = -1
        # p00 += P4, p10 += P4
        W[cc * n + cr][base + 3] = 1
        W[cc * n + (cr+1)][base + 3] = 1
        
        # P5 = (m00+m01)n11
        U[mr * m + mc][base + 4] = 1
        U[mr * m + (mc+1)][base + 4] = 1
        V[(nr+1) * p + (nc+1)][base + 4] = 1
        # p00 -= P5, p01 += P5
        W[cc * n + cr][base + 4] = -1
        W[(cc+1) * n + cr][base + 4] = 1
        
        # P6 = (m10-m00)(n00+n01)
        U[(mr+1) * m + mc][base + 5] = 1
        U[mr * m + mc][base + 5] = -1
        V[nr * p + nc][base + 5] = 1
        V[nr * p + (nc+1)][base + 5] = 1
        # p11 += P6
        W[(cc+1) * n + (cr+1)][base + 5] = 1
        
        # P7 = (m01-m11)(n10+n11)
        U[mr * m + (mc+1)][base + 6] = 1
        U[(mr+1) * m + (mc+1)][base + 6] = -1
        V[(nr+1) * p + nc][base + 6] = 1
        V[(nr+1) * p + (nc+1)][base + 6] = 1
        # p00 += P7
        W[cc * n + cr][base + 6] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
