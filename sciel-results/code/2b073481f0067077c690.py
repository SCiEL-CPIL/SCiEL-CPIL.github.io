"""Fast matrix multiplication for 3x4 * 4x8 using Strassen 2x2x2 blocks."""
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        U[u_idx][rank] = coeff
        V[v_idx][rank] = coeff
        W[w_idx][rank] = coeff
        rank += 1
        return rank

    # Block structure:
    # A is 3x4, B is 4x8
    # Split A into:
    #   A_top = A[0:2, :]  (2x4)
    #   A_bot = A[2:3, :]  (1x4)
    # Split B into 4 column blocks of 2 columns each:
    #   B_j = B[:, 2*j : 2*j+2] for j=0,1,2,3 (4x2)
    
    # For each j (0..3), we compute C[:, 2*j : 2*j+2] = A * B_j
    # C[:, 2*j : 2*j+2] is 3x2
    
    # For the top 2 rows of C:
    #   C[0:2, 2*j : 2*j+2] = A_top * B_j  (2x4 * 4x2 = 2x2)
    #   Use Strassen: 7 multiplications
    # For the bottom row of C:
    #   C[2, 2*j : 2*j+2] = A_bot * B_j  (1x4 * 4x2 = 1x2)
    #   Use 4 scalar multiplications
    
    # Strassen for 2x4 * 4x2:
    # A_top = [[a00, a01, a02, a03],
    #          [a10, a11, a12, a13]]
    # B_j = [[b00, b01],
    #        [b10, b11],
    #        [b20, b21],
    #        [b30, b31]]
    # 
    # Split A_top into 2x2 blocks:
    #   M00 = [[a00, a01], [a10, a11]]
    #   M01 = [[a02, a03], [a12, a13]]
    # 
    # B_j is 4x2, split into 2x2 blocks (row-wise):
    #   N0 = [[b00, b01], [b10, b11]]
    #   N1 = [[b20, b21], [b30, b31]]
    # 
    # C_top = M00 * N0 + M01 * N1
    # 
    # Apply Strassen to M00 * N0:
    #   P1 = (M00[0] + N1[1]) * (M00[1] + N0[0])  -- wait, this is for 2x2*2x2
    # 
    # Actually, let me just use the standard Strassen on 2x2 blocks.
    # M00 is 2x2, N0 is 2x2:
    #   P1 = (M00[0:1] + N0[1:2]) * (M00[1:2] + N0[0:1])
    #   P2 = M00[1:2] * (N0[0:1] + N0[1:2])
    #   P3 = (M00[0:1] + M00[1:2]) * N0[0:1]
    #   P4 = M00[0:1] * N0[1:2]
    #   P5 = (M00[0:1] + N0[1:2]) * (M00[1:2] - N0[0:1])
    #   P6 = (M00[1:2] - M00[0:1]) * (N0[0:1] + N0[1:2])
    #   P7 = (M00[0:1] - M00[1:2]) * N0[1:2]
    # 
    # But this is getting complicated with the indexing. Let me just enumerate the 7 Strassen products.
    
    # Let's use a simpler approach: just enumerate all 7 Strassen terms explicitly.
    
    # For M00 * N0 (2x2 * 2x2):
    # M00 = [[a00, a01], [a10, a11]]
    # N0 = [[b00, b01], [b10, b11]]
    # 
    # P1 = (a00 + a11) * (b00 + b11)
    # P2 = (a10 + a11) * b00
    # P3 = a00 * (b01 + b11)
    # P4 = a11 * (b10 - b00)
    # P5 = (a00 + a01) * b10
    # P6 = (a10 - a00) * (b00 + b01)
    # P7 = (a01 - a11) * (b10 + b11)
    
    # Result:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 - P6
    
    # For M01 * N1 (2x2 * 2x2):
    # M01 = [[a02, a03], [a12, a13]]
    # N1 = [[b20, b21], [b30, b31]]
    # 
    # P8 = (a02 + a13) * (b20 + b31)
    # P9 = (a12 + a13) * b20
    # P10 = a02 * (b21 + b31)
    # P11 = a13 * (b30 - b20)
    # P12 = (a02 + a03) * b30
    # P13 = (a12 - a02) * (b20 + b21)
    # P14 = (a03 - a13) * (b30 + b31)
    
    # Result:
    # C00 += P8 + P11 - P12 + P14
    # C01 += P10 + P12
    # C10 += P9 + P11
    # C11 += P8 - P9 + P10 - P13
    
    # So for each j, we have 14 Strassen products for the top 2x2 block.
    # Plus 4 scalar products for the bottom 1x2 block.
    # Total per j: 18 products.
    # Total: 18 * 4 = 72.
    
    # Wait, that's less than 88. Let me re-check.
    
    # Actually, I think I'm overcomplicating this. Let me just use the schoolbook for the bottom row
    # and Strassen for the top.
    
    # Let me restart with a cleaner approach.
    
    # Reset
    rank = 0
    U = [[0] * 88 for _ in range(n * m)]
    V = [[0] * 88 for _ in range(m * p)]
    W = [[0] * 88 for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal rank
        U[u_idx][rank] = coeff
        V[v_idx][rank] = coeff
        W[w_idx][rank] = coeff
        rank += 1
    
    # For each column block j (0..3):
    for j in range(4):
        col0 = 2 * j
        col1 = 2 * j + 1
        
        # Top 2x2 block: C[0:2, col0:col1+1] = A[0:2, :] * B[:, col0:col1+1]
        # A_top = A[0:2, :] (2x4)
        # B_j = B[:, col0:col1+1] (4x2)
        
        # Split A_top into left 2x2 and right 2x2:
        # M00 = A[0:2, 0:2]
        # M01 = A[0:2, 2:4]
        
        # Split B_j into top 2x2 and bottom 2x2:
        # N0 = B[0:2, col0:col1+1]
        # N1 = B[2:4, col0:col1+1]
        
        # C_top = M00 * N0 + M01 * N1
        
        # Strassen for M00 * N0:
        # a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
        # b00=B[0,col0], b01=B[0,col1], b10=B[1,col0], b11=B[1,col1]
        
        a00, a01 = 0*4+0, 0*4+1
        a10, a11 = 1*4+0, 1*4+1
        b00, b01 = 0*8+col0, 0*8+col1
        b10, b11 = 1*8+col0, 1*8+col1
        
        # P1 = (a00 + a11) * (b00 + b11)
        # This requires: U has a00 and a11, V has b00 and b11
        # But we can't directly add them in U and V without creating separate terms.
        # So we need to create 7 separate terms for the 7 Strassen products.
        
        # P1: (a00 + a11) * (b00 + b11)
        # We need: U[a00]=1, U[a11]=1, V[b00]=1, V[b11]=1
        # But this is a single term with combined U and V.
        # The decomposition is: sum_r U[i,r] * V[j,r] * W[k,r]
        # So for P1, we have one term where U has both a00 and a11 set to 1,
        # and V has both b00 and b11 set to 1.
        
        # This is not how it works. Each term r is a rank-1 component.
        # U[i,r], V[j,r], W[k,r] are scalars.
        # The product is U[i,r] * V[j,r] for each (i,j), and then multiplied by W[k,r].
        
        # So for P1 = (a00 + a11) * (b00 + b11):
        # We need U[0, r] = 1, U[1, r] = 1 (for a00 and a11)
        # V[0, r] = 1, V[1, r] = 1 (for b00 and b11)
        # But then U[0,r]*V[0,r] = 1*1 = 1, which gives a00*b00
        # U[0,r]*V[1,r] = 1*1 = 1, which gives a00*b11
        # U[1,r]*V[0,r] = 1*1 = 1, which gives a11*b00
        # U[1,r]*V[1,r] = 1*1 = 1, which gives a11*b11
        # So this single term gives a00*b00 + a00*b11 + a11*b00 + a11*b11
        # Which is exactly (a00 + a11)*(b00 + b11).
        
        # So P1 is a single term!
        
        # Similarly for the other 6 Strassen products.
        
        # P1: (a00 + a11) * (b00 + b11)
        U[a00][rank] = 1
        U[a11][rank] = 1
        V[b00][rank] = 1
        V[b11][rank] = 1
        # Result goes to C[0,0] and C[1,1] (relative to the 2x2 block)
        # C[0,0] += P1, C[1,1] += P1
        W[col0*n + 0][rank] = 1  # C[0, col0]
        W[col1*n + 1][rank] = 1  # C[1, col1]
        rank += 1
        
        # P2: (a10 + a11) * b00
        U[a10][rank] = 1
        U[a11][rank] = 1
        V[b00][rank] = 1
        W[col0*n + 1][rank] = 1  # C[1, col0]
        rank += 1
        
        # P3: a00 * (b01 + b11)
        U[a00][rank] = 1
        V[b01][rank] = 1
        V[b11][rank] = 1
        W[col1*n + 0][rank] = 1  # C[0, col1]
        rank += 1
        
        # P4: a11 * (b10 - b00)
        U[a11][rank] = 1
        V[b10][rank] = 1
        V[b00][rank] = -1
        W[col0*n + 1][rank] = 1  # C[1, col0]
        W[col1*n + 1][rank] = -1  # C[1, col1]
        rank += 1
        
        # P5: (a00 + a01) * b10
        U[a00][rank] = 1
        U[a01][rank] = 1
        V[b10][rank] = 1
        W[col0*n + 0][rank] = -1  # C[0, col0]
        W[col1*n + 0][rank] = 1  # C[0, col1]
        rank += 1
        
        # P6: (a10 - a00) * (b00 + b01)
        U[a10][rank] = 1
        U[a00][rank] = -1
        V[b00][rank] = 1
        V[b01][rank] = 1
        W[col1*n + 1][rank] = -1  # C[1, col1]
        rank += 1
        
        # P7: (a01 - a11) * (b10 + b11)
        U[a01][rank] = 1
        U[a11][rank] = -1
        V[b10][rank] = 1
        V[b11][rank] = 1
        W[col0*n + 0][rank] = 1  # C[0, col0]
        W[col1*n + 0][rank] = -1  # C[0, col1]
        rank += 1
        
        # Now for M01 * N1:
        # M01 = A[0:2, 2:4]
        # N1 = B[2:4, col0:col1+1]
        
        a02, a03 = 0*4+2, 0*4+3
        a12, a13 = 1*4+2, 1*4+3
        b20, b21 = 2*8+col0, 2*8+col1
        b30, b31 = 3*8+col0, 3*8+col1
        
        # P8: (a02 + a13) * (b20 + b31)
        U[a02][rank] = 1
        U[a13][rank] = 1
        V[b20][rank] = 1
        V[b31][rank] = 1
        W[col0*n + 0][rank] = 1
        W[col1*n + 1][rank] = 1
        rank += 1
        
        # P9: (a12 + a13) * b20
        U[a12][rank] = 1
        U[a13][rank] = 1
        V[b20][rank] = 1
        W[col0*n + 1][rank] = 1
        rank += 1
        
        # P10: a02 * (b21 + b31)
        U[a02][rank] = 1
        V[b21][rank] = 1
        V[b31][rank] = 1
        W[col1*n + 0][rank] = 1
        rank += 1
        
        # P11: a13 * (b30 - b20)
        U[a13][rank] = 1
        V[b30][rank] = 1
        V[b20][rank] = -1
        W[col0*n + 1][rank] = 1
        W[col1*n + 1][rank] = -1
        rank += 1
        
        # P12: (a02 + a03) * b30
        U[a02][rank] = 1
        U[a03][rank] = 1
        V[b30][rank] = 1
        W[col0*n + 0][rank] = -1
        W[col1*n + 0][rank] = 1
        rank += 1
        
        # P13: (a12 - a02) * (b20 + b21)
        U[a12][rank] = 1
        U[a02][rank] = -1
        V[b20][rank] = 1
        V[b21][rank] = 1
        W[col1*n + 1][rank] = -1
        rank += 1
        
        # P14: (a03 - a13) * (b30 + b31)
        U[a03][rank] = 1
        U[a13][rank] = -1
        V[b30][rank] = 1
        V[b31][rank] = 1
        W[col0*n + 0][rank] = 1
        W[col1*n + 0][rank] = -1
        rank += 1
        
        # Now for the bottom row: C[2, col0:col1+1] = A[2, :] * B[:, col0:col1+1]
        # A[2, :] = [A[2,0], A[2,1], A[2,2], A[2,3]]
        # B[:, col0:col1+1] = [[B[0,col0], B[0,col1]],
        #                      [B[1,col0], B[1,col1]],
        #                      [B[2,col0], B[2,col1]],
        #                      [B[3,col0], B[3,col1]]]
        
        # C[2, col0] = A[2,0]*B[0,col0] + A[2,1]*B[1,col0] + A[2,2]*B[2,col0] + A[2,3]*B[3,col0]
        # C[2, col1] = A[2,0]*B[0,col1] + A[2,1]*B[1,col1] + A[2,2]*B[2,col1] + A[2,3]*B[3,col1]
        
        # 4 scalar multiplications:
        # A[2,0] * B[0,col0]
        U[2*4+0][rank] = 1
        V[0*8+col0][rank] = 1
        W[col0*n + 2][rank] = 1
        rank += 1
        
        # A[2,1] * B[1,col0]
        U[2*4+1][rank] = 1
        V[1*8+col0][rank] = 1
        W[col0*n + 2][rank] = 1
        rank += 1
        
        # A[2,2] * B[2,col0]
        U[2*4+2][rank] = 1
        V[2*8+col0][rank] = 1
        W[col0*n + 2][rank] = 1
        rank += 1
        
        # A[2,3] * B[3,col0]
        U[2*4+3][rank] = 1
        V[3*8+col0][rank] = 1
        W[col0*n + 2][rank] = 1
        rank += 1
        
        # A[2,0] * B[0,col1]
        U[2*4+0][rank] = 1
        V[0*8+col1][rank] = 1
        W[col1*n + 2][rank] = 1
        rank += 1
        
        # A[2,1] * B[1,col1]
        U[2*4+1][rank] = 1
        V[1*8+col1][rank] = 1
        W[col1*n + 2][rank] = 1
        rank += 1
        
        # A[2,2] * B[2,col1]
        U[2*4+2][rank] = 1
        V[2*8+col1][rank] = 1
        W[col1*n + 2][rank] = 1
        rank += 1
        
        # A[2,3] * B[3,col1]
        U[2*4+3][rank] = 1
        V[3*8+col1][rank] = 1
        W[col1*n + 2][rank] = 1
        rank += 1
    
    # Total rank should be 88
    assert rank == 88, f"Expected rank 88, got {rank}"
    
    return {"U": U, "V": V, "W": W}
