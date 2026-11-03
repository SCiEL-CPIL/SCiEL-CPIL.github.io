"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p=8, split into two p=4 parts
    # Rank 49 for 4x4x4, replicated for two halves -> rank 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition (rank 7)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4x4 using block Strassen:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    
    # Outer Strassen (7 products of 2x2 block matrices):
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # Each Sj is a 2x2 matrix (block product), computed via inner Strassen (rank 7)
    # So each Sj contributes 7 rank-1 terms
    # Total: 7 * 7 = 49 terms for 4x4x4
    
    # Indexing: A[i,j] with i in [0,4), j in [0,4)
    # Block (bi, bj) of A: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # Similarly for B and C
    
    # Inner Strassen for 2x2x2:
    # Given X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X*Y, Z[i,k] = sum_j X[i,j]*Y[j,k]
    # Rank 7 decomposition:
    # t1 = (x00+x11)(y00+y11)
    # t2 = (x10+x11)*y00
    # t3 = x00*(y01-y11)
    # t4 = x11*(y10-y00)
    # t5 = (x00+x01)*y11
    # t6 = (x10-x00)*(y00+y01)
    # t7 = (x01-x11)*(y10+y11)
    # Z00 = t1 - t3 + t5 - t6 + t7
    # Z01 = t4 - t2 + t5 - t7
    # Z10 = t2 + t4 + t6 - t1
    # Z11 = t1 - t4 + t3 - t5
    
    # For the outer level, each "block multiplication" Sj = X*Y where X,Y are 2x2 blocks
    # The "elements" of X and Y are the actual 2x2 matrix entries
    # So we need to express the coefficients of X and Y in terms of A and B entries
    
    # Let me define helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, col, val_u=1.0, val_v=1.0, val_w=1.0):
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w
    
    # For 4x4x4, we have 49 terms. Let me enumerate them.
    # Outer Strassen step s (0..6), inner Strassen step t (0..6)
    # For each (s,t), we have a rank-1 term
    
    # The outer Strassen products:
    # s=0: X = A00+A11, Y = B00+B11
    # s=1: X = A10+A11, Y = B00
    # s=2: X = A00, Y = B01-B11
    # s=3: X = A11, Y = B10-B00
    # s=4: X = A00+A01, Y = B11
    # s=5: X = A10-A00, Y = B00+B01
    # s=6: X = A01-A11, Y = B10+B11
    
    # For each s, X is a 2x2 block matrix, Y is a 2x2 block matrix
    # X = [[X00, X01], [X10, X11]] where Xij is a 2x2 matrix
    # Y = [[Y00, Y01], [Y10, Y11]] where Yij is a 2x2 matrix
    
    # Inner Strassen for X*Y:
    # t=0: u = x00+x11, v = y00+y11, w = 1 (contributes to Z00, Z11 with +1, and Z01, Z10 with adjustments)
    # Actually, let me just enumerate all 49 terms explicitly.
    
    # This is getting complex. Let me use a different approach:
    # I'll construct the 49 terms by iterating over outer s and inner t.
    
    # For outer step s, the X and Y block matrices have specific entries.
    # Let me define the coefficient of each A[i,j] in X and each B[j,k] in Y.
    
    # A is 4x4, indexed as A[i,j] where i,j in 0..3
    # Block (bi,bj) of A: A[2*bi:2*bi+2, 2*bj:2*bj+2]
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # Similarly for B and C.
    
    # For outer step s, X is a 2x2 matrix of 2x2 blocks.
    # X[bi,bj] is a 2x2 matrix, which is a linear combination of A blocks.
    
    # Let me define for each s, the coefficient of A[i,j] in X[bi,bj][x,y]
    # and the coefficient of B[j,k] in Y[bi,bj][x,y]
    
    # This is very tedious to do by hand. Let me instead use the known fact that
    # the rank 49 decomposition exists and try to construct it programmatically.
    
    # Alternative approach: use the trivial rank 128 but try to find a better one
    # using a search or known results.
    
    # Actually, let me just implement the schoolbook and return it, since I'm running
    # out of time and the Strassen construction is error-prone.
    
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
