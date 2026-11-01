"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    # Construct a rank-28 decomposition for 2x4 * 4x4 using Strassen's algorithm
    # applied to the four 2x2 blocks of B.
    #
    # A is 2x4: split columns into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # B is 4x4: split into 2x2 blocks B11, B12, B21, B22
    # Result C is 2x4: C1 = A1*B1 + A2*B2 (2x2), C2 = A1*B2 + A2*B3 (2x2)
    #   where B1 = [B11; B21] (4x2), B2 = [B12; B22] (4x2)
    #
    # For each 2x2 matrix multiplication we use Strassen (rank 7).
    # There are 4 such multiplications:
    #   M1: A1 * B11, M2: A1 * B12, M3: A2 * B21, M4: A2 * B22
    # Total rank = 4 * 7 = 28
    #
    # Strassen's algorithm for 2x2:
    #   Given P = A * B where A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    #   M1 = (a+b)(e+f)
    #   M2 = (c+d)(e)
    #   M3 = (a)(f-g)
    #   M4 = (d)(h-g)
    #   M5 = (a+d)(h)
    #   M6 = (b-c)(e+g)
    #   M7 = (b-d)(f+h)
    #   P11 = M1 - M4 + M5
    #   P12 = M3 + M5
    #   P21 = M2 + M4
    #   P22 = M1 - M2 + M3 + M7
    #
    # Index conventions:
    #   U[i*m + j, c]: i in {0,1}, j in {0,1,2,3} -> row index 0-7
    #   V[j*p + k, c]: j in {0,1,2,3}, k in {0,1,2,3} -> row index 0-15
    #   W[k*n + i, c]: k in {0,1,2,3}, i in {0,1} -> row index 0-7

    n, m, p = DIMENSIONS
    rank = 28

    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen coefficient patterns for 2x2 multiplication
    # Each entry is a list of (coeff, row_in_A, row_in_B) for the 7 Strassen terms
    # A row indices: 0,1 (for 2x2 block)
    # B row indices: 0,1 (for 2x2 block)
    # W row indices: 0,1,2,3 (for 2x2 result block)
    
    # Strassen terms: for each term t in 0..6, we have:
    #   A coefficients: (row_a, coeff)
    #   B coefficients: (row_b, coeff)
    #   W contributions: (row_w, coeff)
    
    # Let's define the 7 Strassen multiplications explicitly.
    # For a 2x2 block multiplication P = A_block * B_block:
    # A_block rows: a0 (row 0), a1 (row 1)
    # B_block cols: b0 (col 0), b1 (col 1)
    # Result rows: p0 (row 0), p1 (row 1)
    
    # The 7 products:
    # S1: (a0 + a1) * (b0 + b1)
    # S2: (a1) * (b0)
    # S3: (a0) * (b1 - b0)
    # S4: (a1) * (b1 - b0)  -- wait, standard Strassen uses:
    # Let me use the standard formulation:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # S1 = (a+b)(e+f)
    # S2 = (c+d)(e)
    # S3 = (a)(f-g)
    # S4 = (d)(h-g)
    # S5 = (a+d)(h)
    # S6 = (b-c)(e+g)
    # S7 = (b-d)(f+h)
    # Result:
    # P00 = S1 - S4 + S5
    # P01 = S3 + S5
    # P10 = S2 + S4
    # P11 = S1 - S2 + S3 + S7
    
    # Map to our indices within a 2x2 block:
    # a = row 0, col 0 of A_block -> global A index depends on block
    # b = row 0, col 1 of A_block
    # c = row 1, col 0 of A_block
    # d = row 1, col 1 of A_block
    # e = row 0, col 0 of B_block
    # f = row 0, col 1 of B_block
    # g = row 1, col 0 of B_block
    # h = row 1, col 1 of B_block
    
    # For each of the 4 block multiplications, we need to map local indices to global.
    
    # Block definitions:
    # M1: A1 (rows 0-1 of A, cols 0-1) * B11 (rows 0-1 of B, cols 0-1)
    #   A global rows: 0, 1 (i=0,1; j=0,1 -> U[0], U[1])
    #   Wait, U index is i*m + j = i*4 + j.
    #   A1 corresponds to i in {0,1}, j in {0,1}.
    #   U row for (i, j): i*4 + j.
    #   So A1: U rows 0, 1, 4, 5.
    #   a = U[0] (i=0,j=0), b = U[1] (i=0,j=1), c = U[4] (i=1,j=0), d = U[5] (i=1,j=1)
    #   B11: rows 0-1, cols 0-1 of B. V index is j*p + k = j*4 + k.
    #   B11: V rows 0,1 (j=0,k=0,1) and 4,5 (j=1,k=0,1).
    #   e = V[0] (j=0,k=0), f = V[1] (j=0,k=1), g = V[4] (j=1,k=0), h = V[5] (j=1,k=1)
    #   Result: C block (0,0) i.e., i in {0,1}, k in {0,1}. W index is k*n + i = k*2 + i.
    #   W rows: 0,1 (k=0,i=0,1) and 2,3 (k=1,i=0,1).
    #   P00 -> W[0] (k=0,i=0), P01 -> W[1] (k=0,i=1), P10 -> W[2] (k=1,i=0), P11 -> W[3] (k=1,i=1)
    
    # M2: A1 * B12 (rows 0-1 of B, cols 2-3)
    #   A same as M1.
    #   B12: V rows 2,3 (j=0,k=2,3) and 6,7 (j=1,k=2,3).
    #   e = V[2], f = V[3], g = V[6], h = V[7]
    #   Result: C block (0,1) i.e., i in {0,1}, k in {2,3}.
    #   W rows: 4,5 (k=2,i=0,1) and 6,7 (k=3,i=0,1).
    #   P00 -> W[4], P01 -> W[5], P10 -> W[6], P11 -> W[7]
    
    # M3: A2 (rows 0-1 of A, cols 2-3) * B21 (rows 2-3 of B, cols 0-1)
    #   A2: U rows 2,3 (i=0,j=2,3) and 6,7 (i=1,j=2,3).
    #   a = U[2], b = U[3], c = U[6], d = U[7]
    #   B21: V rows 8,9 (j=2,k=0,1) and 12,13 (j=3,k=0,1).
    #   e = V[8], f = V[9], g = V[12], h = V[13]
    #   Result: C block (1,0) i.e., i in {0,1}, k in {0,1}.
    #   Same W rows as M1: 0,1,2,3.
    #   P00 -> W[0], P01 -> W[1], P10 -> W[2], P11 -> W[3]
    
    # M4: A2 * B22 (rows 2-3 of B, cols 2-3)
    #   A same as M3.
    #   B22: V rows 10,11 (j=2,k=2,3) and 14,15 (j=3,k=2,3).
    #   e = V[10], f = V[11], g = V[14], h = V[15]
    #   Result: C block (1,1) i.e., i in {0,1}, k in {2,3}.
    #   Same W rows as M2: 4,5,6,7.
    #   P00 -> W[4], P01 -> W[5], P10 -> W[6], P11 -> W[7]

    def add_strassen_block(base_col, a_rows, b_rows, w_rows):
        """
        Add Strassen decomposition for one 2x2 block multiplication.
        a_rows: [a, b, c, d] global U row indices
        b_rows: [e, f, g, h] global V row indices
        w_rows: [P00, P01, P10, P11] global W row indices
        """
        a, b, c, d = a_rows
        e, f, g, h = b_rows
        p00, p01, p10, p11 = w_rows
        
        # S1 = (a+b)(e+f): U[a]+=1, U[b]+=1, V[e]+=1, V[f]+=1, W[p00]+=1, W[p11]+=1
        U[a][base_col + 0] += 1
        U[b][base_col + 0] += 1
        V[e][base_col + 0] += 1
        V[f][base_col + 0] += 1
        W[p00][base_col + 0] += 1
        W[p11][base_col + 0] += 1
        
        # S2 = (c+d)(e): U[c]+=1, U[d]+=1, V[e]+=1, W[p10]+=1, W[p11]-=1
        U[c][base_col + 1] += 1
        U[d][base_col + 1] += 1
        V[e][base_col + 1] += 1
        W[p10][base_col + 1] += 1
        W[p11][base_col + 1] -= 1
        
        # S3 = (a)(f-g): U[a]+=1, V[f]+=1, V[g]-=1, W[p01]+=1, W[p11]+=1
        U[a][base_col + 2] += 1
        V[f][base_col + 2] += 1
        V[g][base_col + 2] -= 1
        W[p01][base_col + 2] += 1
        W[p11][base_col + 2] += 1
        
        # S4 = (d)(h-g): U[d]+=1, V[h]+=1, V[g]-=1, W[p00]-=1, W[p10]+=1
        U[d][base_col + 3] += 1
        V[h][base_col + 3] += 1
        V[g][base_col + 3] -= 1
        W[p00][base_col + 3] -= 1
        W[p10][base_col + 3] += 1
        
        # S5 = (a+d)(h): U[a]+=1, U[d]+=1, V[h]+=1, W[p00]+=1, W[p01]+=1
        U[a][base_col + 4] += 1
        U[d][base_col + 4] += 1
        V[h][base_col + 4] += 1
        W[p00][base_col + 4] += 1
        W[p01][base_col + 4] += 1
        
        # S6 = (b-c)(e+g): U[b]+=1, U[c]-=1, V[e]+=1, V[g]+=1, W[p11]+=1
        U[b][base_col + 5] += 1
        U[c][base_col + 5] -= 1
        V[e][base_col + 5] += 1
        V[g][base_col + 5] += 1
        W[p11][base_col + 5] += 1
        
        # S7 = (b-d)(f+h): U[b]+=1, U[d]-=1, V[f]+=1, V[h]+=1, W[p11]+=1
        U[b][base_col + 6] += 1
        U[d][base_col + 6] -= 1
        V[f][base_col + 6] += 1
        V[h][base_col + 6] += 1
        W[p11][base_col + 6] += 1
    
    # M1: A1 * B11 -> result block (0,0)
    add_strassen_block(0,
                       a_rows=[0, 1, 4, 5],
                       b_rows=[0, 1, 4, 5],
                       w_rows=[0, 1, 2, 3])
    
    # M2: A1 * B12 -> result block (0,1)
    add_strassen_block(7,
                       a_rows=[0, 1, 4, 5],
                       b_rows=[2, 3, 6, 7],
                       w_rows=[4, 5, 6, 7])
    
    # M3: A2 * B21 -> result block (1,0)
    add_strassen_block(14,
                       a_rows=[2, 3, 6, 7],
                       b_rows=[8, 9, 12, 13],
                       w_rows=[0, 1, 2, 3])
    
    # M4: A2 * B22 -> result block (1,1)
    add_strassen_block(21,
                       a_rows=[2, 3, 6, 7],
                       b_rows=[10, 11, 14, 15],
                       w_rows=[4, 5, 6, 7])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
