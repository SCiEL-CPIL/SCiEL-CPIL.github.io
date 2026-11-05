import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = 3, 3, 8
    
    # We will construct a rank 68 decomposition.
    # Breakdown:
    # C = A B
    # C[0:2, :] = A[0:2, :] B
    # C[2, :] = A[2, :] B
    
    # Part 1: A[0:2, :] B (2x3 * 3x8 -> 2x8)
    # Split A[0:2, :] into A_top_left (2x2) and A_top_right (2x1)
    # Split B into B_left (3x8) ... wait, B is 3x8.
    # A[0:2, :] = [A00 A01 A02] [A10 A11 A12] (rows 0,1)
    # B = [B00 B01 ...] (rows 0,1,2)
    
    # Product P1 = A[0:2, :] B
    # P1 = A[0:2, 0:2] B[0:2, :] + A[0:2, 2] B[2, :]
    
    # Term 1: A[0:2, 0:2] B[0:2, :] (2x2 * 2x8 -> 2x8)
    # Rank for 2x2x8 is 4 * rank(2x2x2) = 4 * 7 = 28.
    # We can implement this as 4 independent Strassen algorithms for the 8 columns?
    # No, Strassen is for 2x2x2.
    # We can group the 8 columns into 4 groups of 2 columns.
    # For each group of 2 columns, use Strassen.
    
    # Term 2: A[0:2, 2] B[2, :] (2x1 * 1x8 -> 2x8)
    # Rank is 2 * 8 = 16.
    
    # Part 2: A[2, :] B (1x3 * 3x8 -> 1x8)
    # Rank is 3 * 8 = 24.
    
    # Total Rank = 28 + 16 + 24 = 68.
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)] # 9x68
    V = [[0] * rank for _ in range(m * p)] # 24x68
    W = [[0] * rank for _ in range(p * n)] # 24x68
    
    col = 0
    
    # --- Part 1: A[0:2, 0:2] B[0:2, :] ---
    # Indices:
    # A rows 0,1; cols 0,1.
    # B rows 0,1; cols 0..7.
    # W indices: k*n + i.
    # k is col of B (0..7), i is row of A (0,1).
    
    # We process 4 blocks of 2 columns from B.
    for block in range(4):
        b_col_start = block * 2
        b_col_end = b_col_start + 2
        
        # Strassen for A[0:2, 0:2] and B[0:2, b_col_start:b_col_end]
        # A_11 = A[0,0], A_12 = A[0,1], A_21 = A[1,0], A_22 = A[1,1]
        # B_11 = B[0, b_col_start], B_12 = B[0, b_col_start+1]
        # B_21 = B[1, b_col_start], B_22 = B[1, b_col_start+1]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][col] = 1; U[3][col] = 1 # A[0,0], A[1,1]
        V[0*2+b_col_start][col] = 1; V[2*2+b_col_start][col] = 1 # B[0, c0], B[1, c0]
        W[(b_col_start)*3 + 0][col] = 1; W[(b_col_start+1)*3 + 0][col] = 1 # C[0, c0], C[0, c1]?
        # Wait, W index is k*n+i.
        # For M1, it contributes to C[0, c0], C[0, c1], C[1, c0], C[1, c1].
        # Strassen factors for 2x2x2:
        # U: A coeffs, V: B coeffs, W: C coeffs.
        
        # Let's implement Strassen factors explicitly for each 2x2x2 block.
        # Block c0, c1 in B.
        
        # M1
        U[0][col] += 1; U[3][col] += 1
        V[0*8+b_col_start][col] += 1; V[1*8+b_col_start][col] += 1
        W[b_col_start*3+0][col] += 1; W[b_col_start*3+1][col] += 1
        W[(b_col_start+1)*3+0][col] += 1; W[(b_col_start+1)*3+1][col] += 1
        col += 1
        
        # M2
        U[1][col] += 1; U[3][col] += 1
        V[0*8+b_col_start][col] += 1
        W[b_col_start*3+1][col] += 1; W[b_col_start*3+0][col] -= 1
        W[(b_col_start+1)*3+1][col] += 1; W[(b_col_start+1)*3+0][col] -= 1
        col += 1
        
        # M3
        U[0][col] += 1
        V[0*8+b_col_start+1][col] += 1; V[1*8+b_col_start+1][col] += 1
        W[b_col_start*3+0][col] -= 1; W[b_col_start*3+1][col] += 1
        W[(b_col_start+1)*3+0][col] -= 1; W[(b_col_start+1)*3+1][col] += 1
        col += 1
        
        # M4
        U[3][col] += 1
        V[0*8+b_col_start][col] += 1; V[0*8+b_col_start+1][col] -= 1
        W[b_col_start*3+0][col] += 1; W[b_col_start*3+1][col] -= 1
        W[(b_col_start+1)*3+0][col] += 1; W[(b_col_start+1)*3+1][col] -= 1
        col += 1
        
        # M5
        U[0][col] += 1; U[2][col] += 1
        V[1*8+b_col_start+1][col] += 1
        W[b_col_start*3+1][col] += 1; W[b_col_start*3+0][col] -= 1
        W[(b_col_start+1)*3+1][col] += 1; W[(b_col_start+1)*3+0][col] -= 1
        col += 1
        
        # M6
        U[1][col] += 1; U[0][col] -= 1
        V[0*8+b_col_start][col] += 1; V[1*8+b_col_start][col] += 1
        W[b_col_start*3+0][col] += 1; W[b_col_start*3+1][col] -= 1
        W[(b_col_start+1)*3+0][col] += 1; W[(b_col_start+1)*3+1][col] -= 1
        col += 1
        
        # M7
        U[2][col] += 1; U[3][col] -= 1
        V[0*8+b_col_start+1][col] += 1; V[1*8+b_col_start+1][col] += 1
        W[b_col_start*3+0][col] -= 1; W[b_col_start*3+1][col] += 1
        W[(b_col_start+1)*3+0][col] -= 1; W[(b_col_start+1)*3+1][col] += 1
        col += 1

    # --- Part 1b: A[0:2, 2] B[2, :] ---
    # A[0,2], A[1,2]
    # B[2, 0..7]
    # W indices: k*3 + i (i=0,1)
    
    for k in range(8):
        # C[0, k] += A[0, 2] * B[2, k]
        U[0*3+2][col] = 1
        V[2*8+k][col] = 1
        W[k*3+0][col] = 1
        col += 1
        
        # C[1, k] += A[1, 2] * B[2, k]
        U[1*3+2][col] = 1
        V[2*8+k][col] = 1
        W[k*3+1][col] = 1
        col += 1

    # --- Part 2: A[2, :] B ---
    # A[2, 0], A[2, 1], A[2, 2]
    # B[0..2, 0..7]
    
    for k in range(8):
        for j in range(3):
            U[2*3+j][col] = 1
            V[j*8+k][col] = 1
            W[k*3+2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
