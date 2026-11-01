"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 13
    # Rank = 4 * p = 52 (Strassen-like decomposition for 2x2 blocks)
    rank = 4 * p
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 26 x rank
    W = [[0] * rank for _ in range(p * n)]  # 26 x rank
    
    # Strassen decomposition for 2x2 matrix multiplication:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # 7 terms:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B00 - B01)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For each column k of B, we apply this decomposition.
    # Since B is 2x13, each row of B has 13 entries.
    # The U factors are the same for all columns (they only depend on A's structure),
    # but the V factors depend on the specific column of B.
    
    # U entries for the 7 Strassen terms (repeated for each of 13 columns):
    # Row 0 = A00, Row 1 = A01, Row 2 = A10, Row 3 = A11
    # Term 1: A00+A11 times (B0k+B1k) -> U[0]=1, U[3]=1
    # Term 2: (A10+A11) times B0k -> U[2]=1, U[3]=1
    # Term 3: A00 times (B1k-B2k... wait, B01-B11 means row0-col1 - row1-col1)
    # Let me be more careful with indexing.
    
    # B is 2x13: B[j][k] where j in {0,1}, k in 0..12
    # A is 2x2: A[i][j] where i,j in {0,1}
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    # Strassen block decomposition:
    # A00 = A[0][0], A01 = A[0][1], A10 = A[1][0], A11 = A[1][1]
    # For each column k of B: B0k = B[0][k], B1k = B[1][k]
    
    # M1_k = (A00+A11) * (B0k+B1k)
    # M2_k = (A10+A11) * B0k
    # M3_k = A00 * B1k
    # M4_k = A11 * B0k  -- wait, let me use the standard form properly
    
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B00-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For our case, "B00" means the top-left block of B, but B is 2x13,
    # so we treat each column separately. The "blocks" here are scalar.
    # B00 -> B[0][k], B11 -> B[1][k], B01 -> B[0][k], B10 -> B[1][k]
    # Wait, this doesn't make sense with 2x2 block structure for a 2x13 matrix.
    
    # Let me reconsider. The standard approach for multiplying a 2x2 by 2xn:
    # We can still use the 7-term Strassen decomposition per column, but it's
    # not directly applicable since Strassen works on 2x2 blocks.
    
    # Actually, for 2x2 times 2xn, the naive rank is 4n. Strassen gives rank 7
    # for 2x2 times 2x2. For 2x2 times 2xn, we can group columns in pairs.
    
    # For 13 columns, we have 6 pairs + 1 single.
    # Each pair of columns: 2x2 x 2x2 -> rank 7
    # Single column: 2x2 x 2x1 -> rank 4 (naive)
    # Total: 6*7 + 4 = 46
    
    # Let's implement this:
    rank = 6 * 7 + 4  # = 46
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 26 x rank
    W = [[0] * rank for _ in range(p * n)]  # 26 x rank
    
    col = 0
    # Process 6 pairs of columns (0-1, 2-3, ..., 10-11)
    for pair_idx in range(6):
        k0 = 2 * pair_idx
        k1 = 2 * pair_idx + 1
        
        # For this pair, B columns are B[:, k0] and B[:, k1]
        # B00 = B[0][k0], B10 = B[1][k0], B01 = B[0][k1], B11 = B[1][k1]
        
        # M1 = (A00+A11)(B00+B11)
        # U: [1, 0, 0, 1]
        # V: B00+B11 = B[0][k0]+B[0][k1] for row0, B[1][k0]+B[1][k1] for row1
        # W: C00 += M1, C11 += M1
        # C00 is at row k0 of C (since C has 13 rows, 2 cols)
        # W index: C[i][k] -> index i*2+k or k*2+i? 
        # Output factor W[k*n+i, r] means W is indexed by (output_col, output_row)
        # C is 2x13, so C[i][k] where i in 0..1, k in 0..12
        # W index = k*n + i = k*2 + i
        
        # Term 1: M1 = (A00+A11)(B[0][k0]+B[0][k1]) ... wait
        # B00 in Strassen = top-left = B[0][k0], B11 = bottom-right = B[1][k1]
        # B00+B11 = B[0][k0] + B[1][k1]
        
        U[0][col] = 1
        U[3][col] = 1
        V[0 * p + k0][col] = 1  # B[0][k0]
        V[1 * p + k1][col] = 1  # B[1][k1]
        W[k0 * n + 0][col] = 1  # C[0][k0] += M1
        W[k1 * n + 1][col] = 1  # C[1][k1] += M1
        col += 1
        
        # Term 2: M2 = (A10+A11)B00 = (A10+A11)B[0][k0]
        U[2][col] = 1
        U[3][col] = 1
        V[0 * p + k0][col] = 1  # B[0][k0]
        W[k0 * n + 1][col] = 1  # C[1][k0] += M2
        col += 1
        
        # Term 3: M3 = A00(B01-B11) = A00(B[0][k1]-B[1][k1])
        U[0][col] = 1
        V[0 * p + k1][col] = 1  # B[0][k1]
        V[1 * p + k1][col] = -1  # -B[1][k1]
        W[k1 * n + 0][col] = 1  # C[0][k1] += M3
        W[k1 * n + 1][col] = 1  # C[1][k1] += M3
        col += 1
        
        # Term 4: M4 = A11(B00-B01) = A11(B[0][k0]-B[0][k1])
        U[3][col] = 1
        V[0 * p + k0][col] = 1  # B[0][k0]
        V[0 * p + k1][col] = -1  # -B[0][k1]
        W[k0 * n + 0][col] = -1  # C[0][k0] -= M4
        W[k0 * n + 1][col] = 1  # C[1][k0] += M4
        col += 1
        
        # Term 5: M5 = (A00+A01)B11 = (A00+A01)B[1][k1]
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + k1][col] = 1  # B[1][k1]
        W[k0 * n + 0][col] = 1  # C[0][k0] += M5
        W[k1 * n + 1][col] = 1  # C[1][k1] += M5
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01) = (A10-A00)(B[0][k0]+B[0][k1])
        U[0][col] = -1
        U[2][col] = 1
        V[0 * p + k0][col] = 1  # B[0][k0]
        V[0 * p + k1][col] = 1  # B[0][k1]
        W[k1 * n + 1][col] = 1  # C[1][k1] += M6
        col += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11) = (A01-A11)(B[1][k0]+B[1][k1])
        U[1][col] = 1
        U[3][col] = -1
        V[1 * p + k0][col] = 1  # B[1][k0]
        V[1 * p + k1][col] = 1  # B[1][k1]
        W[k0 * n + 0][col] = -1  # C[0][k0] -= M7
        W[k1 * n + 0][col] = 1  # C[0][k1] += M7
        col += 1
    
    # Single column 12 (k=12): naive 4 terms
    k = 12
    # C[0][12] = A[0][0]*B[0][12] + A[0][1]*B[1][12]
    # C[1][12] = A[1][0]*B[0][12] + A[1][1]*B[1][12]
    
    # Term 1: A00 * B0k
    U[0][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1
    
    # Term 2: A01 * B1k
    U[1][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1
    
    # Term 3: A10 * B0k
    U[2][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1
    
    # Term 4: A11 * B1k
    U[3][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
