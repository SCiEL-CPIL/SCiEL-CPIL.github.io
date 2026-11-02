"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _strassen_2x2x2():
    # Rank-7 tensor decomposition for 2x2 matrix multiplication.
    # Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # This is the standard Strassen decomposition.
    # U rows: (i,j) for i in 0,1 and j in 0,1 -> indices 0,1,2,3
    # V rows: (j,k) for j in 0,1 and k in 0,1 -> indices 0,1,2,3
    # W rows: (k,i) for k in 0,1 and i in 0,1 -> indices 0,1,2,3
    
    # The 7 Strassen multiplications are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # U factors (for A): row index = i*2+j
    # j=0: cols 0,1 -> row 0 (i=0,j=0), row 2 (i=1,j=0)
    # j=1: cols 0,1 -> row 1 (i=0,j=1), row 3 (i=1,j=1)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1              # B00
    W[2][1] = 1              # C10
    
    # M3 = A00(B10 - B11)
    U[0][2] = 1              # A00
    V[2][2] = 1; V[3][2] = -1 # B10, -B11
    W[1][2] = 1; W[3][2] = 1  # C01, C11
    
    # M4 = A11(B01 - B00)
    U[3][3] = 1              # A11
    V[1][3] = 1; V[0][3] = -1 # B01, -B00
    W[0][3] = 1; W[2][3] = 1  # C00, C10
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1              # B11
    W[1][4] = 1; W[0][4] = -1 # C01, -C00
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1 # A10, -A00
    V[0][5] = 1; V[1][5] = 1  # B00, B01
    W[2][5] = -1              # -C10
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1 # A01, -A11
    V[2][6] = 1; V[3][6] = 1  # B10, B11
    W[1][6] = -1; W[3][6] = 1 # -C01, C11
    
    return U, V, W

def _build_4x4x7_strassen():
    # Decompose 4x4 * 4x7 as block multiplication.
    # A is 4x4, split into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # B is 4x7, split into 2x(2,2,2,1) blocks... this is tricky.
    
    # Actually, let's use a simpler approach:
    # 4x4 * 4x7 = sum over 4x4 * 4x1 (7 times)
    # Each 4x4 * 4x1 can use rank-28 (4*4*1=16 schoolbook, or Strassen-like)
    
    # Better: Use the known rank-49 for 4x4*4x4 and adapt.
    # 4x4*4x7 = 4x4*4x4 + 4x4*4x3
    # Rank 4x4*4x4 = 49 (known optimal)
    # Rank 4x4*4x3 = ? At most 48 (schoolbook), or use Strassen recursively.
    
    # Let's just use schoolbook for 4x4*4x4 (rank 64) and 4x4*4x3 (rank 48) = 112.
    # Or use Strassen for 4x4*4x4: rank 49.
    
    # For now, return schoolbook 4x4*4x7 = rank 112.
    return _schoolbook(4, 4, 7)

def construct():
    start = time.time()
    deadline = start + 140.0
    
    # First, try to build a Strassen-based decomposition.
    # For 4x4*4x7, we can split as:
    # A (4x4) = [[A00, A01], [A10, A11]] where each block is 2x2
    # B (4x7) = [[B00, B01, B02, B03], [B10, B11, B12, B13]]
    #           where B0*, B1* are 2x(2,2,2,1) blocks... not clean.
    
    # Alternative: B (4x7) = [B_left (4x4), B_right (4x3)]
    # C = A * B = A * B_left + A * B_right
    # C_left = A * B_left (4x4), C_right = A * B_right (4x3)
    
    # For A * B_left (4x4 * 4x4), use Strassen: rank 49
    # For A * B_right (4x4 * 4x3), use schoolbook: rank 48
    # Total: 49 + 48 = 97
    
    # Let's implement this.
    def _strassen_4x4x4():
        # Use 2x2 Strassen recursively to get rank-49 for 4x4*4x4.
        # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
        # B = [[B00, B01], [B10, B11]] (2x2 blocks of 2x2)
        
        # C00 = (A00+A11)(B00+B11) + (A10+A11)B00 + A00(B10-B11) + A11(B01-B00) ... 
        # Actually, standard recursive Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mi is a 2x2*2x2 product, which uses rank 7.
        # So total rank = 7 * 7 = 49.
        
        # Build the tensor.
        # U: 16 x 49, V: 16 x 49, W: 16 x 49
        
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        # For each of the 7 Strassen steps, we have a 2x2*2x2 problem.
        # The 2x2*2x2 U,V,W have 4 rows each and 7 cols.
        # We need to map these to the 16-row, 49-col space.
        
        # A row index: i*4+j where i in {0,1} is block row, j in {0,1,2,3} is within block.
        # Wait, A is 4x4. Block (bi, bj) where bi,bj in {0,1}.
        # Element (i,j) of A: block (i//2, j//2), position (i%2, j%2)
        # Row index in U: i*4+j
        
        # For Strassen step s (0..6), the 2x2*2x2 decomposition has:
        # U_s: 4x7, V_s: 4x7, W_s: 4x7
        # Row r in U_s corresponds to (block_row, within_block_row) = (r//2, r%2)
        # This maps to A rows: for block_row=0: A rows 0,1; block_row=1: A rows 2,3
        # Within block: r%2=0 -> first row of block, r%2=1 -> second row of block
        # So U_s row r maps to A row: block_row*2 + r%2 = (r//2)*2 + r%2
        
        # Similarly for V_s: maps to B rows.
        # W_s: maps to C columns? No, W is (k,i) where k is B col, i is A row.
        # W_s row r: (k_block, i_block, within)... 
        # Actually W row index is k*n+i. For 4x4*4x4, n=p=4.
        # W_s row r: k_block = r//2, i_block = r%2? No...
        
        # Let me re-derive. For 2x2*2x2:
        # U row (i,j) -> i*2+j
        # V row (j,k) -> j*2+k
        # W row (k,i) -> k*2+i
        
        # For 4x4*4x4 with Strassen:
        # The "outer" 2x2 blocks: A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
        # Each Aij, Bjk is 2x2.
        
        # For Strassen step s, we compute M_s = (linear combo of A blocks) * (linear combo of B blocks)
        # This is a 2x2*2x2 product, giving a 2x2 result block.
        
        # Let's index the 49 columns as s*7 + t where s in 0..6, t in 0..6.
        
        # Get the 2x2*2x2 Strassen factors.
        U2, V2, W2 = _strassen_2x2x2()
        
        for s in range(7):
            # For each Strassen step s, the U2 column t gives coefficients.
            # U2[i2, t] where i2 = block_row*2 + within_row
            # This means: for block_row in {0,1}, within_row in {0,1}:
            #   A row = block_row*2 + within_row
            #   coefficient = U2[block_row*2+within_row, t]
            
            # Similarly for V2[j2, t]: block_col of B, within_col
            #   B row = block_col*2 + within_col
            #   coefficient = V2[block_col*2+within_col, t]
            
            # W2[k2, t]: k2 = k_block*2 + i_within? No.
            # W row for 2x2*2x2: (k,i) -> k*2+i
            # k is B column, i is A row.
            # In the 4x4 context: k_block, i_block, and within positions.
            # W2[k_block*2 + i_within, t]... hmm, this doesn't map cleanly.
            
            # Actually, W2 row index = k*2+i where k in {0,1} (B col block), i in {0,1} (A row block).
            # The within-block positions are handled by the 2x2 structure.
            # W2[k*2+i, t] means: C block (i, k) gets this coefficient at the appropriate within-block position.
            # But W2 is 4x7, so rows are (k,i) for k,i in {0,1}.
            # In the 4x4 output C, block (i,k) is at rows i*2..i*2+1, cols k*2..k*2+1.
            # W row index in the full 4x4*4x4: col*4+row = k*4+i for block level.
            # But we need within-block: (k_block*2+k_within)*4 + (i_block*2+i_within)
            
            # This is getting complicated. Let me just use schoolbook for 4x4*4x4 = rank 64.
            pass
        
        # For simplicity and correctness, use schoolbook.
        return _schoolbook(4, 4, 4)
    
    # Given complexity, just return schoolbook 4x4*4x7 = rank 112.
    U, V, W = _schoolbook(4, 4, 7)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
