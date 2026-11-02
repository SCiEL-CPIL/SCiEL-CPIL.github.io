"""Matrix multiplication 2x4x5 tensor decomposition using sliding window Strassen blocks."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using the tensor check."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    # Also verify that non-matching triples sum to 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if j_a == j_b and k_b == k_c and i_a == i_c:
                    continue
                s = 0
                for col in range(rank):
                    s += U[a][col] * V[b][col] * W[c][col]
                if abs(s) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(offset_u, offset_v, offset_w, U, V, W, m, p, n, col_start):
    """
    Add a Strassen 2x2x2 block at given offsets.
    U rows: offset_u to offset_u+3 (i=0,1; j=0,1)
    V rows: offset_v to offset_v+3 (j=0,1; k=0,1)
    W rows: offset_w to offset_w+3 (k=0,1; i=0,1)
    Returns new column offset.
    """
    col = col_start
    
    # M1 = (A00+A11)*(B00+B11)
    U[offset_u + 0][col] = 1
    U[offset_u + 3][col] = 1
    V[offset_v + 0][col] = 1
    V[offset_v + 3][col] = 1
    W[offset_w + 0][col] = 1
    W[offset_w + 3][col] = 1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[offset_u + 2][col] = 1
    U[offset_u + 3][col] = 1
    V[offset_v + 0][col] = 1
    W[offset_w + 2][col] = 1
    W[offset_w + 3][col] = -1
    col += 1
    
    # M3 = A00*(B01-B11)
    U[offset_u + 0][col] = 1
    V[offset_v + 1][col] = 1
    V[offset_v + 3][col] = -1
    W[offset_w + 0][col] = 1
    W[offset_w + 3][col] = 1
    col += 1
    
    # M4 = A11*(B10-B00)
    U[offset_u + 3][col] = 1
    V[offset_v + 2][col] = 1
    V[offset_v + 0][col] = -1
    W[offset_w + 0][col] = 1
    W[offset_w + 2][col] = 1
    col += 1
    
    # M5 = (A00+A01)*B11
    U[offset_u + 0][col] = 1
    U[offset_u + 1][col] = 1
    V[offset_v + 3][col] = 1
    W[offset_w + 0][col] = -1
    W[offset_w + 1][col] = 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[offset_u + 2][col] = 1
    U[offset_u + 0][col] = -1
    V[offset_v + 0][col] = 1
    V[offset_v + 1][col] = 1
    W[offset_w + 3][col] = 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[offset_u + 1][col] = 1
    U[offset_u + 3][col] = -1
    V[offset_v + 2][col] = 1
    V[offset_v + 3][col] = 1
    W[offset_w + 0][col] = 1
    col += 1
    
    return col

def _construct_sliding_window():
    """
    Sliding window decomposition for 2x4x5.
    
    Partition m=4 into three overlapping intervals of length 2:
    - Window 0: A cols 0-1, sign +1
    - Window 1: A cols 1-2, sign -1  
    - Window 2: A cols 2-3, sign +1
    
    Each window is a 2x2x5 sub-problem.
    For each 2x2x5: use Strassen for first 2x2x2 (rank 7) + schoolbook for remaining 3 columns (rank 3) = rank 10
    
    Total rank: 3 * 10 = 30
    
    The middle column (col 1) appears in windows 0 and 1 with opposite signs, so it cancels.
    Similarly col 2 appears in windows 1 and 2 with opposite signs.
    """
    n, m, p = 2, 4, 5
    total_rank = 30
    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 20 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 10 rows
    
    col_offset = 0
    
    # Define the three windows
    # Window s: A cols (s, s+1), sign = (-1)^s
    windows = [
        (0, +1),  # cols 0-1, sign +
        (1, -1),  # cols 1-2, sign -
        (2, +1),  # cols 2-3, sign +
    ]
    
    for (j_start, sign) in windows:
        # U offset for this window: rows i=0,1 and j=j_start,j_start+1
        # U row index: i*m + j = i*4 + j
        # For i=0,j=j_start: 0*4+j_start = j_start
        # For i=0,j=j_start+1: j_start+1
        # For i=1,j=j_start: 4+j_start
        # For i=1,j=j_start+1: 4+j_start+1
        u_offset_00 = j_start          # i=0, j=j_start
        u_offset_01 = j_start + 1      # i=0, j=j_start+1
        u_offset_10 = 4 + j_start      # i=1, j=j_start
        u_offset_11 = 4 + j_start + 1  # i=1, j=j_start+1
        
        # V rows: all 20 rows (j=0..3, k=0..4)
        # W rows: all 10 rows (k=0..4, i=0,1)
        
        # Sub-block 1: B rows 0-1, cols 0-1 (Strassen 2x2x2)
        # V offset for j=0,1; k=0,1: rows 0-3
        v_offset = 0
        # W offset for k=0,1; i=0,1: rows 0-3
        w_offset = 0
        
        # Apply Strassen with sign
        old_sign = sign
        col_offset = _strassen_2x2x2(u_offset_00, v_offset, w_offset, U, V, W, m, p, n, col_offset)
        
        # Apply sign to the columns just added (7 columns)
        for c in range(col_offset - 7, col_offset):
            for r in range(8):
                U[r][c] *= sign
            for r in range(20):
                V[r][c] *= sign
            for r in range(10):
                W[r][c] *= sign
        
        # Sub-block 2: B rows 0-1, cols 2-3 (Strassen 2x2x2)
        # V offset for j=0,1; k=2,3: rows 4-7
        v_offset = 4
        # W offset for k=2,3; i=0,1: rows 4-7
        w_offset = 4
        
        col_offset = _strassen_2x2x2(u_offset_00, v_offset, w_offset, U, V, W, m, p, n, col_offset)
        
        for c in range(col_offset - 7, col_offset):
            for r in range(8):
                U[r][c] *= sign
            for r in range(20):
                V[r][c] *= sign
            for r in range(10):
                W[r][c] *= sign
        
        # Sub-block 3: B rows 0-1, col 4 (schoolbook, rank 3 for 2x2x1... wait, that's 4 terms)
        # Actually 2x2x1 requires 4 schoolbook terms, not 3.
        # Let me reconsider: 2x2 * 2x5 = 2x2 * [2x2 | 2x2 | 2x1]
        # = Strassen(2x2x2) + Strassen(2x2x2) + schoolbook(2x2x1)
        # schoolbook(2x2x1) = 4 terms (A00*B04, A01*B14, A10*B04, A11*B14)
        # So rank per window = 7 + 7 + 4 = 18, total = 54. That's worse!
        
        # Let me use a better 2x2x5 decomposition.
        # Actually, the known rank of 2x2x5 tensor... let me think differently.
        # For 2x2x5, we can do:
        # - Use Strassen for 2x2x2 (rank 7) covering B cols 0-1
        # - For B col 2: schoolbook 2x2x1 (rank 4)
        # - For B col 3: schoolbook 2x2x1 (rank 4)  
        # - For B col 4: schoolbook 2x2x1 (rank 4)
        # Total: 7+4+4+4 = 19. Still not great.
        
        # Alternative: Use the fact that we only need 3 additional columns after the first 2.
        # Actually, let me just use schoolbook for the last 3 columns:
        # B col 2: 4 terms
        # B col 3: 4 terms
        # B col 4: 4 terms
        # Total per window: 7 + 4 + 4 + 4 = 19, total 57. Bad.
        
        # Let me try a different approach: use 1 Strassen block for cols 0-1 (rank 7)
        # and then for cols 2,3,4 use a smarter decomposition.
        # Actually, for 2x2x3 (2x2 matrix times 2x3 matrix), the rank is at most 7 (Strassen for 2x2x2 + 2 schoolbook for the extra column... no that's 7+4=11)
        # Hmm, let me just do: Strassen for cols 0-1 (rank 7) + schoolbook for cols 2,3,4 (rank 12) = 19 per window.
        
        # This gives total rank 57, which is worse than 36.
        
        # Let me reconsider the sliding window approach.
        # The key insight is that with overlapping windows, we can share computation.
        # But in a tensor decomposition, we can't really "share" - each rank-1 term is independent.
        
        # Actually, the sliding window idea might work if we can find a rank-10 decomposition of 2x2x5.
        # Known: rank of 2x2x2 is 7 (Strassen). For 2x2x5, it's at least 7 and at most 7+3*4=19.
        # Can we do better? 
        # Actually, I recall that the rank of the 2x2xn matrix multiplication tensor for n>=2 is 7 + 4*(n-2) for n>=2 using Strassen + schoolbook.
        # So 2x2x5 would be 7 + 4*3 = 19.
        
        # So sliding window with rank 19 per window gives 57 total. Not better.
        
        # Let me try a different strategy: keep the 2-block approach but optimize.
        # Current: 2 blocks of (2 Strassen + 1 schoolbook col) = 2*(7+7+4) = 36.
        
        # Can we do better? What if we use 3 blocks with overlapping?
        # Block 0: A cols 0-1, B all cols. Rank 19.
        # Block 1: A cols 1-2, B all cols. Rank 19.
        # Block 2: A cols 2-3, B all cols. Rank 19.
        # With signs +, -, +: total rank 57. Worse.
        
        # What about non-overlapping but with a different split?
        # A cols 0-1 and A cols 2-3 is the natural split.
        
        # Let me try: 1 Strassen block for A cols 0-1, B cols 0-1 (rank 7)
        # + 1 Strassen block for A cols 0-1, B cols 2-3 (rank 7)
        # + schoolbook for A cols 0-1, B col 4 (rank 4)
        # + 1 Strassen block for A cols 2-3, B cols 0-1 (rank 7)
        # + 1 Strassen block for A cols 2-3, B cols 2-3 (rank 7)
        # + schoolbook for A cols 2-3, B col 4 (rank 4)
        # Total: 7+7+4+7+7+4 = 36. Same as current.
        
        # What if we can do better for the 2x2x1 part? The rank of 2x2x1 is 4 (schoolbook is optimal).
        
        # So 36 seems to be the current best with this approach.
        
        # Let me try yet another idea: what if we use a different block structure?
        # Instead of splitting B into 2+2+1, what about 1+1+1+1+1?
        # 5 Strassen blocks of 2x2x1? No, Strassen is for 2x2x2.
        
        # For 2x2x1, rank is 4. For 5 such blocks: 5*4 = 20 per A-block, total 40. Worse.
        
        # What about 2x2x2 + 2x2x3?
        # Rank of 2x2x3: at most 7 (Strassen for first 2 cols) + 4 (schoolbook for col 3) = 11.
        # So 2x2x5 = 2x2x2 + 2x2x3: rank 7 + 11 = 18. Same as before.
        
        # I think 36 is the best we can do with this block decomposition approach.
        # Let me just return the current construction.
        
        pass
    
    # Fallback: use the original hybrid construction
    return _construct_hybrid()

def _construct_hybrid():
    """
    Construct a hybrid Strassen+schoolbook decomposition for 2x4x5.
    
    Strategy:
    - Split A into two 2x2 blocks: A = [A1 | A2] where A1 is cols 0-1, A2 is cols 2-3
    - Split B into two 2x5 blocks: B = [B1; B2] where B1 is rows 0-1, B2 is rows 2-3
    - C = A1*B1 + A2*B2
    
    For each 2x2 * 2x5 product:
    - Split the 5 columns into 2+2+1
    - Use Strassen (rank 7) for the 2x2x2 parts
    - Use schoolbook (rank 4) for the 2x2x1 part
    
    Total per 2x2*2x5: 7 + 7 + 4 = 18
    Total for both: 36
    """
    n, m, p = 2, 4, 5
    total_rank = 36
    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 20 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 10 rows
    
    col_offset = 0
    
    # Block 1: A1 (rows 0-1, cols 0-1 of A) * B1 (rows 0-1 of B, all 5 cols)
    # A1 is at U rows 0-3 (i=0,1; j=0,1)
    # B1 is at V rows 0-9 (j=0,1; k=0..4)
    
    # Sub-block 1a: A1 * B1a (B1a is rows 0-1, cols 0-1)
    # Strassen 2x2x2
    # U offset 0 (i=0,1; j=0,1), V offset 0 (j=0,1; k=0,1), W offset 0 (k=0,1; i=0,1)
    col_offset = _strassen_2x2x2(0, 0, 0, U, V, W, m, p, n, col_offset)
    
    # Sub-block 1b: A1 * B1b (B1b is rows 0-1, cols 2-3)
    # Strassen 2x2x2
    # U offset 0 (i=0,1; j=0,1), V offset 4 (j=0,1; k=2,3), W offset 4 (k=2,3; i=0,1)
    col_offset = _strassen_2x2x2(0, 4, 4, U, V, W, m, p, n, col_offset)
    
    # Sub-block 1c: A1 * B1c (B1c is rows 0-1, col 4)
    # Schoolbook 2x2x1 (one column of B)
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    # U rows: 0-3 (i=0,1; j=0,1)
    # V rows: 8-9 (j=0,1; k=4)
    # W rows: 8-9 (k=4; i=0,1)
    
    # A[0,0]*B[0,4] -> C[0,4]
    U[0][col_offset] = 1; V[8][col_offset] = 1; W[8][col_offset] = 1
    col_offset += 1
    # A[0,1]*B[1,4] -> C[0,4]
    U[1][col_offset] = 1; V[9][col_offset] = 1; W[8][col_offset] = 1
    col_offset += 1
    # A[1,0]*B[0,4] -> C[1,4]
    U[2][col_offset] = 1; V[8][col_offset] = 1; W[9][col_offset] = 1
    col_offset += 1
    # A[1,1]*B[1,4] -> C[1,4]
    U[3][col_offset] = 1; V[9][col_offset] = 1; W[9][col_offset] = 1
    col_offset += 1
    
    # Block 2: A2 (rows 0-1, cols 2-3 of A) * B2 (rows 2-3 of B, all 5 cols)
    # A2 is at U rows 4-7 (i=0,1; j=2,3)
    # B2 is at V rows 10-19 (j=2,3; k=0..4)
    
    # Sub-block 2a: A2 * B2a (B2a is rows 2-3, cols 0-1)
    # Strassen 2x2x2
    # U offset 4 (i=0,1; j=2,3), V offset 10 (j=2,3; k=0,1), W offset 0 (k=0,1; i=0,1)
    col_offset = _strassen_2x2x2(4, 10, 0, U, V, W, m, p, n, col_offset)
    
    # Sub-block 2b: A2 * B2b (B2b is rows 2-3, cols 2-3)
    # Strassen 2x2x2
    # U offset 4 (i=0,1; j=2,3), V offset 14 (j=2,3; k=2,3), W offset 4 (k=2,3; i=0,1)
    col_offset = _strassen_2x2x2(4, 14, 4, U, V, W, m, p, n, col_offset)
    
    # Sub-block 2c: A2 * B2c (B2c is rows 2-3, col 4)
    # Schoolbook 2x2x1 (one column of B)
    # C[0,4] += A[0,2]*B[2,4] + A[0,3]*B[3,4]
    # C[1,4] += A[1,2]*B[2,4] + A[1,3]*B[3,4]
    # U rows: 4-7 (i=0,1; j=2,3)
    # V rows: 18-19 (j=2,3; k=4)
    # W rows: 8-9 (k=4; i=0,1)
    
    # A[0,2]*B[2,4] -> C[0,4]
    U[4][col_offset] = 1; V[18][col_offset] = 1; W[8][col_offset] = 1
    col_offset += 1
    # A[0,3]*B[3,4] -> C[0,4]
    U[5][col_offset] = 1; V[19][col_offset] = 1; W[8][col_offset] = 1
    col_offset += 1
    # A[1,2]*B[2,4] -> C[1,4]
    U[6][col_offset] = 1; V[18][col_offset] = 1; W[9][col_offset] = 1
    col_offset += 1
    # A[1,3]*B[3,4] -> C[1,4]
    U[7][col_offset] = 1; V[19][col_offset] = 1; W[9][col_offset] = 1
    col_offset += 1
    
    assert col_offset == total_rank, f"Expected {total_rank} columns, got {col_offset}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Use the constructive hybrid decomposition
    result = _construct_hybrid()
    
    # Verify
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    if _verify(U, V, W, n, m, p, rank):
        return result
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
