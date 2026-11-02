"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _check_deadline():
    return time.time() < DEADLINE - SAFETY_MARGIN

def _construct_2x3x2_rank7():
    """
    Rank-7 decomposition of 2x3x2 matrix multiplication.
    A is 2x3, B is 3x2, C is 2x2.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    We use 7 rank-1 tensors.
    """
    # U: (2*3, 7) = (6, 7)
    # V: (3*2, 7) = (6, 7)
    # W: (2*2, 7) = (4, 7)
    #
    # Indexing:
    # U row: i*3 + j for i in 0..1, j in 0..2
    # V row: j*2 + k for j in 0..2, k in 0..1
    # W row: k*2 + i for k in 0..1, i in 0..1 (trace convention W[k*n+i, r])
    
    U = [[0]*7 for _ in range(6)]
    V = [[0]*7 for _ in range(6)]
    W = [[0]*7 for _ in range(4)]
    
    # Use the 2x2x2 Strassen-like decomposition extended to 2x3x2
    # Think of A as [A_left (2x2), A_right (2x1)] and B as [B_top (2x2), B_bottom (1x2)]
    # C = A_left * B_top + A_right * B_bottom
    # 
    # For A_left * B_top (2x2x2), use rank 7
    # For A_right * B_bottom (2x1x2), use rank 4
    # But we can share some structure.
    #
    # Let me use a direct construction.
    # 
    # Products:
    # p0 = (A00 + A01)(B00 + B10)  -> contributes to C00
    # p1 = (A00 + A01)(B01 + B11)  -> contributes to C01
    # p2 = (A10 + A11)(B00 + B10)  -> contributes to C10
    # p3 = (A10 + A11)(B01 + B11)  -> contributes to C11
    # p4 = A02 * B20               -> contributes to C00
    # p5 = A02 * B21               -> contributes to C01
    # p6 = A12 * B20 + A12 * B21 ... no, that's two products
    #
    # This gives 7 products but p6 would need to be two terms. Let me reconsider.
    #
    # Alternative: Use 8 products (schoolbook) and see if I can merge.
    # Actually, let me just use rank 8 for 2x3x2 if I can't find rank 7 easily.
    # But let me try harder.
    
    # Let me use a different approach. The 2x3x2 tensor can be seen as:
    # 3 terms for each of 4 outputs = 12 scalar products, but with sharing.
    # 
    # Known: R(2,3,2) = 7. Let me construct it explicitly.
    # 
    # Following the pattern of 2x2x2 rank 7:
    # For 2x2x2: C = [[a,b],[c,d]] * [[e,f],[g,h]]
    # p1 = (a+b)(e+f), p2 = (a+b)(g+h), p3 = (c+d)(e+g), p4 = (c+d)(f+h)
    # p5 = a(e+h), p6 = b(f+g), p7 = c(e+d) ... 
    # 
    # I'll use a verified construction. Let me just use rank 8 to be safe,
    # giving total rank 32 for 2x3x8.
    
    # Actually, let me just use schoolbook for 2x3x2 (rank 6) times 4 = 24, 
    # but that's worse than my earlier 28. Wait, 2x3x2 schoolbook is 6 products.
    # 4 blocks of 6 = 24. That's better than 28!
    
    # But can I do 2x3x2 in rank 7? If yes, 4*7=28. If schoolbook 6, 4*6=24.
    # Schoolbook for 2x3x2 is 6, which is less than 7. So rank 6 is achievable.
    # Is rank 6 optimal for 2x3x2? The tensor has 12 non-zero entries in a 4x6x6 space... 
    # Actually the tensor has 12 rank-1 terms in schoolbook. The rank could be lower.
    # But 6 is already quite low. Let me just use rank 6 per block.
    
    # 2x3x2 schoolbook: 6 rank-1 tensors
    # For block covering k in {2b, 2b+1}:
    for c in range(6):
        i = c // 3
        j = c % 3
        # k=0
        U[i*3 + j][c] = 1
        V[j*2 + 0][c] = 1
        W[0*2 + i][c] = 1  # k=0, i
        # Wait, I need separate columns for k=0 and k=1
        pass
    
    # Let me redo this properly.
    U = [[0]*6 for _ in range(6)]
    V = [[0]*6 for _ in range(6)]
    W = [[0]*6 for _ in range(4)]
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*3 + j][col] = 1
                V[j*2 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    # Use 4 blocks of 2x3x2, each with rank 6 (schoolbook)
    # Total rank: 4 * 6 = 24
    block_rank = 6
    num_blocks = p // 2  # 4
    total_rank = block_rank * num_blocks  # 24
    
    U = [[0] * total_rank for _ in range(n * m)]  # (6, 24)
    V = [[0] * total_rank for _ in range(m * p)]  # (24, 24)
    W = [[0] * total_rank for _ in range(p * n)]  # (16, 24)
    
    for block in range(num_blocks):
        if not _check_deadline():
            break
        # Get the 2x3x2 decomposition
        bU, bV, bW = _construct_2x3x2_rank7()
        
        for c in range(block_rank):
            global_col = block * block_rank + c
            # U: same indices (i*3+j)
            for row in range(n * m):
                U[row][global_col] = bU[row][c]
            
            # V: row j*p + k, where k is local (0 or 1) + block*2
            for j in range(m):
                for local_k in range(2):
                    v_row = j * p + (block * 2 + local_k)
                    local_v_row = j * 2 + local_k
                    V[v_row][global_col] = bV[local_v_row][c]
            
            # W: row k*n + i, where k is local + block*2
            for local_k in range(2):
                for i in range(n):
                    w_row = (block * 2 + local_k) * n + i
                    local_w_row = local_k * n + i
                    W[w_row][global_col] = bW[local_w_row][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
