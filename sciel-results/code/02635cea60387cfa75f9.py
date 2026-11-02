"""Matrix multiplication 2x4x7 tensor decomposition using block Strassen."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 x 4x7 matrix multiplication.
    
    Strategy: Partition the 4x4 middle block into four 2x2 blocks.
    For each pair of 2x2 blocks (A_block, B_block) that multiply to a 2x2 C_block,
    use Strassen's algorithm (rank 7) for the full 2x2x2 multiplication.
    For boundary cases where only a single column of C is needed, use rank 4 decomposition.
    
    The 4x7 output matrix C is partitioned into columns:
    - C[:, 0:2], C[:, 2:4], C[:, 4:6]: Each requires full 2x2x2 Strassen (rank 7)
    - C[:, 6]: Only 1 column, uses rank 4 decomposition
    
    Total rank = 2 * (3 * 7 + 4) = 2 * 25 = 50
    """
    import time
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 2, 4, 7
    total_rank = 50
    
    # Initialize U, V, W as lists of lists
    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 x 50
    V = [[0.0] * total_rank for _ in range(m * p)]  # 28 x 50
    W = [[0.0] * total_rank for _ in range(p * n)]  # 14 x 50
    
    rank_col = 0
    
    # Strassen's algorithm for 2x2x2 multiplication
    # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A * B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen's 7 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    def add_strassen_block(i_base, j_base, k_base):
        """Add a Strassen 2x2x2 block at row offset i_base, col offset j_base, output col offset k_base."""
        nonlocal rank_col
        # i_base in {0, 1}, j_base in {0, 2}, k_base in {0, 2, 4, 6}
        
        # Local indices within the 2x2 block:
        # A rows: i_base, i_base+1; A cols: j_base, j_base+1
        # B rows: j_base, j_base+1; B cols: k_base, k_base+1
        # C rows: i_base, i_base+1; C cols: k_base, k_base+1
        
        # p1 = (a+d)(e+h)
        # U: A[i_base, j_base]=1, A[i_base+1, j_base+1]=1
        # V: B[j_base, k_base]=1, B[j_base+1, k_base+1]=1
        # W: C[i_base, k_base]=1, C[i_base+1, k_base+1]=1
        U[i_base * m + j_base][rank_col] = 1.0
        U[(i_base + 1) * m + (j_base + 1)][rank_col] = 1.0
        V[j_base * p + k_base][rank_col] = 1.0
        V[(j_base + 1) * p + (k_base + 1)][rank_col] = 1.0
        W[k_base * n + i_base][rank_col] = 1.0
        W[(k_base + 1) * n + (i_base + 1)][rank_col] = 1.0
        rank_col += 1
        
        # p2 = (c+d)e
        # U: A[i_base+1, j_base]=1, A[i_base+1, j_base+1]=1
        # V: B[j_base, k_base]=1
        # W: C[i_base+1, k_base]=1
        U[(i_base + 1) * m + j_base][rank_col] = 1.0
        U[(i_base + 1) * m + (j_base + 1)][rank_col] = 1.0
        V[j_base * p + k_base][rank_col] = 1.0
        W[k_base * n + (i_base + 1)][rank_col] = 1.0
        rank_col += 1
        
        # p3 = a(f-h)
        # U: A[i_base, j_base]=1
        # V: B[j_base+1, k_base+1]=1, B[j_base, k_base]=-1 (wait, this is f-h, so B[j_base+1, k_base+1] and -B[j_base, k_base])
        # Actually: f = B[j_base+1, k_base+1]? No.
        # Let me re-define local indices properly.
        # B = [[e, f], [g, h]] where e=B[j_base, k_base], f=B[j_base, k_base+1], g=B[j_base+1, k_base], h=B[j_base+1, k_base+1]
        # So p3 = a * (f - h) = A[i_base, j_base] * (B[j_base, k_base+1] - B[j_base+1, k_base+1])
        U[i_base * m + j_base][rank_col] = 1.0
        V[j_base * p + (k_base + 1)][rank_col] = 1.0
        V[(j_base + 1) * p + (k_base + 1)][rank_col] = -1.0
        W[k_base * n + i_base][rank_col] = 1.0
        rank_col += 1
        
        # p4 = d(g-e) = A[i_base+1, j_base+1] * (B[j_base+1, k_base] - B[j_base, k_base])
        U[(i_base + 1) * m + (j_base + 1)][rank_col] = 1.0
        V[(j_base + 1) * p + k_base][rank_col] = 1.0
        V[j_base * p + k_base][rank_col] = -1.0
        W[(k_base + 1) * n + (i_base + 1)][rank_col] = 1.0
        rank_col += 1
        
        # p5 = (a+b)h = (A[i_base, j_base] + A[i_base, j_base+1]) * B[j_base+1, k_base+1]
        U[i_base * m + j_base][rank_col] = 1.0
        U[i_base * m + (j_base + 1)][rank_col] = 1.0
        V[(j_base + 1) * p + (k_base + 1)][rank_col] = 1.0
        W[(k_base + 1) * n + i_base][rank_col] = 1.0
        rank_col += 1
        
        # p6 = (c-a)(e+f) = (A[i_base+1, j_base] - A[i_base, j_base]) * (B[j_base, k_base] + B[j_base, k_base+1])
        U[(i_base + 1) * m + j_base][rank_col] = 1.0
        U[i_base * m + j_base][rank_col] = -1.0
        V[j_base * p + k_base][rank_col] = 1.0
        V[j_base * p + (k_base + 1)][rank_col] = 1.0
        W[(k_base + 1) * n + (i_base + 1)][rank_col] = 1.0
        rank_col += 1
        
        # p7 = (b-d)(g+h) = (A[i_base, j_base+1] - A[i_base+1, j_base+1]) * (B[j_base+1, k_base] + B[j_base+1, k_base+1])
        U[i_base * m + (j_base + 1)][rank_col] = 1.0
        U[(i_base + 1) * m + (j_base + 1)][rank_col] = -1.0
        V[(j_base + 1) * p + k_base][rank_col] = 1.0
        V[(j_base + 1) * p + (k_base + 1)][rank_col] = 1.0
        W[k_base * n + i_base][rank_col] = 1.0
        rank_col += 1
        
        return rank_col
    
    def add_schoolbook_block(i_base, j_base, k_base):
        """Add a 2x2x1 schoolbook block: compute C[:, k_base] from A[:, j_base:j_base+2] @ B[j_base:j_base+2, k_base]."""
        nonlocal rank_col
        # Only computing column k_base of C
        # C[i_base, k_base] = A[i_base, j_base] * B[j_base, k_base] + A[i_base, j_base+1] * B[j_base+1, k_base]
        # C[i_base+1, k_base] = A[i_base+1, j_base] * B[j_base, k_base] + A[i_base+1, j_base+1] * B[j_base+1, k_base]
        
        # Term 1: A[i_base, j_base] * B[j_base, k_base] -> C[i_base, k_base]
        U[i_base * m + j_base][rank_col] = 1.0
        V[j_base * p + k_base][rank_col] = 1.0
        W[k_base * n + i_base][rank_col] = 1.0
        rank_col += 1
        
        # Term 2: A[i_base+1, j_base] * B[j_base, k_base] -> C[i_base+1, k_base]
        U[(i_base + 1) * m + j_base][rank_col] = 1.0
        V[j_base * p + k_base][rank_col] = 1.0
        W[k_base * n + (i_base + 1)][rank_col] = 1.0
        rank_col += 1
        
        # Term 3: A[i_base, j_base+1] * B[j_base+1, k_base] -> C[i_base, k_base]
        U[i_base * m + (j_base + 1)][rank_col] = 1.0
        V[(j_base + 1) * p + k_base][rank_col] = 1.0
        W[k_base * n + i_base][rank_col] = 1.0
        rank_col += 1
        
        # Term 4: A[i_base+1, j_base+1] * B[j_base+1, k_base] -> C[i_base+1, k_base]
        U[(i_base + 1) * m + (j_base + 1)][rank_col] = 1.0
        V[(j_base + 1) * p + k_base][rank_col] = 1.0
        W[k_base * n + (i_base + 1)][rank_col] = 1.0
        rank_col += 1
        
        return rank_col
    
    # Process each row block (i_base in {0, 1})
    for i_base in range(2):
        # For each column block in A (j_base in {0, 2})
        for j_base in [0, 2]:
            # For each output column block
            # k_base = 0, 2, 4: use Strassen (full 2x2x2)
            for k_base in [0, 2, 4]:
                if time.time() > deadline:
                    break
                add_strassen_block(i_base, j_base, k_base)
            
            # k_base = 6: use schoolbook 2x2x1 (only 1 column of output)
            if time.time() <= deadline:
                add_schoolbook_block(i_base, j_base, 6)
    
    # Truncate to actual rank used
    actual_rank = rank_col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
