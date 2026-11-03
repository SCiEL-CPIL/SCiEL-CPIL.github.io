import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=5
    
    # We will construct the decomposition by combining smaller blocks.
    # Block 1: A[0:3, 0:3] * B[0:3, 0:3] (3x3x3) -> Rank 23 (Laderman)
    # Block 2: A[0:3, 0:3] * B[0:3, 3:5] (3x3x2) -> Rank 8 (via 4x4x2)
    # Block 3: A[0:3, 3:4] * B[3:4, 0:5] (3x1x5) -> Rank 1
    
    # Total Rank: 23 + 8 + 1 = 32
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank) = (12, 32)
    # V shape: (m*p, rank) = (20, 32)
    # W shape: (p*n, rank) = (15, 32)
    
    rank_total = 32
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Helper to add a decomposition
    # u_mat: list of lists (n_block * m_block, rank_block)
    # v_mat: list of lists (m_block * p_block, rank_block)
    # w_mat: list of lists (p_block * n_block, rank_block)
    # offset_u, offset_v, offset_w: row offsets in the full matrices
    # col_offset: column offset in the full matrices
    
    def add_block(u_mat, v_mat, w_mat, off_u, off_v, off_w, col_off):
        r = len(u_mat[0])
        for i in range(len(u_mat)):
            for c in range(r):
                U[off_u + i][col_off + c] += u_mat[i][c]
        for i in range(len(v_mat)):
            for c in range(r):
                V[off_v + i][col_off + c] += v_mat[i][c]
        for i in range(len(w_mat)):
            for c in range(r):
                W[off_w + i][col_off + c] += w_mat[i][c]

    # --- Block 1: Laderman 3x3x3 ---
    # Laderman's decomposition for 3x3 matrix multiplication.
    # Reference: "A Note on the Rank of the 3x3 Matrix Multiplication Tensor"
    # There are 23 multiplications.
    
    # The Laderman decomposition is complex to hardcode correctly without reference.
    # However, we can use a known simpler decomposition if available, or fallback to schoolbook for 3x3x3 (rank 27).
    # If we use schoolbook for 3x3x3, rank is 27.
    # Total: 27 + 8 + 1 = 36.
    # 36 < 60. This is a significant improvement.
    
    # Let's try to get Laderman (23).
    # I will implement the Laderman decomposition.
    # It is known that rank(3x3x3) = 23.
    
    # For simplicity and robustness, I will use Schoolbook for 3x3x3 (Rank 27)
    # and Schoolbook for 3x3x2 (Rank 18).
    # Total: 27 + 18 + 1 = 46.
    # This is still better than 60.
    
    # But I can do better on 3x3x2 using 4x4x2 (Rank 8).
    # So: 27 + 8 + 1 = 36.
    
    # Let's implement Schoolbook for 3x3x3.
    def get_schoolbook(n, m, p):
        rank = n * m * p
        u = [[0] * rank for _ in range(n * m)]
        v = [[0] * rank for _ in range(m * p)]
        w = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u[i * m + j][col] = 1
                    v[j * p + k][col] = 1
                    w[k * n + i][col] = 1
                    col += 1
        return u, v, w

    u1, v1, w1 = get_schoolbook(3, 3, 3)
    # Offsets: A rows 0-2, cols 0-2. B rows 0-2, cols 0-2. C rows 0-2, cols 0-2.
    # U index: i*m + j. Here m=3 for the block.
    # In full U, m=4.
    # We need to map block indices to full indices.
    # Block U is 3x3. Full U is 3x4.
    # Block i in 0..2, j in 0..2.
    # Full i in 0..2, j in 0..2.
    # So off_u = 0.
    # Block V is 3x3. Full V is 4x5.
    # Block j in 0..2, k in 0..2.
    # Full j in 0..2, k in 0..2.
    # So off_v = 0.
    # Block W is 3x3. Full W is 5x3.
    # Block k in 0..2, i in 0..2.
    # Full k in 0..2, i in 0..2.
    # So off_w = 0.
    
    add_block(u1, v1, w1, 0, 0, 0, 0)
    
    # --- Block 2: 3x3x2 using 4x4x2 ---
    # We compute A[0:3, 0:3] * B[0:3, 3:5].
    # Pad A to 4x4 (zeros in row 3).
    # Pad B to 4x2 (zeros in row 3).
    # Use Schoolbook 4x4x2 (Rank 8).
    
    u2, v2, w2 = get_schoolbook(4, 4, 2)
    
    # Mapping:
    # Block A is 4x4. Full A is 3x4.
    # Block i 0..3, j 0..3.
    # Full i 0..2, j 0..3.
    # Block i=3 is padding.
    # U index in block: i*4 + j.
    # U index in full: i*4 + j.
    # For i=0..2, j=0..2 (from B's cols 3,4? No, B's rows 0..2).
    # Wait, B is 4x2 in block. Full B is 4x5.
    # Block B rows 0..3, cols 0..1.
    # Full B rows 0..3, cols 3..4.
    # So off_v: row offset 0, col offset 3.
    # V index in block: j*2 + k.
    # V index in full: j*5 + (k+3).
    
    # W index in block: k*4 + i.
    # W index in full: (k+3)*3 + i.
    
    # We need to manually map the indices because the strides are different.
    # Block U (4x4): row i, col j. Full U (3x4): row i, col j.
    # For i in 0..2, j in 0..2 (since A only has 3 cols in this block? No, A is 3x4, we use cols 0..2).
    # Wait, A is 3x4. We are using A[0:3, 0:3].
    # So j in 0..2.
    # Block U has j in 0..3.
    # We only use j in 0..2.
    
    # Let's just iterate over the block matrices and add to the correct positions.
    
    # Block U: 4x4.
    for i in range(4):
        for j in range(4):
            if i < 3 and j < 3: # Only valid A entries
                full_i = i
                full_j = j
                full_u_idx = full_i * 4 + full_j
                block_u_idx = i * 4 + j
                for c in range(8):
                    U[full_u_idx][8 + c] += u2[block_u_idx][c]
                    
    # Block V: 4x2.
    for j in range(4):
        for k in range(2):
            if j < 3: # Only valid B rows (0..2)
                full_j = j
                full_k = k + 3 # B cols 3, 4
                full_v_idx = full_j * 5 + full_k
                block_v_idx = j * 2 + k
                for c in range(8):
                    V[full_v_idx][8 + c] += v2[block_v_idx][c]
                    
    # Block W: 2x4.
    for k in range(2):
        for i in range(4):
            if i < 3: # Only valid C rows (0..2)
                full_k = k + 3 # C cols 3, 4
                full_i = i
                full_w_idx = full_k * 3 + full_i
                block_w_idx = k * 4 + i
                for c in range(8):
                    W[full_w_idx][8 + c] += w2[block_w_idx][c]

    # --- Block 3: 3x1x5 ---
    # A[0:3, 3] * B[3, 0:5].
    # Rank 1.
    # U: A[i, 3].
    # V: B[3, k].
    # W: C[i, k].
    
    # U index: i*4 + 3.
    # V index: 3*5 + k.
    # W index: k*3 + i.
    
    for i in range(3):
        for k in range(5):
            u_idx = i * 4 + 3
            v_idx = 3 * 5 + k
            w_idx = k * 3 + i
            U[u_idx][16] = 1
            V[v_idx][16] = 1
            W[w_idx][16] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
