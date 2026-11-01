import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2
    # A is 2x2, B is 2x2
    # U_2: (4, 7), V_2: (4, 7), W_2: (4, 7)
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U_2 = np.array([
        [1, 1, 0, 0, 1, 0, 0], # a
        [0, 0, 1, 1, 0, 0, 1], # b
        [0, 1, 1, 0, 0, 1, 0], # c
        [1, 0, 0, 1, 0, 0, -1]  # d
    ])
    
    V_2 = np.array([
        [1, 0, 0, 0, 0, 1, 0], # e
        [0, 0, 1, 0, 1, 0, 0], # f
        [0, 1, 0, 1, 0, 0, 1], # g
        [1, 0, -1, 0, 0, 0, 1]  # h
    ])
    
    W_2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ])
    
    # Standard decomposition for 2x1
    # A is 2x2, B is 2x1
    # U_1: (4, 4), V_1: (2, 4), W_1: (2, 4)
    # A indices: 0, 1, 2, 3
    # B indices: 0, 1
    # C indices: 0, 1
    
    U_1 = np.array([
        [1, 0, 0, 0], # a
        [0, 1, 0, 0], # b
        [0, 0, 1, 0], # c
        [0, 0, 0, 1]  # d
    ])
    
    V_1 = np.array([
        [1, 0, 1, 0], # e (B0)
        [0, 1, 0, 1]  # f (B1)
    ])
    
    W_1 = np.array([
        [1, 1, 0, 0], # C0
        [0, 0, 1, 1]  # C1
    ])
    
    # Construct full decomposition
    # p = 15 = 7 * 2 + 1
    num_strassen = 7
    num_standard = 1
    
    rank_total = num_strassen * 7 + num_standard * 4
    
    # U shape: (n*m, rank_total) = (4, rank_total)
    # V shape: (m*p, rank_total) = (2*15, rank_total) = (30, rank_total)
    # W shape: (p*n, rank_total) = (15*2, rank_total) = (30, rank_total)
    
    U_full = np.zeros((4, rank_total))
    V_full = np.zeros((30, rank_total))
    W_full = np.zeros((30, rank_total))
    
    col_offset = 0
    b_col_offset = 0
    
    for i in range(num_strassen):
        # Place Strassen block
        U_full[:, col_offset:col_offset+7] = U_2
        # V_2 is 4x7, place in rows b_col_offset to b_col_offset+3
        V_full[b_col_offset:b_col_offset+4, col_offset:col_offset+7] = V_2
        # W_2 is 4x7, place in rows b_col_offset to b_col_offset+3 (since p*n indexing is k*n+i, and k is 0..1 for this block)
        # Wait, W index is k*n + i.
        # For the i-th Strassen block, the columns of B are 2*i and 2*i+1.
        # So k ranges from 2*i to 2*i+1.
        # The rows of W corresponding to this block are 2*i*n to 2*i*n + n - 1.
        # n=2, so rows 4*i to 4*i+1? No.
        # W index: k*n + i.
        # k is the column index of B (0 to p-1).
        # i is the row index of A (0 to n-1).
        # So for a specific block of B columns $k \in [2i, 2i+1]$, the W rows are $2i \cdot 2 + 0$ and $2i \cdot 2 + 1$.
        # i.e., $4i$ and $4i+1$.
        # But W_2 is 4x7. The rows of W_2 correspond to C indices 00, 01, 10, 11.
        # C index is $k \cdot n + i_{row}$.
        # So row 0 of W_2 corresponds to $k=0, i_{row}=0$.
        # Row 1 of W_2 corresponds to $k=0, i_{row}=1$.
        # Row 2 of W_2 corresponds to $k=1, i_{row}=0$.
        # Row 3 of W_2 corresponds to $k=1, i_{row}=1$.
        # So we need to map W_2 rows to W_full rows.
        # W_full row for $k, i_{row}$ is $k \cdot n + i_{row}$.
        # For block i, $k$ is $2i$ and $2i+1$.
        # So W_full rows are $2i \cdot 2 + 0 = 4i$, $2i \cdot 2 + 1 = 4i+1$, $(2i+1) \cdot 2 + 0 = 4i+2$, $(2i+1) \cdot 2 + 1 = 4i+3$.
        
        W_full[4*i:4*i+4, col_offset:col_offset+7] = W_2
        
        col_offset += 7
        b_col_offset += 4 # B has 4 entries per 2x2 block (2 rows, 2 cols)
        
    # Place Standard block for the last column
    # B column index is 14.
    # U is same for all blocks (A is shared)
    U_full[:, col_offset:col_offset+4] = U_1
    
    # V_1 is 2x4. B indices for this block are row 0 and 1 of the 2x1 block.
    # The B matrix is 2x15.
    # The last column is index 14.
    # V_full rows correspond to $j \cdot p + k$.
    # j is row of B (0, 1). k is col of B (14).
    # So rows are $0 \cdot 15 + 14 = 14$ and $1 \cdot 15 + 14 = 29$.
    # V_1 row 0 corresponds to B[0, 14].
    # V_1 row 1 corresponds to B[1, 14].
    
    V_full[14, col_offset:col_offset+4] = V_1[0]
    V_full[29, col_offset:col_offset+4] = V_1[1]
    
    # W_1 is 2x4. C indices are $k \cdot n + i$.
    # k=14. i=0, 1.
    # Rows are $14 \cdot 2 + 0 = 28$ and $14 \cdot 2 + 1 = 29$.
    
    W_full[28, col_offset:col_offset+4] = W_1[0]
    W_full[29, col_offset:col_offset+4] = W_1[1]
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
