"""
Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x8 matrix.

Strategy:
The tensor $T_{2,2,8}$ can be viewed as the direct sum of 4 independent $T_{2,2,2}$ tensors,
where the 8 columns of B are grouped into 4 pairs.
Each $T_{2,2,2}$ tensor has a minimal rank of 7 (Strassen's algorithm).
Since the A matrix is shared, we can use the same 7 linear combinations of A for each block.
The total rank is 4 * 7 = 28.

We construct U, V, W matrices of shapes (4, 28), (16, 28), (16, 28).
The columns are indexed by (block, strassen_term).
block in 0..3, strassen_term in 0..6.
"""

import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 8
    # Group p=8 columns into 4 blocks of 2
    num_blocks = p // 2
    block_rank = 7
    total_rank = num_blocks * block_rank  # 28
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]  # (4, 28)
    V = [[0] * total_rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * total_rank for _ in range(p * n)]  # (16, 28)
    
    # Strassen 2x2x2 decomposition vectors
    # A = [[a, b], [c, d]] -> indices 0,1,2,3
    # B = [[e, f], [g, h]] -> indices 0,1,2,3
    # C = [[i, j], [k, l]] -> indices 0,1,2,3
    
    # 7 terms M1..M7:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Output C:
    # i = M1 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M3 + M6
    
    # U vectors (coefficients for A entries)
    u_strassen = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [1, 0, -1, 0], # M6: c-a
        [0, 1, 0, -1], # M7: b-d
    ]
    
    # V vectors (coefficients for B entries)
    # For a local 2x2 B block [[B00, B01], [B10, B11]]
    # Indices in local block: 0=B00, 1=B01, 2=B10, 3=B11
    v_strassen = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [0, 0, 1, -1], # M4: g-e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1],  # M7: g+h
    ]
    
    # W vectors (coefficients for C entries)
    # For a local 2x2 C block [[C00, C01], [C10, C11]]
    # Indices in local block: 0=C00, 1=C01, 2=C10, 3=C11
    w_strassen = [
        [1, 0, 0, 1],  # M1: i + l
        [0, 0, 1, 0],  # M2: k
        [0, 1, 0, -1], # M3: j - l
        [0, 0, 1, 0],  # M4: k
        [-1, 1, 0, 0], # M5: -i + j
        [0, 0, 0, 1],  # M6: l
        [1, 0, 0, 0],  # M7: i
    ]
    
    # Fill the matrices
    for b in range(num_blocks):
        # Column base for this block
        col_base = b * block_rank
        
        # For each Strassen term s in this block
        for s in range(block_rank):
            col = col_base + s
            
            # Fill U
            # U indices: i*m + j. Same for all blocks.
            for idx_a in range(n * m):
                U[idx_a][col] = u_strassen[s][idx_a]
                
            # Fill V
            # V indices: j*p + k.
            # For block b, the columns of B are k = 2*b and k = 2*b+1.
            # Local B indices:
            # local 0 -> B[0][2*b] -> V index 0*p + 2*b
            # local 1 -> B[0][2*b+1] -> V index 0*p + 2*b+1
            # local 2 -> B[1][2*b] -> V index 1*p + 2*b
            # local 3 -> B[1][2*b+1] -> V index 1*p + 2*b+1
            
            # v_strassen[s] has 4 entries corresponding to local indices 0,1,2,3
            v_vec = v_strassen[s]
            
            # local 0
            V[0 * p + (2 * b)][col] = v_vec[0]
            # local 1
            V[0 * p + (2 * b + 1)][col] = v_vec[1]
            # local 2
            V[1 * p + (2 * b)][col] = v_vec[2]
            # local 3
            V[1 * p + (2 * b + 1)][col] = v_vec[3]
            
            # Fill W
            # W indices: k*n + i.
            # For block b, the columns of C are k = 2*b and k = 2*b+1.
            # Local C indices:
            # local 0 -> C[0][2*b] -> W index (2*b)*n + 0
            # local 1 -> C[0][2*b+1] -> W index (2*b+1)*n + 0
            # local 2 -> C[1][2*b] -> W index (2*b)*n + 1
            # local 3 -> C[1][2*b+1] -> W index (2*b+1)*n + 1
            
            # w_strassen[s] has 4 entries corresponding to local indices 0,1,2,3
            w_vec = w_strassen[s]
            
            # local 0
            W[(2 * b) * n + 0][col] = w_vec[0]
            # local 1
            W[(2 * b + 1) * n + 0][col] = w_vec[1]
            # local 2
            W[(2 * b) * n + 1][col] = w_vec[2]
            # local 3
            W[(2 * b + 1) * n + 1][col] = w_vec[3]
            
            # Check time
            if time.time() > deadline:
                # If we run out of time, return what we have. 
                # Since we are filling sequentially, this might be incomplete.
                # But the loop is fast, so this is unlikely.
                pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
