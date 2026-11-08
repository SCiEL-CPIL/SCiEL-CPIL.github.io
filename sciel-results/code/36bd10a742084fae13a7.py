import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x3 by 3x7 matrix multiplication.
    
    This implementation uses a structured block-decomposition approach.
    It defines the multiplication tensor T[i,j,k] = sum_m A[i,m] B[m,k].
    We decompose the index ranges into blocks and apply known optimal decompositions
    (Strassen for 2x2x2, Schoolbook for smaller/rectangular blocks) to each block.
    
    Strategy:
    - Dimensions: n=2, m=3, p=7.
    - We partition the 'j' dimension (size 3) into [2, 1].
    - For the j-block of size 2 (indices 0,1):
        - We partition the 'k' dimension (size 7) into [2, 2, 2, 1].
        - For each 2x2 block in k, we use Strassen (rank 7).
        - For the 2x1 block in k, we use Schoolbook (rank 4).
    - For the j-block of size 1 (index 2):
        - We partition the 'k' dimension (size 7) into [1, 1, 1, 1, 1, 1, 1] or just treat as 7.
        - Actually, it's better to treat the remaining part as a single block if not split further, 
          or split into 1x1s. Schoolbook for 2x1x1 is rank 2. 
          Wait, A is 2x3, B is 3x7.
          C = A * B.
          C[i,k] = sum_j A[i,j] B[j,k].
          
          Let's stick to the proven rank 39 structure but implement it via a generic engine.
          
          Blocks:
          J-split: [2, 1]
          K-split: [2, 2, 2, 1] for the first J-block? No, K is global.
          
          Actually, the most efficient way is to view the tensor as a sum of outer products of structured matrices.
          
          Let's implement a simple recursive block solver that returns the U, V, W contributions.
    """
    start_time = time.time()
    
    n, m, p = 2, 3, 7
    
    # We will build the matrices U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Indexing conventions:
    # U[i, j, t] corresponds to A[i, j]
    # V[j, k, t] corresponds to B[j, k]
    # W[k, i, t] corresponds to C[k, i] (Note: Task says W[k*n+i, r], so row is k*n+i)
    
    # We will accumulate terms.
    # Each term is a rank-1 tensor decomposition:
    # T = sum_t u_t \otimes v_t \otimes w_t
    # where u_t is in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}.
    
    terms = []
    
    def add_strassen_block(i_start, j_start, k_start):
        """
        Adds Strassen decomposition for a 2x2x2 block.
        A block is A[i_start:i_start+2, j_start:j_start+2] * B[j_start:j_start+2, k_start:k_start+2]
        Producing C[i_start:i_start+2, k_start:k_start+2]
        """
        # Indices in U (n*m)
        a00 = (i_start) * m + (j_start)
        a01 = (i_start) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start)
        a11 = (i_start + 1) * m + (j_start + 1)
        
        # Indices in V (m*p)
        b00 = (j_start) * p + (k_start)
        b01 = (j_start) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start)
        b11 = (j_start + 1) * p + (k_start + 1)
        
        # Indices in W (p*n), row index = k*n + i
        c00 = (k_start) * n + (i_start)
        c01 = (k_start) * n + (i_start + 1)
        c10 = (k_start + 1) * n + (i_start)
        c11 = (k_start + 1) * n + (i_start + 1)
        
        def make_u(indices, signs):
            vec = [0.0] * (n * m)
            for idx, s in zip(indices, signs):
                vec[idx] = s
            return vec

        def make_v(indices, signs):
            vec = [0.0] * (m * p)
            for idx, s in zip(indices, signs):
                vec[idx] = s
            return vec

        def make_w(indices, signs):
            vec = [0.0] * (p * n)
            for idx, s in zip(indices, signs):
                vec[idx] = s
            return vec

        # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        u1 = make_u([a00, a11], [1, 1])
        v1 = make_v([b00, b11], [1, 1])
        w1 = make_w([c00, c11], [1, 1])
        terms.append((u1, v1, w1))
        
        # M2 = (A10 + A11)(B00) -> C10 += M2, C11 -= M2
        u2 = make_u([a10, a11], [1, 1])
        v2 = make_v([b00], [1])
        w2 = make_w([c10, c11], [1, -1])
        terms.append((u2, v2, w2))
        
        # M3 = (A00)(B01 - B11) -> C01 += M3, C11 -= M3
        u3 = make_u([a00], [1])
        v3 = make_v([b01, b11], [1, -1])
        w3 = make_w([c01, c11], [1, -1])
        terms.append((u3, v3, w3))
        
        # M4 = (A11)(B10 - B00) -> C00 += M4, C10 -= M4
        u4 = make_u([a11], [1])
        v4 = make_v([b10, b00], [1, -1])
        w4 = make_w([c00, c10], [1, -1])
        terms.append((u4, v4, w4))
        
        # M5 = (A00 + A01)(B11) -> C00 -= M5, C01 += M5
        u5 = make_u([a00, a01], [1, 1])
        v5 = make_v([b11], [1])
        w5 = make_w([c00, c01], [-1, 1])
        terms.append((u5, v5, w5))
        
        # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
        u6 = make_u([a10, a00], [1, -1])
        v6 = make_v([b00, b01], [1, 1])
        w6 = make_w([c11], [1])
        terms.append((u6, v6, w6))
        
        # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
        u7 = make_u([a01, a11], [1, -1])
        v7 = make_v([b10, b11], [1, 1])
        w7 = make_w([c00], [1])
        terms.append((u7, v7, w7))

    def add_schoolbook_block(i_start, j_start, k_start, ni, nj, nk):
        """
        Adds Schoolbook decomposition for an ni x nj x nk block.
        Rank = ni * nj * nk.
        """
        for di in range(ni):
            for dj in range(nj):
                for dk in range(nk):
                    i_idx = i_start + di
                    j_idx = j_start + dj
                    k_idx = k_start + dk
                    
                    u_idx = i_idx * m + j_idx
                    v_idx = j_idx * p + k_idx
                    w_idx = k_idx * n + i_idx
                    
                    u_vec = [0.0] * (n * m)
                    u_vec[u_idx] = 1.0
                    v_vec = [0.0] * (m * p)
                    v_vec[v_idx] = 1.0
                    w_vec = [0.0] * (p * n)
                    w_vec[w_idx] = 1.0
                    
                    terms.append((u_vec, v_vec, w_vec))

    # --- Structural Plan ---
    # n=2, m=3, p=7
    
    # Split J (m=3) into J1=[0,1] (size 2) and J2=[2] (size 1)
    # For J1 (j=0,1):
    #   Split K (p=7) into K1=[0,1] (size 2), K2=[2,3] (size 2), K3=[4,5] (size 2), K4=[6] (size 1)
    #   For K1: Block is i=[0,1], j=[0,1], k=[0,1] -> 2x2x2 Strassen
    #   For K2: Block is i=[0,1], j=[0,1], k=[2,3] -> 2x2x2 Strassen
    #   For K3: Block is i=[0,1], j=[0,1], k=[4,5] -> 2x2x2 Strassen
    #   For K4: Block is i=[0,1], j=[0,1], k=[6]   -> 2x2x1 Schoolbook
    # For J2 (j=2):
    #   K is still [0..6]. 
    #   We can treat this as blocks of i=[0,1], j=[2], k=[0..6].
    #   Since j-size is 1, it's effectively two independent 1x1x7 multiplications for i=0 and i=1?
    #   No, A[i, 2] * B[2, k].
    #   This is just standard multiplication.
    #   Block: i=[0,1], j=[2], k=[0..6]. Size 2x1x7.
    #   Schoolbook rank = 2 * 1 * 7 = 14.
    
    # Apply Strassen blocks
    add_strassen_block(0, 0, 0) # i=0, j=0, k=0
    add_strassen_block(0, 0, 2) # i=0, j=0, k=2
    add_strassen_block(0, 0, 4) # i=0, j=0, k=4
    
    # Apply Schoolbook block for k=6, j=[0,1]
    # i=[0,1], j=[0,1], k=[6]
    add_schoolbook_block(0, 0, 6, 2, 2, 1)
    
    # Apply Schoolbook block for j=2, k=[0..6]
    # i=[0,1], j=[2], k=[0..6]
    add_schoolbook_block(0, 2, 0, 2, 1, 7)
    
    # --- Assemble Matrices ---
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for t, (u_vec, v_vec, w_vec) in enumerate(terms):
        for i in range(n * m):
            if u_vec[i] != 0:
                U[i][t] = u_vec[i]
        for j in range(m * p):
            if v_vec[j] != 0:
                V[j][t] = v_vec[j]
        for k in range(p * n):
            if w_vec[k] != 0:
                W[k][t] = w_vec[k]
                
    # Convert to standard Python lists of numbers (int/float)
    # The task accepts integer/half-integer. Our construction uses 1, -1, 0.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
