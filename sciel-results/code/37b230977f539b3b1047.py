"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Strategy: Split the 4x9 matrix B into B_left (4x4) and B_right (4x5).
  - A * B_left uses rank-49 Strassen for 4x4*4x4.
  - A * B_right is split further:
      * B_right has columns 0..3 (4x4) and column 4 (4x1).
      * A * B_right[0:4] uses rank-49 Strassen.
      * A * B_right[4:5] uses rank-16 schoolbook for 4x4*4x1.
  - Total rank: 49 + 49 + 16 = 114.

Actually, let's use a better decomposition:
  Split B (4x9) into three parts: B1 (4x3), B2 (4x3), B3 (4x3).
  For each 4x3 block, we can use a decomposition of rank 4*3 = 12? No, that's schoolbook.
  Better: use 4x4*4x4 Strassen for the first 8 columns (two 4x4 blocks) and 4x4*4x1 for the last column.
  Rank: 49 + 49 + 16 = 114.

Alternative: Can we do 4x9 as 4x4 + 4x4 + 4x1? Yes, that's what we have.
Is there a rank < 114 algorithm? 

Let's try a different block decomposition:
Split A (4x4) into A_top (2x4) and A_bottom (2x4).
Split B (4x9) into B_top (2x9) and B_bottom (2x9).

C_top (2x9) = A_top * B  => rank(2,4,9)
C_bottom (2x9) = A_bottom * B  => rank(2,4,9)

What is the rank of 2x4*4x9?
We can split B into B1 (4x4) and B2 (4x5).
2x4 * 4x4: rank? 
  2x2*2x2 is Strassen rank 7.
  2x4*4x4: Split A into A1 (2x2), A2 (2x2). Split B into B11 (2x4), B12 (2x4), B21 (2x4), B22 (2x4)? No.
  Let's use the fact that 2x4*4x4 = (2x2 * 2x4) + (2x2 * 2x4) where the two 2x2s are rows of A and the two 2x4s are column-blocks of B.
  Rank of 2x2 * 2x4: 
    Split 2x4 into 2x2 and 2x2.
    2x2 * 2x2 has rank 7 (Strassen).
    So 2x2 * 2x4 has rank 14? 
    Actually, 2x2 * 2x4 is a tensor of shape (2,2,4).
    T_{i,j,k} = A_{i,l} B_{l,k}.
    We can view this as two independent 2x2*2x2 multiplications if we split B into B[:,0:2] and B[:,2:4].
    So rank is 7 + 7 = 14.
  Therefore 2x4 * 4x4 has rank 14 + 14 = 28.

  Now 2x4 * 4x5:
    Split B into B1 (4x4) and B2 (4x1).
    2x4 * 4x4 has rank 28.
    2x4 * 4x1: 
      T_{i,j,k} = A_{i,l} B_{l,k} where k is only 0.
      This is a linear map. Rank is 2*4 = 8? 
      Actually, for 2x4 * 4x1, the output is 2x1.
      C_i = sum_l A_{i,l} B_{l,0}.
      This is just a bilinear form. The rank of the tensor T_{i,l,k} (with k fixed to 0, so really T_{i,l}) is 2*4 = 8? 
      Wait, the tensor is T_{i, l, k} where i in 2, l in 4, k in 1.
      T_{i,l,0} = A_{i,l} B_{l,0}.
      This is a rank-1 term for each (i,l). So 2*4 = 8 terms.
    So 2x4 * 4x5 has rank 28 + 8 = 36.

  Therefore 2x4 * 4x9 has rank 28 + 36 = 64.
  Total rank for 4x4 * 4x9 is 64 + 64 = 128.
  This is worse than 114.

Let's try splitting A into 1x4 + 1x4 + 1x4 + 1x4.
1x4 * 4x9: 
  T_{i,l,k} with i fixed. Rank is 4*9 = 36.
  Total 4*36 = 144. Worse.

Let's try splitting A into 3x4 + 1x4.
3x4 * 4x9:
  Split B into 4x4 and 4x5.
  3x4 * 4x4:
    Split A into 2x4 + 1x4.
    2x4 * 4x4 has rank 28 (as above).
    1x4 * 4x4:
      T_{i,l,k} with i fixed. Rank is 4*4 = 16.
    So 3x4 * 4x4 has rank 28 + 16 = 44.
  3x4 * 4x5:
    Split B into 4x4 and 4x1.
    3x4 * 4x4 has rank 44.
    3x4 * 4x1:
      T_{i,l,k} with k fixed. Rank is 3*4 = 12.
    So 3x4 * 4x5 has rank 44 + 12 = 56.
  Total 3x4 * 4x9 has rank 44 + 56 = 100.
  1x4 * 4x9 has rank 36.
  Total 4x4 * 4x9 has rank 100 + 36 = 136. Worse.

It seems 114 is the best we can do with simple block decompositions.
Let's stick with the 114 rank decomposition: two 4x4 Strassen blocks and one 4x4*4x1 schoolbook block.

Implementation:
1. Precompute the 49 terms for 4x4*4x4 Strassen.
2. Construct U, V, W for:
   - Block 1: A * B[:, 0:4] -> C[:, 0:4]  (49 terms)
   - Block 2: A * B[:, 4:8] -> C[:, 4:8]  (49 terms)
   - Block 3: A * B[:, 8:9] -> C[:, 8:9]  (16 terms)
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _build_strassen_4x4_terms():
    """Build the 49 rank-1 terms for Strassen 4x4*4x4."""
    terms = []
    
    # Strassen 2x2 meta-terms
    strassen_meta = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # 7 block-level Strassen multiplications
    block_mults = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0):1, (1,1):1}),
        ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0):1, (1,1):-1}),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1):1, (1,1):1}),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0):1, (1,0):1}),
        ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0):-1, (0,1):1}),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1):1}),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0):1}),
    ]
    
    for A_blocks, B_blocks, C_blocks in block_mults:
        for A_meta, B_meta, C_meta in strassen_meta:
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            for (li, lj), meta_c in A_meta.items():
                for (br, bc, block_c) in A_blocks:
                    gi = br * 2 + li
                    gj = bc * 2 + lj
                    u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + meta_c * block_c
            
            for (lj, lk), meta_c in B_meta.items():
                for (br, bc, block_c) in B_blocks:
                    gj = br * 2 + lj
                    gk = bc * 2 + lk
                    v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + meta_c * block_c
            
            for (li, lk), meta_c in C_meta.items():
                for (br, bc), block_c in C_blocks.items():
                    gi = br * 2 + li
                    gk = bc * 2 + lk
                    w_dict[(gi, gk)] = w_dict.get((gi, gk), 0) + meta_c * block_c
            
            terms.append((u_dict, v_dict, w_dict))
    
    return terms

_STRASSEN_4X4_TERMS = _build_strassen_4x4_terms()

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add 49 Strassen terms for a 4x4*4x4 block."""
        for u_dict, v_dict, w_dict in _STRASSEN_4X4_TERMS:
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            
            for (gi, gj), coeff in u_dict.items():
                u_vec[(A_base_i + gi) * m + (A_base_j + gj)] = coeff
            
            for (gj, gk), coeff in v_dict.items():
                v_vec[(B_base_j + gj) * p + (B_base_k + gk)] = coeff
            
            for (gi, gk), coeff in w_dict.items():
                w_vec[(C_base_k + gk) * n + (C_base_i + gi)] = coeff
            
            add_term(u_vec, v_vec, w_vec)
    
    # Block 1: A * B[:, 0:4] -> C[:, 0:4]
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Block 2: A * B[:, 4:8] -> C[:, 4:8]
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Block 3: A * B[:, 8:9] -> C[:, 8:9] (schoolbook, 16 terms)
    for i in range(n):
        for j in range(m):
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    for col in range(rank):
        for row in range(n*m):
            U[row, col] = U_cols[col][row]
        for row in range(m*p):
            V[row, col] = V_cols[col][row]
            W[row, col] = W_cols[col][row]
    
    # Verification
    if time.time() < deadline - 10:
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        valid = np.allclose(T, 1.0, atol=1e-10)
        
        if not valid:
            # Fallback to schoolbook
            U_cols.clear()
            V_cols.clear()
            W_cols.clear()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_vec = [0] * (n*m)
                        v_vec = [0] * (m*p)
                        w_vec = [0] * (p*n)
                        u_vec[i * m + j] = 1
                        v_vec[j * p + k] = 1
                        w_vec[k * n + i] = 1
                        add_term(u_vec, v_vec, w_vec)
            
            rank = len(U_cols)
            U = np.zeros((n*m, rank), dtype=int)
            V = np.zeros((m*p, rank), dtype=int)
            W = np.zeros((p*n, rank), dtype=int)
            for col in range(rank):
                for row in range(n*m):
                    U[row, col] = U_cols[col][row]
                for row in range(m*p):
                    V[row, col] = V_cols[col][row]
                    W[row, col] = W_cols[col][row]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
