"""
5x5 matrix multiplication with reduced rank using Strassen for 2x2 blocks.

Strategy: Partition 5x5 matrices into blocks:
- 2x2, 2x2, 1x1 in rows
- 2x2, 2x2, 1x1 in columns

This gives a 3x3 block structure with block sizes (2,2,1).
For each block position (i_block, j_block, k_block), we compute:
  C[i_block, k_block] += A[i_block, j_block] * B[j_block, k_block]

The blocks are:
- 2x2 * 2x2 -> use Strassen (rank 7)
- 2x2 * 1x1 -> schoolbook (rank 2)
- 1x1 * 2x2 -> schoolbook (rank 2)
- 1x1 * 1x1 -> schoolbook (rank 1)

Total rank calculation:
- 2x2*2x2 blocks: positions where both row-block and col-block are size 2
  Row blocks of size 2: indices 0,1 (covering rows 0-3)
  Col blocks of size 2: indices 0,1 (covering cols 0-3)
  So 2x2 positions: (0,0), (0,1), (1,0), (1,1) -> 4 blocks * rank 7 = 28
  
- 2x2*1x1: row-block size 2, col-block size 1
  Row blocks size 2: 0,1; Col block size 1: 2
  Positions: (0,2), (1,2) -> 2 blocks * rank 2 = 4

- 1x1*2x2: row-block size 1, col-block size 2
  Row block size 1: 2; Col blocks size 2: 0,1
  Positions: (2,0), (2,1) -> 2 blocks * rank 2 = 4

- 1x1*1x1: position (2,2) -> rank 1 = 1

Total rank = 28 + 4 + 4 + 1 = 37

Wait, let me reconsider. The block structure for matrix multiplication:
C = A * B where A is 5x5, B is 5x5, C is 5x5.

Partition rows of A and C: r1=2, r2=2, r3=1 (total 5)
Partition columns of A and rows of B: c1=2, c2=2, c3=1 (total 5)
Partition columns of B and C: d1=2, d2=2, d3=1 (total 5)

So A is partitioned as:
  A = [[A11(2x2), A12(2x2), A13(2x1)],
       [A21(2x2), A22(2x2), A23(2x1)],
       [A31(1x2), A32(1x2), A33(1x1)]]

B is partitioned as:
  B = [[B11(2x2), B12(2x2), B13(2x1)],
       [B21(2x2), B22(2x2), B23(2x1)],
       [B31(1x2), B32(1x2), B33(1x1)]]

C is partitioned as:
  C = [[C11(2x2), C12(2x2), C13(2x1)],
       [C21(2x2), C22(2x2), C23(2x1)],
       [C31(1x2), C32(1x2), C33(1x1)]]

C[i][j] = sum_k A[i][k] * B[k][j]

For each (i,j) pair of block indices:
  C[i][j] = sum over k: A[i][k] * B[k][j]

The rank contribution for each (i,j,k) triple depends on the sizes:
- If A[i][k] is ai x ak and B[k][j] is ak x bj, the schoolbook rank is ai*ak*bj
- If ai=ak=2, bj=2: Strassen rank 7 vs schoolbook 8
- etc.

Let me compute more carefully:

Block sizes: rows = [2,2,1], cols = [2,2,1], depth = [2,2,1]

For each (i, k, j) where i is row block, k is column/depth block, j is output column block:
  A[i][k] has shape (rows[i] x cols[k]) = (r_i x c_k)
  B[k][j] has shape (cols[k] x rows... wait no.

Actually B is 5x5, partitioned with the same col/row sizes.
B[k][j] has shape (c_k x r_j) where c_k is the "column" partition of B's rows... 

Let me use standard notation:
- A is m x n, B is n x p, C is m x p
- Here m=n=p=5
- Partition m into [2,2,1], n into [2,2,1], p into [2,2,1]

A[i][k] has shape (m_i x n_k)
B[k][j] has shape (n_k x p_j)
C[i][j] has shape (m_i x p_j)

For each (i,k,j):
  schoolbook rank = m_i * n_k * p_j
  If m_i=2, n_k=2, p_j=2: use Strassen rank 7

Let's enumerate:
m_i values: [2,2,1]
n_k values: [2,2,1]  
p_j values: [2,2,1]

For each (i,k,j), rank = min(schoolbook, strassen if applicable)

Strassen applies when all three dimensions are 2 (i.e., 2x2 * 2x2).

Cases:
- (2,2,2): 4 combos for (i,k) where m_i=2,n_k=2, times 2 for p_j=2 → wait no
  i in {0,1} (m_i=2), k in {0,1} (n_k=2), j in {0,1} (p_j=2)
  That's 2*2*2 = 8 triples, each with rank 7 instead of 8
  Saving: 8 * (8-7) = 8

- (2,2,1): i in {0,1}, k in {0,1}, j in {2}
  2*2*1 = 4 triples, rank = 2*2*1 = 4 (no Strassen benefit since p_j=1)
  
- (2,1,2): i in {0,1}, k in {2}, j in {0,1}
  2*1*2 = 4 triples, rank = 2*1*2 = 4

- (2,1,1): i in {0,1}, k in {2}, j in {2}
  2*1*1 = 2 triples, rank = 2*1*1 = 2

- (1,2,2): i in {2}, k in {0,1}, j in {0,1}
  1*2*2 = 4 triples, rank = 1*2*2 = 4

- (1,2,1): i in {2}, k in {0,1}, j in {2}
  1*2*1 = 2 triples, rank = 1*2*1 = 2

- (1,1,2): i in {2}, k in {2}, j in {0,1}
  1*1*2 = 2 triples, rank = 1*1*2 = 2

- (1,1,1): i in {2}, k in {2}, j in {2}
  1*1*1 = 1 triple, rank = 1*1*1 = 1

Total schoolbook rank = 8*8 + 4*4 + 4*4 + 2*2 + 4*4 + 2*2 + 2*2 + 1*1
= 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125 ✓

With Strassen on the (2,2,2) cases:
Total rank = 8*7 + 4*4 + 4*4 + 2*2 + 4*4 + 2*2 + 2*2 + 1*1
= 56 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 117

So rank = 117.
"""

import time

DIMENSIONS = [5, 5, 5]

def construct():
    """Construct rank-117 tensor decomposition for 5x5 matrix multiplication
    using Strassen's algorithm for 2x2x2 blocks and schoolbook for others."""
    
    n, m, p = DIMENSIONS  # all 5
    
    # Block partitions
    # rows of A and C: [0,1], [2,3], [4]
    # cols of A and rows of B: [0,1], [2,3], [4]
    # cols of B and C: [0,1], [2,3], [4]
    
    row_blocks = [(0, 2), (2, 4), (4, 5)]  # (start, end)
    col_blocks = [(0, 2), (2, 4), (4, 5)]
    p_blocks = [(0, 2), (2, 4), (4, 5)]
    
    U = []  # list of columns, each column is a vector of length n*m = 25
    V = []  # list of columns, each column is a vector of length m*p = 25
    W = []  # list of columns, each column is a vector of length p*n = 25
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec, v_vec, w_vec are lists of length 25."""
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Strassen's 7 terms for 2x2 * 2x2 -> 2x2
    # Given A (2x2), B (2x2), compute C = A*B
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen term, we need to create U, V, W vectors
    # U corresponds to A entries, V to B entries, W to C entries
    # A is indexed by (i,j) -> index i*m + j = i*5 + j
    # B is indexed by (j,k) -> index j*p + k = j*5 + k  
    # C is indexed by (k,i) -> index k*n + i = k*5 + i  [trace convention W[k*n+i,r]]
    
    # Block (i_blk, k_blk, j_blk) all of size 2:
    # A rows: i_blk_start to i_blk_start+1, A cols: k_blk_start to k_blk_start+1
    # B rows: k_blk_start to k_blk_start+1, B cols: j_blk_start to j_blk_start+1
    # C rows: i_blk_start to i_blk_start+1, C cols: j_blk_start to j_blk_start+1
    
    def strassen_term(a_coeffs, b_coeffs, c_coeffs, a_offset, b_offset, c_offset):
        """
        Add a Strassen term.
        a_coeffs: dict mapping (local_i, local_j) to coefficient for A
        b_coeffs: dict mapping (local_j, local_k) to coefficient for B
        c_coeffs: dict mapping (local_i, local_k) to coefficient for C
        a_offset: (row_start, col_start) for A block
        b_offset: (row_start, col_start) for B block
        c_offset: (row_start, col_start) for C block
        """
        u_vec = [0] * 25
        v_vec = [0] * 25
        w_vec = [0] * 25
        
        for (li, lj), coeff in a_coeffs.items():
            global_i = a_offset[0] + li
            global_j = a_offset[1] + lj
            u_vec[global_i * 5 + global_j] = coeff
        
        for (lj, lk), coeff in b_coeffs.items():
            global_j = b_offset[0] + lj
            global_k = b_offset[1] + lk
            v_vec[global_j * 5 + global_k] = coeff
        
        for (li, lk), coeff in c_coeffs.items():
            global_i = c_offset[0] + li
            global_k = c_offset[1] + lk
            w_vec[global_k * 5 + global_i] = coeff  # W[k*n+i]
        
        add_term(u_vec, v_vec, w_vec)
    
    # The 7 Strassen terms:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # For each term, we decompose it into rank-1 components
    # M1 = (a11+a22)(b11+b22)
    #   = a11*b11 + a11*b22 + a22*b11 + a22*b22
    #   But Strassen computes it as one multiplication, so the rank-1 term is:
    #   U: a11 + a22 -> {a11: 1, a22: 1}
    #   V: b11 + b22 -> {b11: 1, b22: 1}
    #   W: C11 and C22 get contributions
    
    # Actually, each Strassen "multiplication" is one rank-1 term in the tensor decomposition.
    # The U vector has the linear combination of A entries, V has linear combo of B entries,
    # and W distributes the result to C entries.
    
    # M1 contributes to C11 and C22:
    # C11 = M1 + M4 - M5 + M7
    # C22 = M1 - M2 + M3 + M6
    
    # So for M1:
    # U: a11=1, a22=1
    # V: b11=1, b22=1
    # W: C11=1, C22=1
    
    # M2:
    # U: a21=1, a22=1
    # V: b11=1
    # W: C21=1, C22=-1
    
    # M3:
    # U: a11=1
    # V: b12=1, b22=-1
    # W: C12=1, C22=1
    
    # M4:
    # U: a22=1
    # V: b21=1, b11=-1
    # W: C11=1, C21=1
    
    # M5:
    # U: a11=1, a12=1
    # V: b22=1
    # W: C11=-1, C12=1
    
    # M6:
    # U: a21=1, a11=-1
    # V: b11=1, b12=1
    # W: C22=1
    
    # M7:
    # U: a12=1, a22=-1
    # V: b21=1, b22=1
    # W: C11=1
    
    strassen_terms = [
        # (a_coeffs, b_coeffs, c_coeffs)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}, {(0,0): 1, (1,1): 1}),  # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),           # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),             # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),             # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),             # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),             # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),             # M7
    ]
    
    # Process all block triples
    for i_blk in range(3):
        for k_blk in range(3):
            for j_blk in range(3):
                ri = i_blk  # which row block of A
                rk = k_blk  # which col block of A / row block of B
                rj = j_blk  # which col block of B / col block of C
                
                r_start = row_blocks[ri][0]
                r_end = row_blocks[ri][1]
                r_size = r_end - r_start
                
                c_start = col_blocks[rk][0]
                c_end = col_blocks[rk][1]
                c_size = c_end - c_start
                
                p_start = p_blocks[rj][0]
                p_end = p_blocks[rj][1]
                p_size = p_end - p_start
                
                a_offset = (r_start, c_start)
                b_offset = (c_start, p_start)
                c_offset = (r_start, p_start)
                
                if r_size == 2 and c_size == 2 and p_size == 2:
                    # Use Strassen
                    for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
                        strassen_term(a_coeffs, b_coeffs, c_coeffs, a_offset, b_offset, c_offset)
                else:
                    # Schoolbook
                    for li in range(r_size):
                        for lk in range(c_size):
                            for lj in range(p_size):
                                u_vec = [0] * 25
                                v_vec = [0] * 25
                                w_vec = [0] * 25
                                
                                u_vec[(r_start + li) * 5 + (c_start + lk)] = 1
                                v_vec[(c_start + lk) * 5 + (p_start + lj)] = 1
                                w_vec[(p_start + lj) * 5 + (r_start + li)] = 1
                                
                                add_term(u_vec, v_vec, w_vec)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}
