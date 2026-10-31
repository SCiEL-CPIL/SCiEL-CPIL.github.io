"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-49 Strassen decomposition for 4x4 matrix multiplication.
    
    The 4x4 matmul can be decomposed using two 2x2 Strassen blocks.
    Each 2x2 Strassen block has rank 7, and we need to combine them appropriately.
    
    Actually, let me think more carefully. The standard approach for 4x4:
    
    We can partition A (4x4), B (4x4), C (4x4) into 2x2 blocks of 2x2 matrices:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    Where each Cij is 2x2.
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each of these is a 2x2 matmul (2x2 * 2x2 = 2x2).
    
    For each 2x2 matmul, Strassen uses 7 multiplications.
    We have 4 such 2x2 matmuls, so naive would be 4*7 = 28? No, that's not right.
    
    Actually, the tensor for 4x4 matmul has dimension 4*4*4 = 64.
    The rank of the 4x4 matmul tensor is known to be at most 49 (via recursive Strassen: 2^3 = 8 for 2x2, then 8*8=64... no).
    
    Let me reconsider. The Strassen algorithm for n x n matmul has complexity O(n^log2(7)).
    For 2x2: rank 7
    For 4x4: we can view it as 2x2 blocks, each block being 2x2. The tensor rank of 4x4 matmul is at most 7^2 = 49.
    
    The construction: The 4x4 matmul tensor T_4 can be written as a sum of 49 rank-1 tensors.
    
    The recursive Strassen construction:
    - 2x2 Strassen has 7 "multiplication steps" with specific linear combinations.
    - For 4x4, we apply the 2x2 Strassen pattern at the block level, where each "element" is itself a 2x2 matrix, and we recursively use 2x2 Strassen for each of those.
    
    Let me define the 2x2 Strassen formulas:
    For computing C = A*B where A, B, C are 2x2:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-f)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 4x4, we partition into 2x2 blocks. Let's denote the blocks as:
    A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    
    The 4x2 block-level Strassen:
    We need to compute:
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each Cij is 2x2.
    
    Using Strassen at the block level (treating blocks as "elements"):
    M00 = (A00 + A11)(B00 + B11)
    M01 = (A10 + A11)B00
    M02 = A00(B01 - B11)
    M03 = A11(B10 - B00)
    M04 = (A00 + A01)B11
    M05 = (A10 - A00)(B00 + B01)
    M06 = (A01 - A11)(B10 + B11)
    
    Then:
    C00 = M00 + M03 - M04 + M06
    C01 = M02 + M04
    C10 = M01 + M03
    C11 = M00 - M01 + M02 + M05
    
    Now each Mij is a product of two 2x2 matrices (after the linear combinations). Each such product requires 7 scalar multiplications via Strassen.
    
    So total rank = 7 (block-level) * 7 (element-level) = 49.
    
    Now I need to express this as U, V, W matrices.
    
    The tensor T[i,j,k] = delta(i,k) for matmul... actually the matmul tensor is:
    T_{(i,j),(j',k),(k',i')} = delta(j,j') * delta(k,k') ... 
    
    Let me use the standard indexing. For C = A * B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i,j,k] where the multiplication tensor has entries T[i,j,k] = 1 if the indices correspond to a valid matmul term.
    
    In the factorization: C = sum_r U[:,r] * V[:,r] * W[:,r] (outer product / Khatri-Rao)
    
    Where:
    U has shape (n*m, rank) = (16, 49)
    V has shape (m*p, rank) = (16, 49)  
    W has shape (p*n, rank) = (16, 49)
    
    And the output follows W[k*n+i, r] convention.
    
    So W is indexed by [k*n + i, r] where i is the row of C and k is the column of C.
    
    Let me build this explicitly.
    
    First, let me define the 2x2 Strassen basis. For a 2x2 matmul with entries:
    A = [[a, b], [c, d]] indexed as A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    B = [[e, f], [g, h]] indexed as B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    C = [[p, q], [r, s]] indexed as C[0,0]=p, C[0,1]=q, C[1,0]=r, C[1,1]=s
    
    The 7 Strassen terms:
    t0: (a+d)(e+h) -> U: a=1,d=1; V: e=1,h=1; W: p=1, r=-1... 
    
    Wait, I need to be more careful. Each rank-1 term contributes to U, V, W.
    
    For a single 2x2 Strassen term t:
    U_t is a vector in R^4 (for the 4 entries of A)
    V_t is a vector in R^4 (for the 4 entries of B)
    W_t is a vector in R^4 (for the 4 entries of C, with the trace convention C[k*2+i])
    
    The 7 terms for 2x2:
    Term 0: (a+d)(e+h)
      U: [1, 0, 0, 1]  (a and d)
      V: [1, 0, 0, 1]  (e and h)
      W: contributes to C[0,0]=+1, C[1,1]=-1, and also... 
      
    Let me just write out all 7 terms and their contributions to C.
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    M1 = (a+d)(e+h): U=[1,0,0,1], V=[1,0,0,1]
    M2 = (c+d)(e): U=[0,0,1,1], V=[1,0,0,0]
    M3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1]
    M4 = d(g-f): U=[0,0,0,1], V=[0,-1,1,0]
    M5 = (a+b)h: U=[1,1,0,0], V=[0,0,0,1]
    M6 = (c-a)(e+f): U=[-1,0,1,0], V=[1,1,0,0]
    M7 = (b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1]
    
    Contributions to C (with C indexed as [C00, C01, C10, C11]):
    M1: C00+=1, C11+=1 -> W=[1,0,0,1]
    M2: C10+=1, C11+=-1 -> W=[0,0,1,-1]
    M3: C01+=1, C11+=1 -> W=[0,1,0,1]
    M4: C00+=1, C10+=1 -> W=[1,0,1,0]
    M5: C00+=-1, C01+=1 -> W=[-1,1,0,0]
    M6: C11+=1 -> W=[0,0,0,1]
    M7: C00+=1 -> W=[1,0,0,0]
    
    Let me verify:
    C00 = M1+M4-M5+M7 = 1+1-(-1)+1 = 1+1+1+1 = 4? That's wrong.
    
    Let me redo. C00 = M1 + M4 - M5 + M7
    Coefficient of M1 in C00: +1
    Coefficient of M4 in C00: +1
    Coefficient of M5 in C00: -1
    Coefficient of M7 in C00: +1
    
    So W for M1: [1, 0, 0, 1] (C00+=1, C11+=1)
    W for M2: [0, 0, 1, -1] (C10+=1, C11+=-1)
    W for M3: [0, 1, 0, 1] (C01+=1, C11+=1)
    W for M4: [1, 0, 1, 0] (C00+=1, C10+=1)
    W for M5: [-1, 1, 0, 0] (C00+=-1, C01+=1)
    W for M6: [0, 0, 0, 1] (C11+=1)
    W for M7: [1, 0, 0, 0] (C00+=1)
    
    Check C00: M1*1 + M4*1 + M5*(-1) + M7*1 = 1+1-1+1 = 2? Should be 1.
    
    Hmm, I think I'm making an error. Let me look up the exact Strassen formulas.
    
    Standard Strassen for 2x2:
    M1 = (a+d)(e+h)
    M2 = (c+d)(e)
    M3 = (a)(f-h)
    M4 = (d)(g-f)
    M5 = (a+b)(h)
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    So:
    C11 gets: M1(+1), M4(+1), M5(-1), M7(+1)
    C12 gets: M3(+1), M5(+1)
    C21 gets: M2(+1), M4(+1)
    C22 gets: M1(+1), M2(-1), M3(+1), M6(+1)
    
    So W vectors (indexed [C11, C12, C21, C22]):
    M1: [1, 0, 0, 1]
    M2: [0, 0, 1, -1]
    M3: [0, 1, 0, 1]
    M4: [1, 0, 1, 0]
    M5: [-1, 1, 0, 0]
    M6: [0, 0, 0, 1]
    M7: [1, 0, 0, 0]
    
    Now for the 4x4 case, I'll construct it recursively.
    
    The 4x4 matmul can be seen as a 2x2 block matmul where each block is 2x2.
    
    Block-level Strassen (7 terms), where each "element" is a 2x2 matrix:
    Let A_blocks = [[A00, A01], [A10, A11]]
    Let B_blocks = [[B00, B01], [B10, B11]]
    
    Block Strassen terms:
    BM1 = (A00+A11)(B00+B11)
    BM2 = (A10+A11)(B00)
    BM3 = (A00)(B01-B11)
    BM4 = (A11)(B10-B00)
    BM5 = (A00+A01)(B11)
    BM6 = (A10-A00)(B00+B01)
    BM7 = (A01-A11)(B10+B11)
    
    C00 = BM1 + BM4 - BM5 + BM7
    C01 = BM3 + BM5
    C10 = BM2 + BM4
    C11 = BM1 - BM2 + BM3 + BM6
    
    Each BMi is a 2x2 matmul (product of two 2x2 matrices), which itself uses 7 scalar multiplications.
    
    So total: 7 * 7 = 49 scalar multiplications.
    
    Now I need to build U, V, W of shape (16, 49).
    
    The indexing:
    A is 4x4, flattened as A[i*4+j] for i,j in 0..3
    B is 4x4, flattened as B[j*4+k] for j,k in 0..3
    C is 4x4, with W indexed as W[k*4+i] for i (row), k (col)
    
    Block A00 = A[0:2, 0:2] -> indices 0,1,4,5 in flat (i*4+j)
    Block A01 = A[0:2, 2:4] -> indices 2,3,6,7
    Block A10 = A[2:4, 0:2] -> indices 8,9,12,13
    Block A11 = A[2:4, 2:4] -> indices 10,11,14,15
    
    Within each 2x2 block, the local indices are:
    [0,0] -> 0, [0,1] -> 1, [1,0] -> 2, [1,1] -> 3
    
    For U (shape 16x49), each column corresponds to one of the 49 scalar multiplications.
    
    Let me index the 49 terms as (block_term, inner_term) where block_term in 0..6 and inner_term in 0..6.
    
    For a given block_term bt and inner_term it:
    - The U vector gets contributions from the block-level U for bt, applied to the appropriate block, and within that block, the inner-level U for it.
    
    Block-level U vectors (for the 2x2 block matmul, treating blocks as elements):
    BU[bt] is a vector in R^4 (for the 4 blocks A00, A01, A10, A11)
    
    BU[0] (BM1): [1, 0, 0, 1] (A00 and A11)
    BU[1] (BM2): [0, 0, 1, 1] (A10 and A11)
    BU[2] (BM3): [1, 0, 0, 0] (A00)
    BU[3] (BM4): [0, 0, 0, 1] (A11)
    BU[4] (BM5): [1, 1, 0, 0] (A00 and A01)
    BU[5] (BM6): [-1, 0, 1, 0] (A10 and -A00)
    BU[6] (BM7): [0, 1, 0, -1] (A01 and -A11)
    
    Similarly for BV (block-level V):
    BV[0]: [1, 0, 0, 1] (B00 and B11)
    BV[1]: [1, 0, 0, 0] (B00)
    BV[2]: [0, 1, 0, -1] (B01 and -B11)
    BV[3]: [0, 0, 1, -1] (B10 and -B00)
    BV[4]: [0, 0, 0, 1] (B11)
    BV[5]: [1, 1, 0, 0] (B00 and B01)
    BV[6]: [0, 0, 1, 1] (B10 and B11)
    
    And BW (block-level W, for C blocks C00, C01, C10, C11):
    BW[0]: [1, 0, 0, 1] (C00 and C11)
    BW[1]: [0, 0, 1, -1] (C10 and -C11)
    BW[2]: [0, 1, 0, 1] (C01 and C11)
    BW[3]: [1, 0, 1, 0] (C00 and C10)
    BW[4]: [-1, 1, 0, 0] (-C00 and C01)
    BW[5]: [0, 0, 0, 1] (C11)
    BW[6]: [1, 0, 0, 0] (C00)
    
    Now for each (bt, it) pair:
    U[col] = for each block b where BU[bt][b] != 0, add BU[bt][b] * IU[it] placed in the corresponding block positions
    V[col] = for each block b where BV[bt][b] != 0, add BV[bt][b] * IV[it] placed in the corresponding block positions
    W[col] = for each block b where BW[bt][b] != 0, add BW[bt][b] * IW[it] placed in the corresponding block positions
    
    Where IU, IV, IW are the 2x2 Strassen vectors defined earlier.
    
    Let me code this up.
    """
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # 2x2 Strassen vectors
    # IU[it] for A entries [a, b, c, d]
    IU = np.array([
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1]  # M7: b-d
    ], dtype=float)
    
    # IV[it] for B entries [e, f, g, h]
    IV = np.array([
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [0, -1, 1, 0], # M4: g-f
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1]   # M7: g+h
    ], dtype=float)
    
    # IW[it] for C entries [C11, C12, C21, C22]
    IW = np.array([
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ], dtype=float)
    
    # Block-level vectors (for 2x2 block matmul)
    # BU[bt] for blocks [A00, A01, A10, A11]
    BU = np.array([
        [1, 0, 0, 1],   # BM1: A00+A11
        [0, 0, 1, 1],   # BM2: A10+A11
        [1, 0, 0, 0],   # BM3: A00
        [0, 0, 0, 1],   # BM4: A11
        [1, 1, 0, 0],   # BM5: A00+A01
        [-1, 0, 1, 0],  # BM6: A10-A00
        [0, 1, 0, -1]   # BM7: A01-A11
    ], dtype=float)
    
    # BV[bt] for blocks [B00, B01, B10, B11]
    BV = np.array([
        [1, 0, 0, 1],   # BM1: B00+B11
        [1, 0, 0, 0],   # BM2: B00
        [0, 1, 0, -1],  # BM3: B01-B11
        [0, 0, 1, -1],  # BM4: B10-B00
        [0, 0, 0, 1],   # BM5: B11
        [1, 1, 0, 0],   # BM6: B00+B01
        [0, 0, 1, 1]    # BM7: B10+B11
    ], dtype=float)
    
    # BW[bt] for blocks [C00, C01, C10, C11]
    BW = np.array([
        [1, 0, 0, 1],   # BM1: C00+C11
        [0, 0, 1, -1],  # BM2: C10-C11
        [0, 1, 0, 1],   # BM3: C01+C11
        [1, 0, 1, 0],   # BM4: C00+C10
        [-1, 1, 0, 0],  # BM5: -C00+C01
        [0, 0, 0, 1],   # BM6: C11
        [1, 0, 0, 0]    # BM7: C00
    ], dtype=float)
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Block positions in the 4x4 matrix (flat index i*4+j)
    # A blocks:
    # A00: rows 0-1, cols 0-1 -> flat indices 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> flat indices 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> flat indices 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> flat indices 10,11,14,15
    
    # B blocks (B is m x p = 4x4, flat index j*4+k):
    # B00: rows 0-1, cols 0-1 -> flat indices 0,1,4,5
    # B01: rows 0-1, cols 2-3 -> flat indices 2,3,6,7
    # B10: rows 2-3, cols 0-1 -> flat indices 8,9,12,13
    # B11: rows 2-3, cols 2-3 -> flat indices 10,11,14,15
    
    # C blocks (W is indexed as k*4+i, where i=row, k=col):
    # C00: rows 0-1, cols 0-1 -> W indices: for i in 0-1, k in 0-1: k*4+i
    #   i=0,k=0: 0; i=1,k=0: 1; i=0,k=1: 4; i=1,k=1: 5
    # C01: rows 0-1, cols 2-3 -> W indices: i in 0-1, k in 2-3: k*4+i
    #   i=0,k=2: 8; i=1,k=2: 9; i=0,k=3: 12; i=1,k=3: 13
    # C10: rows 2-3, cols 0-1 -> W indices: i in 2-3, k in 0-1: k*4+i
    #   i=2,k=0: 2; i=3,k=0: 3; i=2,k=1: 6; i=3,k=1: 7
    # C11: rows 2-3, cols 2-3 -> W indices: i in 2-3, k in 2-3: k*4+i
    #   i=2,k=2: 10; i=3,k=2: 11; i=2,k=3: 14; i=3,k=3: 15
    
    # Local indices within a 2x2 block: [0,0]->0, [0,1]->1, [1,0]->2, [1,1]->3
    
    # For U: block b local index l -> global index
    # A00 (b=0): local [0,0]->0, [0,1]->1, [1,0]->4, [1,1]->5
    # A01 (b=1): local [0,0]->2, [0,1]->3, [1,0]->6, [1,1]->7
    # A10 (b=2): local [0,0]->8, [0,1]->9, [1,0]->12, [1,1]->13
    # A11 (b=3): local [0,0]->10, [0,1]->11, [1,0]->14, [1,1]->15
    
    U_block_map = {
        0: [0, 1, 4, 5],   # A00
        1: [2, 3, 6, 7],   # A01
        2: [8, 9, 12, 13], # A10
        3: [10, 11, 14, 15] # A11
    }
    
    # For V: block b local index l -> global index (B flat: j*4+k)
    V_block_map = {
        0: [0, 1, 4, 5],   # B00
        1: [2, 3, 6, 7],   # B01
        2: [8, 9, 12, 13], # B10
        3: [10, 11, 14, 15] # B11
    }
    
    # For W: block b local index l -> global index (W flat: k*4+i)
    # C00 (b=0): local [0,0]->(i=0,k=0)=0, [0,1]->(i=0,k=1)=4, [1,0]->(i=1,k=0)=1, [1,1]->(i=1,k=1)=5
    #   So local index order [C11,C12,C21,C22] maps to:
    #   C11=(i=0,k=0)=0, C12=(i=0,k=1)=4, C21=(i=1,k=0)=1, C22=(i=1,k=1)=5
    # C01 (b=1): C11=(i=0,k=2)=8, C12=(i=0,k=3)=12, C21=(i=1,k=2)=9, C22=(i=1,k=3)=13
    # C10 (b=2): C11=(i=2,k=0)=2, C12=(i=2,k=1)=6, C21=(i=3,k=0)=3, C22=(i=3,k=1)=7
    # C11 (b=3): C11=(i=2,k=2)=10, C12=(i=2,k=3)=14, C21=(i=3,k=2)=11, C22=(i=3,k=3)=15
    
    W_block_map = {
        0: [0, 4, 1, 5],   # C00
        1: [8, 12, 9, 13], # C01
        2: [2, 6, 3, 7],   # C10
        3: [10, 14, 11, 15] # C11
    }
    
    col = 0
    for bt in range(7):
        for it in range(7):
            # U
            for b in range(4):
                if BU[bt, b] != 0:
                    for l in range(4):
                        if IU[it, l] != 0:
                            U[U_block_map[b][l], col] += BU[bt, b] * IU[it, l]
            
            # V
            for b in range(4):
                if BV[bt, b] != 0:
                    for l in range(4):
                        if IV[it, l] != 0:
                            V[V_block_map[b][l], col] += BV[bt, b] * IV[it, l]
            
            # W
            for b in range(4):
                if BW[bt, b] != 0:
                    for l in range(4):
                        if IW[it, l] != 0:
                            W[W_block_map[b][l], col] += BW[bt, b] * IW[it, l]
            
            col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
