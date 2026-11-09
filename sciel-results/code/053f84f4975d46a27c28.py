"""Matrix multiplication 3x4x8 - rank-minimal tensor decomposition.

Strategy:
  1. Start with the schoolbook decomposition (rank 96).
  2. Apply algebraic rank-reduction techniques:
     - Pair columns that differ only in the k-index (for fixed i,j) and
       combine them using the fact that sum_k A[i,j] B[j,k] can be
       computed with fewer terms when structure is exploited.
     - Use the fact that for fixed i, the output C[i,k] = sum_j A[i,j] B[j,k].
       If we can share computation across different i values, we reduce rank.
  3. For small dimensions (3,4,8), try exhaustive or semi-exhaustive search
     for rank-reducing linear combinations of the schoolbook basis.

Given the time constraint, we focus on:
  - Known results: The tensor rank for 3x4x8 matrix multiplication.
  - The schoolbook rank is 3*4*8 = 96.
  - Known lower bounds and upper bounds for small matrices.
  - For 2x2x2, Strassen gives rank 7 (vs schoolbook 8).
  - For 3x3x3, the best known is 23 (vs schoolbook 27).
  
  For 3x4x8, we can decompose as:
    - Split the 3 into 2+1: use Strassen for the 2x4x8 part, schoolbook for 1x4x8.
    - 2x4x8: split 4 into 2+2: use Strassen for 2x2x4, Strassen for 2x2x4.
      2x2x4: split 4 into 2+2: Strassen for 2x2x2 twice = rank 7+7=14? No, that's not right.
      
  Actually, the tensor product structure means:
    - For multiplying an (n1+n2) x m matrix by m x (p1+p2) matrix:
      rank(n1+n2, m, p1+p2) <= rank(n1,m,p1) + rank(n1,m,p2) + rank(n2,m,p1) + rank(n2,m,p2)
    - For multiplying an n1 x (m1+m2) matrix by (m1+m2) x p matrix:
      rank(n1, m1+m2, p) <= rank(n1,m1,p) + rank(n1,m2,p)
    - For multiplying an (n1+m1) x p... no wait.
    
  Let me use the standard decomposition rules:
    - [n1 x m1] * [m1 x p1] + [n2 x m2] * [m2 x p2] (block diagonal):
      rank <= rank(n1,m1,p1) + rank(n2,m2,p2)
    - [n x (m1+m2)] * [(m1+m2) x p] (vertical split of middle):
      rank <= rank(n,m1,p) + rank(n,m2,p)
    - [(n1+n2) x m] * [m x p] (vertical split of first):
      This is NOT directly decomposable this way.
    - [n x m] * [m x (p1+p2)] (horizontal split of second):
      rank <= rank(n,m,p1) + rank(n,m,p2)
      
  Actually, the correct decomposition rules for tensor rank of matrix multiplication:
    The multiplication tensor M_{n,m,p} has indices (i,j,k) for i in [n], j in [m], k in [p].
    
    If we split n = n1 + n2:
      M_{n,m,p} = M_{n1,m,p} ⊕ M_{n2,m,p} (disjoint union)
      rank(M_{n,m,p}) <= rank(M_{n1,m,p}) + rank(M_{n2,m,p})
      
    If we split m = m1 + m2:
      M_{n,m,p} = M_{n,m1,p} ⊕ M_{n,m2,p}
      rank(M_{n,m,p}) <= rank(M_{n,m1,p}) + rank(M_{n,m2,p})
      
    If we split p = p1 + p2:
      M_{n,m,p} = M_{n,m,p1} ⊕ M_{n,m,p2}
      rank(M_{n,m,p}) <= rank(M_{n,m,p1}) + rank(M_{n,m,p2})
      
  So for 3x4x8:
    Split n=3 into 2+1: rank(3,4,8) <= rank(2,4,8) + rank(1,4,8)
    rank(1,4,8) = 1*4*8 = 32 (schoolbook, and it's known to be minimal since it's a rank-1 matrix... wait no)
    Actually for 1 x m x p, the tensor is just the identity-like tensor, and rank = m*p = 32.
    
    rank(2,4,8): split m=4 into 2+2: rank(2,4,8) <= rank(2,2,8) + rank(2,2,8) = 2*rank(2,2,8)
    rank(2,2,8): split p=8 into 2+2+2+2: rank(2,2,8) <= 4*rank(2,2,2) = 4*7 = 28 (using Strassen)
    So rank(2,4,8) <= 2*28 = 56
    Total: rank(3,4,8) <= 56 + 32 = 88
    
    Can we do better?
    Split p=8 into 4+4: rank(2,2,8) <= 2*rank(2,2,4)
    rank(2,2,4): split p=4 into 2+2: rank(2,2,4) <= 2*rank(2,2,2) = 14
    So rank(2,2,8) <= 2*14 = 28 (same as before)
    
    Alternative: for rank(2,2,4), is there a better decomposition than 14?
    The schoolbook for 2x2x4 is 16. Using Strassen on the 2x2x2 blocks gives 2*7=14.
    Is 14 optimal for 2x2x4? The lower bound... 
    
    Actually, let me think about this differently. The key insight is that we can use
    Strassen's algorithm recursively. The rank of matrix multiplication tensor M_{n,m,p}
    can be bounded using the "doubling" technique.
    
    For our specific case 3x4x8 = 96 schoolbook, the Strassen-based bound gives 88.
    
    Can we do even better? Let's try:
    Split 3 into 1+1+1: rank(3,4,8) <= 3*rank(1,4,8) = 3*32 = 96 (no improvement)
    Split 4 into 1+1+1+1: rank(3,4,8) <= 4*rank(3,1,8) = 4*24 = 96 (no improvement)
    
    The best we can do with simple splitting is 88.
    
    But wait, there might be more clever decompositions that don't follow the simple
    block structure. For small matrices, ad hoc decompositions can beat the recursive
    Strassen bounds.
    
    Given the time constraint, let me implement the 88-rank decomposition using
    Strassen for the 2x2x2 building blocks, and see if we can find any further
    reductions through local optimizations.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

def construct():
    deadline = time.time() + 140  # 140s limit, leaving 10s margin
    
    n, m, p = DIMENSIONS
    
    # Build the 88-rank decomposition using Strassen recursively
    # Strategy: 3x4x8 = (2+1) x 4 x 8
    #   2x4x8 = 2 x (2+2) x 8 = 2x2x8 + 2x2x8
    #   2x2x8 = 2x2x(2+2+2+2) = 4 x 2x2x2 (each using Strassen, rank 7)
    #   So 2x2x8 has rank 4*7 = 28
    #   2x4x8 has rank 2*28 = 56
    #   1x4x8 has rank 1*4*8 = 32 (schoolbook)
    #   Total: 56 + 32 = 88
    
    # Strassen decomposition for 2x2x2:
    # The 7 rank-1 tensors for 2x2x2 matrix multiplication
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # W[k*2+i, j] ... wait, the convention is W[k*n+i, j] for n=m=2
    # So W has 2*2=4 rows, indexed by k*2+i
    # U has 2*2=4 rows, indexed by i*2+j
    # V has 2*2=4 rows, indexed by j*2+k
    
    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, each M_t gives a rank-1 term:
    # U[:,t] * V[:,t] * W[:,t] where the product reconstructs the contribution
    
    # Let me define the Strassen decomposition properly.
    # For 2x2x2, we need U (4x7), V (4x7), W (4x7)
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen decomposition.
        U is 4x7 (rows: i*2+j), V is 4x7 (rows: j*2+k), W is 4x7 (rows: k*2+i)
        """
        U = np.zeros((4, 7), dtype=object)
        V = np.zeros((4, 7), dtype=object)
        W = np.zeros((4, 7), dtype=object)
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00=1, A11=1 -> row 0 (i=0,j=0)=1, row 3 (i=1,j=1)=1
        # V: B00=1, B11=1 -> row 0 (j=0,k=0)=1, row 3 (j=1,k=1)=1
        # W: C00=1, C11=1 -> row 0 (k=0,i=0)=1, row 3 (k=1,i=1)=1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2 = (A10 + A11) * B00
        # U: A10=1 (i=1,j=0), A11=1 (i=1,j=1) -> row 2, row 3
        # V: B00=1 -> row 0
        # W: C10=1 (k=1,i=0), C11=1 (k=1,i=1) -> row 2, row 3
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = 1
        
        # M3 = A00 * (B01 - B11)
        # U: A00=1 -> row 0
        # V: B01=1 (j=0,k=1), B11=-1 (j=1,k=1) -> row 1, row 3=-1
        # W: C01=1 (k=0,i=1) -> row 1
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1
        
        # M4 = A11 * (B10 - B00)
        # U: A11=1 -> row 3
        # V: B10=1 (j=1,k=0), B00=-1 (j=0,k=0) -> row 2, row 0=-1
        # W: C00=1 (k=0,i=0), C10=1 (k=1,i=0) -> row 0, row 2
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # M5 = (A00 + A01) * B11
        # U: A00=1 (i=0,j=0), A01=1 (i=0,j=1) -> row 0, row 1
        # V: B11=1 (j=1,k=1) -> row 3
        # W: C00=-1 (k=0,i=0), C01=1 (k=0,i=1) -> row 0=-1, row 1
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # U: A10=1 (i=1,j=0), A00=-1 (i=0,j=0) -> row 2, row 0=-1
        # V: B00=1 (j=0,k=0), B01=1 (j=0,k=1) -> row 0, row 1
        # W: C11=1 (k=1,i=1) -> row 3
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        # U: A01=1 (i=0,j=1), A11=-1 (i=1,j=1) -> row 1, row 3=-1
        # V: B10=1 (j=1,k=0), B11=1 (j=1,k=1) -> row 2, row 3
        # W: C00=1 (k=0,i=0), C11=1 (k=1,i=1) -> row 0, row 3
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[3,6] = 1
        
        return U, V, W
    
    # Build the full 3x4x8 decomposition
    # U: (3*4) x 88 = 12 x 88
    # V: (4*8) x 88 = 32 x 88  
    # W: (8*3) x 88 = 24 x 88
    
    U_full = np.zeros((12, 88), dtype=object)
    V_full = np.zeros((32, 88), dtype=object)
    W_full = np.zeros((24, 88), dtype=object)
    
    # Part 1: 2x4x8 (rows 0-7 of U, all of V, rows 0-15 of W)
    # Split 4 into 2+2: two 2x2x8 blocks
    # Each 2x2x8: split 8 into 4 2x2x2 blocks, each Strassen rank 7
    # So 2x2x8 has rank 28, and 2x4x8 has rank 56
    
    # For the first 2x2x8 block (j=0,1 of the 4):
    # V indices: j in {0,1}, k in {0,1,2,3,4,5,6,7} -> rows 0-15 of V
    # W indices: k in {0,...,7}, i in {0,1} -> rows 0-15 of W
    
    S_U, S_V, S_W = strassen_2x2x2()
    
    col_offset = 0
    for j_block in range(2):  # two blocks of 2 columns each
        for k_block in range(4):  # four blocks of 2 rows each
            # This is a 2x2x2 subproblem
            # U: i in {0,1}, j in {2*j_block, 2*j_block+1}
            # V: j in {2*j_block, 2*j_block+1}, k in {2*k_block, 2*k_block+1}
            # W: k in {2*k_block, 2*k_block+1}, i in {0,1}
            
            for t in range(7):
                # U contribution: rows i*4 + (2*j_block + j_local) for i in {0,1}, j_local in {0,1}
                for i in range(2):
                    for j_loc in range(2):
                        val = S_U[i*2 + j_loc, t]
                        if val != 0:
                            U_full[i*4 + (2*j_block + j_loc), col_offset] += val
                
                # V contribution: rows (2*j_block + j_local)*8 + (2*k_block + k_local)
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = S_V[j_loc*2 + k_loc, t]
                        if val != 0:
                            V_full[(2*j_block + j_loc)*8 + (2*k_block + k_loc), col_offset] += val
                
                # W contribution: rows (2*k_block + k_local)*2 + i
                for k_loc in range(2):
                    for i in range(2):
                        val = S_W[k_loc*2 + i, t]
                        if val != 0:
                            W_full[(2*k_block + k_loc)*2 + i, col_offset] += val
                
                col_offset += 1
    
    # Part 2: 1x4x8 (row 8-11 of U, all of V, rows 16-23 of W)
    # Schoolbook: rank 32
    # U: i=2 (fixed), j in {0,1,2,3}
    # V: j in {0,1,2,3}, k in {0,...,7}
    # W: k in {0,...,7}, i=2 (fixed)
    
    for j in range(4):
        for k in range(8):
            U_full[8 + j, col_offset] = 1  # i=2, j
            V_full[j*8 + k, col_offset] = 1
            W_full[k*2 + 2, col_offset] = 1  # k, i=2
            col_offset += 1
    
    # Convert to lists of lists with integer/fraction values
    def to_list(mat):
        rows = []
        for row in mat:
            r = []
            for val in row:
                if isinstance(val, Fraction):
                    r.append(val)
                else:
                    r.append(int(val))
            rows.append(r)
        return rows
    
    U_list = to_list(U_full)
    V_list = to_list(V_full)
    W_list = to_list(W_full)
    
    return {"U": U_list, "V": V_list, "W": W_list}
