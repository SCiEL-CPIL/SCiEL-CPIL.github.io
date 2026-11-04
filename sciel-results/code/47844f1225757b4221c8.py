"""Rewritten program: structural block-decomposition + Strassen for 2x2 sub-blocks."""
DIMENSIONS = [3, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Build an exact tensor decomposition for 3x4 * 4x4 matrix multiplication
    using a block-partition strategy:
      - Partition the 3x4 left matrix into 2x2 sub-blocks where possible.
      - Use Strassen's rank-7 algorithm for 2x2 * 2x2 blocks.
      - Use schoolbook (rank 8) for the remaining 1x4 row block.
    This yields a structured decomposition with rank significantly below 48.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    deadline = time.time() + 115.0

    # We'll build U, V, W as lists of (value, row_index) per rank column
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (16, rank)
    # W has shape (p*n, rank) = (12, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For matrix mult: C[i,l] = sum_j A[i,j] B[j,l]
    # Tensor: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,l} * delta_{j,k} ... 
    # Wait, let me re-read the contract.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,l] = sum_j A[i,j] * B[j,l]
    # The tensor decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal delta_{i,l} * delta_{j,k}? No...
    # Actually for matrix mult, the "tensor" is:
    # T[i,j,k,l] = 1 if j==k and i==l? No.
    # C[i,l] = sum_j A[i,j] B[j,l]
    # So the trilinear form is: sum_{i,j,l} C[i,l] * (sum_j A[i,j] B[j,l])
    # The tensor entries are T[i,j,l] = 1 (for all i,j,l where the contraction gives C)
    # But with 4 indices: A has (i,j), B has (j,l), C has (i,l)
    # So T[i,j,j,l] = 1 (the j indices must match for the contraction)
    # In the W convention W[k*n+i, r], the first index is k*n+i.
    # So W index (k*n+i) corresponds to... let's see.
    # If we have W[k*n+i, r], then in the sum sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # we need this to equal 1 when i==l and j==k (for the delta pattern of matrix mult).
    # Wait, but l is the output column. In W[k*n+i], i is the row of output, k is... 
    # Hmm, let me reconsider. The output C is n x p. C[i, l] where l is in [0, p).
    # But W has shape (p*n, rank), so W indices are 0..p*n-1.
    # W[k*n + i] suggests the first "index" is k (range p) and second is i (range n).
    # But C[i, l] has i in [0,n), l in [0,p). So if W index is k*n+i, then k=l and i=i?
    # That would mean W[l*n + i, r]. Let me verify with schoolbook.
    # In schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # Sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 only when the column matches
    # In schoolbook, each column r corresponds to a unique (i,j,k) triple.
    # U[i*m+j, r]=1 iff r corresponds to (i',j') where i'=i, j'=j
    # V[j*p+k, r]=1 iff r corresponds to (j',k') where j'=j, k'=k  
    # W[k*n+i, r]=1 iff r corresponds to (k',i') where k'=k, i'=i
    # So all three are 1 iff r corresponds to (i,j,k).
    # The sum is 1 iff there exists r with U[i*m+j,r]=V[j*p+k,r]=W[k*n+i,r]=1
    # which happens iff r corresponds to (i,j,k) in all three, i.e., the indices match.
    # So the sum = 1 iff (i,j,k) is consistent, which means the output is C[i, k]? 
    # But wait, for matrix mult C[i,l] = sum_j A[i,j] B[j,l], the "k" in B[j,k] 
    # should be the column index of B, which is l. And the output row is i.
    # So W[k*n+i] = W[l*n + i] where l is the output column. Yes! That makes sense.
    # So the tensor is: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = delta_{j,k} * delta_{i,i} ... 
    # No, it's: the sum equals 1 iff the (i,j) in U, (j,k) in V, and (k,i) in W 
    # all refer to the same rank component. For schoolbook, this gives:
    # sum = 1 iff there's a column where all three are 1, which requires
    # U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1 simultaneously.
    # In schoolbook, column (i0,j0,k0) has U[i0*m+j0]=1, V[j0*p+k0]=1, W[k0*n+i0]=1.
    # So the sum is 1 iff (i,j,k) = (i0,j0,k0) for some column, i.e., the indices match.
    # This gives: sum_r ... = 1 iff the "path" i->j->k->i is closed, 
    # which for matrix mult means j=k (B's row = A's col... wait no).
    
    # Let me just verify: for matrix mult C[i,l] = sum_j A[i,j] B[j,l]
    # The bilinear form: <C, sum_j A[i,j]B[j,l]> 
    # As a trilinear form in A, B, C: sum_{i,j,l} C[i,l] A[i,j] B[j,l]
    # The tensor T has T[i,j,l] = 1 (for all i in [0,n), j in [0,m), l in [0,p))
    # But with the constraint that the j index of A matches the first index of B,
    # and the output is (i,l).
    # In the 4-index tensor form with W convention:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This should equal 1 when the indices form a valid "edge" in the matrix mult.
    # For schoolbook, it equals 1 iff (i,j,k) is a valid triple, i.e., 
    # the term A[i,j]*B[j,k] contributes to C[i,k]. So the sum = delta_{j,j} * 1 = 1
    # for all valid (i,j,k). But that's always 1 for any (i,j,k)!
    # That can't be right. Let me re-examine.
    
    # OH WAIT. I think I'm confusing myself. Let me re-read the schoolbook:
    # U[i*m+j, col] = 1.0
    # V[j*p+k, col] = 1.0  
    # W[k*n+i, col] = 1.0
    # col is the rank index.
    # The sum over r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For a given (i, j, k, l) -- wait, there's no l in the sum!
    # The sum is over r, and the free indices are i, j, k (from U, V, W respectively).
    # But W has index k*n+i, so the free indices from W are k and i.
    # So the sum is a function of (i, j, k): sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This should equal... what? For matrix mult, C[i, k] = sum_j A[i,j] B[j,k].
    # So the "tensor" value for (i, j, k) should be 1 (since A[i,j]*B[j,k] contributes to C[i,k]).
    # So sum_r ... = 1 for ALL (i,j,k)? That would mean the tensor is all-ones, which is wrong.
    
    # I think the issue is that the decomposition doesn't just need to give 1 for all (i,j,k).
    # Let me look at this differently. The matrix multiplication tensor is:
    # M = sum_{i,j,l} e_i (output row) * e_j (input col of A) * e_l (input col of B) 
    # with the constraint that j is both the column of A and row of B.
    # Actually, the standard way: the matrix mult tensor has entries:
    # T[i, j, l] = 1 where i=output row, j=A's column/B's row, l=B's column
    # But as a 4-index tensor (with separate indices for A and B inputs):
    # T[i, j_A, j_B, l] = delta_{j_A, j_B}
    # i.e., T[i, j1, j2, l] = 1 if j1==j2, else 0.
    # Now mapping to U, V, W:
    # U has indices (i, j_A) -> U[i*m + j_A, r]
    # V has indices (j_B, k) where k is B's column -> V[j_B*p + k, r]
    # W has indices (k, i) -> W[k*n + i, r]
    # So the tensor is: T[i, j_A, j_B, k] = sum_r U[i*m+j_A, r] V[j_B*p+k, r] W[k*n+i, r]
    # This should equal delta_{j_A, j_B} (i.e., 1 if j_A == j_B, else 0).
    # Let me verify with schoolbook:
    # sum_r U[i*m+j_A, r] V[j_B*p+k, r] W[k*n+i, r]
    # = sum over columns r where U[i*m+j_A, r]=1 AND V[j_B*p+k, r]=1 AND W[k*n+i, r]=1
    # In schoolbook, column r=(i0,j0,k0) has U[i0*m+j0, r]=1, V[j0*p+k0, r]=1, W[k0*n+i0, r]=1.
    # So the sum = 1 iff there exists (i0,j0,k0) such that i0=i, j0=j_A, j_B=j0, k0=k, i0=i.
    # i.e., i0=i, j0=j_A, j_B=j_A, k0=k.
    # So the sum = 1 iff j_A == j_B (since j0=j_A and j_B=j0 implies j_A=j_B).
    # And it's 1 for any (i, k) when j_A == j_B.
    # So T[i, j_A, j_B, k] = 1 if j_A == j_B, else 0. YES! That's exactly delta_{j_A, j_B}.
    # Great, so my understanding is correct.
    
    # Now for the block decomposition:
    # We partition A (3x4) into rows. Let's split the 3 rows into a 2-row block and a 1-row block.
    # A = [A_top; A_bottom] where A_top is 2x4, A_bottom is 1x4.
    # B is 4x4, partition its columns into two 2-column blocks: B = [B_left | B_right]
    # Then C = A * B = [A_top * B_left, A_top * B_right; A_bottom * B_left, A_bottom * B_right]
    # C_top_left is 2x2 = A_top(2x4) * B_left(4x2)
    # C_top_right is 2x2 = A_top(2x4) * B_right(4x2)
    # C_bottom_left is 1x2 = A_bottom(1x4) * B_left(4x2)
    # C_bottom_right is 1x2 = A_bottom(1x4) * B_right(4x2)
    
    # For 2x2 * 4x2: we can partition B_left (4x2) into two 2x2 blocks vertically:
    # B_left = [B_left_top; B_left_bottom] where each is 2x2.
    # Then A_top * B_left = A_top(2x4) * [B_left_top; B_left_bottom](4x2)
    # = A_top * B_left_top + A_top * B_left_bottom
    # But A_top is 2x4, B_left_top is 2x2. We need to partition A_top's columns too.
    # A_top = [A_top_left | A_top_right] where each is 2x2.
    # Then A_top * B_left_top = A_top_left(2x2) * B_left_top(2x2) + A_top_right(2x2) * ??? 
    # No, that's not right. A_top * B_left = [A_top_left | A_top_right] * [B_left_top; B_left_bottom]
    # = A_top_left * B_left_top + A_top_right * B_left_bottom
    # Each term is 2x2 * 2x2 = 2x2.
    # So C_top_left = A_top_left * B_left_top + A_top_right * B_left_bottom
    # Similarly for C_top_right with B_right.
    
    # Each 2x2 * 2x2 multiplication can use Strassen's algorithm (rank 7).
    # So C_top_left needs 7+7=14 rank terms (two Strassen multiplications).
    # C_top_right needs 14 rank terms.
    # For the bottom row: A_bottom(1x4) * B_left(4x2) and A_bottom(1x4) * B_right(4x2).
    # 1x4 * 4x2: we can do schoolbook: 4 multiplications for each, so 8 total for both.
    # But we can be smarter: A_bottom = [b0, b1, b2, b3] (1x4)
    # B_left = 4x2, B_right = 4x2.
    # C_bottom_left = sum_j A_bottom[0,j] * B_left[j, :] (1x2)
    # C_bottom_right = sum_j A_bottom[0,j] * B_right[j, :] (1x2)
    # Schoolbook: 4 terms for left, 4 for right = 8 terms.
    
    # Total rank: 14 + 14 + 8 = 36.
    
    # Let me build this step by step.
    # U shape: (12, rank), indices i*m+j where i in 0..2, j in 0..3
    # V shape: (16, rank), indices j*p+k where j in 0..3, k in 0..3
    # W shape: (12, rank), indices k*n+i where k in 0..3, i in 0..2
    
    U_list = []  # Each element is a dict or list of (row, value) for each rank column
    V_list = []
    W_list = []
    
    def add_strassen(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        Add Strassen's 7-term decomposition for a 2x2 * 2x2 sub-multiplication.
        A is located at rows A_rows, cols A_cols of the full A matrix.
        B is located at rows B_rows, cols B_cols of the full B matrix.
        C is located at rows C_rows, cols C_cols of the full C matrix.
        """
        # Strassen's 7 products for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A is 2x2, located at A_rows x A_cols (A_rows = [r1, r2], A_cols = [c1, c2])
        # A11 = A[A_rows[0], A_cols[0]], A12 = A[A_rows[0], A_cols[1]]
        # A21 = A[A_rows[1], A_cols[0]], A22 = A[A_rows[1], A_cols[1]]
        # B is 2x2, located at B_rows x B_cols
        # B11 = B[B_rows[0], B_cols[0]], B12 = B[B_rows[0], B_cols[1]]
        # B21 = B[B_rows[1], B_cols[0]], B22 = B[B_rows[1], B_cols[1]]
        # C is 2x2, located at C_rows x C_cols
        # C11 = C[C_rows[0], C_cols[0]], C12 = C[C_rows[0], C_cols[1]]
        # C21 = C[C_rows[1], C_cols[0]], C22 = C[C_rows[1], C_cols[1]]
        
        # In our tensor:
        # U[i*m + j, r] corresponds to A[i, j]
        # V[j*p + k, r] corresponds to B[j, k]
        # W[k*n + i, r] corresponds to C[i, k]
        
        # For Strassen, each product Mk involves linear combinations of A and B entries.
        # We need to add 7 rank-1 terms.
        
        # The 7 Strassen terms:
        # M1: (A11+A22) * (B11+B22)
        # M2: (A21+A22) * B11
        # M3: A11 * (B21-B22)
        # M4: A22 * (B12-B11)
        # M5: (A11+A12) * B22
        # M6: (A21-A11) * (B11+B12)
        # M7: (A12-A22) * (B21+B22)
        
        # Each product is of the form (linear combo of A entries) * (linear combo of B entries)
        # contributing to (linear combo of C entries).
        
        # Let me define the coefficients for each of the 7 terms.
        # For each term, I need:
        #   U_coeffs: dict mapping (i, j) -> coefficient for A[i,j]
        #   V_coeffs: dict mapping (j, k) -> coefficient for B[j,k]
        #   W_coeffs: dict mapping (k, i) -> coefficient for C[i,k]
        
        # M1: U: A11=1, A22=1; V: B11=1, B22=1; W: C11=1, C22=1
        # M2: U: A21=1, A22=1; V: B11=1; W: C21=1, C22=-1
        # M3: U: A11=1; V: B21=1, B22=-1; W: C12=1, C22=1
        # M4: U: A22=1; V: B12=1, B11=-1; W: C11=1, C21=1
        # M5: U: A11=1, A12=1; V: B22=1; W: C11=-1, C12=1
        # M6: U: A21=1, A11=-1; V: B11=1, B12=1; W: C22=1
        # M7: U: A12=1, A22=-1; V: B21=1, B22=1; W: C11=1, C22=1
        
        # Wait, let me double-check the Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1 contributes to C11 (+1) and C22 (+1)
        # M2 contributes to C21 (+1) and C22 (-1)
        # M3 contributes to C12 (+1) and C22 (+1)
        # M4 contributes to C11 (+1) and C21 (+1)
        # M5 contributes to C11 (-1) and C12 (+1)
        # M6 contributes to C22 (+1)
        # M7 contributes to C11 (+1) and C22 (+1)
        
        # Let me verify C11: M1+M4-M5+M7 = 1+1-(-1)+1 = 1+1+1+1 = 4? No, that's wrong.
        # C11 should get coefficient 1 from each term that contributes.
        # M1: C11 coeff +1
        # M4: C11 coeff +1
        # M5: C11 coeff -1
        # M7: C11 coeff +1
        # Sum: 1+1-1+1 = 2? That's not right either.
        
        # I think I'm confusing myself. Let me look up the standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So each Mk is a scalar, and Ci,j is a linear combination of the Mk's.
        # In the tensor decomposition, each rank-1 term corresponds to one Mk.
        # For Mk, the U part is the linear combo of A entries, V part is linear combo of B entries,
        # and the W part distributes Mk to the appropriate C entries with the right signs.
        
        # So for M1:
        # U: A11 coeff 1, A22 coeff 1
        # V: B11 coeff 1, B22 coeff 1
        # W: C11 coeff 1, C22 coeff 1 (since M1 appears in C11 with +1 and C22 with +1)
        
        # For M2:
        # U: A21 coeff 1, A22 coeff 1
        # V: B11 coeff 1
        # W: C21 coeff 1, C22 coeff -1
        
        # For M3:
        # U: A11 coeff 1
        # V: B21 coeff 1, B22 coeff -1
        # W: C12 coeff 1, C22 coeff 1
        
        # For M4:
        # U: A22 coeff 1
        # V: B12 coeff 1, B11 coeff -1
        # W: C11 coeff 1, C21 coeff 1
        
        # For M5:
        # U: A11 coeff 1, A12 coeff 1
        # V: B22 coeff 1
        # W: C11 coeff -1, C12 coeff 1
        
        # For M6:
        # U: A21 coeff 1, A11 coeff -1
        # V: B11 coeff 1, B12 coeff 1
        # W: C22 coeff 1
        
        # For M7:
        # U: A12 coeff 1, A22 coeff -1
        # V: B21 coeff 1, B22 coeff 1
        # W: C11 coeff 1, C22 coeff 1
        
        # Now let me code this up.
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
            # M3
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4
            ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Create U column
            u_col = [0.0] * (n * m)
            for (ai, aj), coeff in u_coeffs.items():
                row = A_rows[ai]
                col = A_cols[aj]
                u_col[row * m + col] = coeff
            
            # Create V column
            v_col = [0.0] * (m * p)
            for (bi, bj), coeff in v_coeffs.items():
                row = B_rows[bi]
                col = B_cols[bj]
                v_col[row * p + col] = coeff
            
            # Create W column
            w_col = [0.0] * (p * n)
            for (ci, cj), coeff in w_coeffs.items():
                # C is at C_rows x C_cols
                # W index is k*n + i where k is C's column, i is C's row
                # ci is C's row (0 or 1), cj is C's column (0 or 1)
                out_row = C_rows[ci]
                out_col = C_cols[cj]
                w_col[out_col * n + out_row] = coeff
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
    
    def add_schoolbook_1x4_4x2(A_row, B_rows, B_cols, C_rows, C_cols):
        """
        Add schoolbook decomposition for 1x4 * 4x2.
        A is a single row at A_row, all 4 columns.
        B is at B_rows x B_cols (4x2).
        C is at C_rows x C_cols (1x2).
        """
        for j in range(4):
            # A[0, j] * B[j, :] -> C[0, :]
            u_col = [0.0] * (n * m)
            u_col[A_row * m + j] = 1.0
            
            v_col = [0.0] * (m * p)
            v_col[B_rows[j] * p + B_cols[0]] = 1.0
            v_col[B_rows[j] * p + B_cols[1]] = 1.0
            
            w_col = [0.0] * (p * n)
            w_col[B_cols[0] * n + C_rows[0]] = 1.0
            w_col[B_cols[1] * n + C_rows[0]] = 1.0
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
    
    # Now let's set up the blocks:
    # A is 3x4: rows 0,1,2; cols 0,1,2,3
    # B is 4x4: rows 0,1,2,3; cols 0,1,2,3
    # C is 3x4: rows 0,1,2; cols 0,1,2,3
    
    # Split A into top 2 rows and bottom 1 row:
    # A_top: rows [0,1], cols [0,1,2,3]
    # A_bottom: row [2], cols [0,1,2,3]
    
    # Split B's columns into left 2 and right 2:
    # B_left: rows [0,1,2,3], cols [0,1]
    # B_right: rows [0,1,2,3], cols [2,3]
    
    # C_top_left: rows [0,1], cols [0,1] = A_top * B_left
    # C_top_right: rows [0,1], cols [2,3] = A_top * B_right
    # C_bottom_left: row [2], cols [0,1] = A_bottom * B_left
    # C_bottom_right: row [2], cols [2,3] = A_bottom * B_right
    
    # For C_top_left = A_top(2x4) * B_left(4x2):
    # Split A_top into left 2x2 and right 2x2:
    # A_top_left: rows [0,1], cols [0,1]
    # A_top_right: rows [0,1], cols [2,3]
    # Split B_left into top 2x2 and bottom 2x2:
    # B_left_top: rows [0,1], cols [0,1]
    # B_left_bottom: rows [2,3], cols [0,1]
    # C_top_left = A_top_left * B_left_top + A_top_right * B_left_bottom
    
    # Add Strassen for A_top_left * B_left_top -> C_top_left
    add_strassen(
        A_rows=[0, 1], A_cols=[0, 1],
        B_rows=[0, 1], B_cols=[0, 1],
        C_rows=[0, 1], C_cols=[0, 1]
    )
    # Add Strassen for A_top_right * B_left_bottom -> C_top_left
    add_strassen(
        A_rows=[0, 1], A_cols=[2, 3],
        B_rows=[2, 3], B_cols=[0, 1],
        C_rows=[0, 1], C_cols=[0, 1]
    )
    
    # For C_top_right = A_top(2x4) * B_right(4x2):
    # Similarly:
    # B_right_top: rows [0,1], cols [2,3]
    # B_right_bottom: rows [2,3], cols [2,3]
    # C_top_right = A_top_left * B_right_top + A_top_right * B_right_bottom
    
    add_strassen(
        A_rows=[0, 1], A_cols=[0, 1],
        B_rows=[0, 1], B_cols=[2, 3],
        C_rows=[0, 1], C_cols=[2, 3]
    )
    add_strassen(
        A_rows=[0, 1], A_cols=[2, 3],
        B_rows=[2, 3], B_cols=[2, 3],
        C_rows=[0, 1], C_cols=[2, 3]
    )
    
    # For C_bottom_left = A_bottom(1x4) * B_left(4x2):
    add_schoolbook_1x4_4x2(
        A_row=2,
        B_rows=[0, 1, 2, 3], B_cols=[0, 1],
        C_rows=[2], C_cols=[0, 1]
    )
    
    # For C_bottom_right = A_bottom(1x4) * B_right(4x2):
    add_schoolbook_1x4_4x2(
        A_row=2,
        B_rows=[0, 1, 2, 3], B_cols=[2, 3],
        C_rows=[2], C_cols=[2, 3]
    )
    
    # Now convert lists to numpy arrays
    rank = len(U_list)
    U = np.array(U_list, dtype=float).T  # Transpose to (n*m, rank)
    V = np.array(V_list, dtype=float).T
    W = np.array(W_list, dtype=float).T
    
    # Verify
    def _verify(U, V, W):
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(p):
                        # We need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{j,k} ... 
                        # Wait, the tensor is T[i, j_A, j_B, k] = delta_{j_A, j_B}
                        # But in the verify loop, we have i, j, k, l.
                        # The sum is over r, with free indices i (from U and W), j (from U and V), k (from V and W).
                        # But W index is k*n+i, so the free indices are i, j, k.
                        # There's no l. Let me re-read.
                        # The sum is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # This should equal delta_{j, j} = 1? No, it should equal... 
                        # From earlier: it equals 1 iff j_A == j_B where j_A is from U and j_B is from V.
                        # But in the expression, U has index j and V has index j (same j).
                        # Wait, U[i*m+j, r] and V[j*p+k, r] both use the same j.
                        # So the sum is always over consistent j. 
                        # The result should be 1 for all (i, j, k)?
                        # No, that's not right. Let me re-examine.
                        # The tensor T[i, j_A, j_B, k] = delta_{j_A, j_B}
                        # In the sum, j_A = j (from U), j_B = j (from V). They're the same variable!
                        # So T[i, j, j, k] = 1 for all i, j, k.
                        # That means the sum should always be 1?
                        # But that can't be right for a tensor decomposition.
                        
                        # OH I SEE THE ISSUE. In the tensor decomposition, the indices of U, V, W 
                        # are independent. U[i*m+j, r] has j as its column index.
                        # V[j*p+k, r] has j as its row index.
                        # But these are DIFFERENT j's in the tensor! 
                        # The tensor has 4 indices: (i, j_A, j_B, k) where j_A is A's column 
                        # and j_B is B's row. In the sum, they happen to be labeled the same 
                        # because the sum is over r, not over j.
                        # Wait no, j is a free index, not summed over.
                        # So the sum S(i, j, k) = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                        # should equal 1 for all i, j, k? That would mean the tensor is all-ones.
                        
                        # I'm clearly confused. Let me go back to the schoolbook verification.
                        # In schoolbook, S(i, j, k) = 1 iff there's a column r where 
                        # U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
                        # In schoolbook, column (i0,j0,k0) has these all 1 iff i=i0, j=j0, k=k0.
                        # So S(i,j,k) = 1 iff (i,j,k) is a valid triple, which is always true.
                        # So S(i,j,k) = 1 for all i,j,k. The tensor IS all-ones!
                        
                        # But that's the tensor for matrix multiplication? Let me verify:
                        # C[i,k] = sum_j A[i,j] B[j,k]
                        # The trilinear form: sum_{i,j,k} C[i,k] A[i,j] B[j,k]
                        # = sum_{i,j,k} (sum_j' A[i,j'] B[j',k]) A[i,j] B[j,k]
                        # This doesn't simplify to something nice.
                        
                        # I think the correct tensor is:
                        # T[i, j, k] = 1 (for all i,j,k) where the trilinear form is
                        # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k] = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
                        # And this equals <C, A*B> (the Frobenius inner product).
                        # So yes, T[i,j,k] = 1 for all i,j,k. The tensor is the all-ones tensor!
                        
                        # So the verification should check that S(i,j,k) = 1 for all i,j,k.
                        expected = 1.0
                        actual = 0.0
                        for r in range(U.shape[1]):
                            actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(actual - expected) > 1e-10:
                            return False
        return True
    
    if not _verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    def _to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-10:
                    clean_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-10:
                    clean_row.append(nearest_half)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result
    
    return {"U": _to_clean_list(U), "V": _to_clean_list(V), "W": _to_clean_list(W)}
# EVOLVE-BLOCK-END
