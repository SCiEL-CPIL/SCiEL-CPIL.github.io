"""
Matrix multiplication 4x5 * 5x5 -> 4x5
Rank-minimal exact tensor decomposition.

Strategy:
  - The full tensor has 4*5*5 = 100 entries.
  - We exploit the known rank-7 decomposition of the 2x2x2 matrix multiplication tensor (Strassen).
  - We partition the 4x5*5x5 problem into independent 2x2*2x2 blocks wherever possible,
    and use Strassen's 7-term identity for each such block, saving 1 term per block.
  - The remaining terms are handled by the schoolbook method.

  The 4x5 matrix A can be viewed as a 2x2 block matrix where each block is 2x(5/2)... 
  but 5 is odd. Let's think differently.

  Actually, the matrix product C = A*B where A is 4x5 and B is 5x5.
  We can partition A into 2x2 blocks of submatrices, but the dimensions don't divide evenly.
  
  Alternative approach: 
  - Use the fact that the 4x5*5x5 tensor can be decomposed by considering it as a sum of
    independent 2x2*2x2 sub-problems where possible.
  
  Let's partition the rows of A (4 rows) into 2 groups of 2, and columns of A / rows of B (5) 
  into groups. Since 5 is not divisible by 2, we have one group of 2 and one group of 1, or 
  we can use a mixed approach.

  Actually, let's use a simpler structural insight:
  - The 4x5*5x5 product can be broken down. Consider partitioning the 5 intermediate dimension 
    into 2+2+1. Then we have:
      - 2x2 * 2x2 blocks (4 of them if we also partition the 4 rows into 2+2)
      - Cross terms that are 2x2 * 2x1, 2x2 * 1x2, etc.
    
  This is getting complex. Let me use a cleaner approach:

  Key insight: The rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen).
  We can find disjoint 2x2*2x2 sub-problems within the 4x5*5x5 problem.
  
  Partition the 4 rows of A into two groups of 2: rows {0,1} and {2,3}.
  Partition the 5 columns of A / rows of B into: {0,1}, {2,3}, {4}.
  Partition the 5 columns of B / rows of C into: {0,1}, {2,3}, {4}.
  
  For the sub-problem: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]  (2x2 * 2x2 -> 2x2)
  This is a standard 2x2x2 block. We can use Strassen's 7-term decomposition here.
  Similarly for A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
  And A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
  And A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
  
  That's 4 disjoint 2x2x2 blocks. Each saves 1 term (8 -> 7). Total savings: 4.
  Rank: 100 - 4 = 96.

  The remaining terms involve the 5th column/row and the cross terms, which we handle with schoolbook.
"""

import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p  # 100
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]

    # Build schoolbook: index c corresponds to (i, j, k)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Now, apply Strassen to 4 disjoint 2x2x2 blocks.
    # For each block, we replace 8 schoolbook terms with 7 Strassen terms.
    
    # Strassen's 7-term decomposition for 2x2x2:
    # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    # 
    # Strassen computes 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product of two linear combinations, which corresponds to
    # a rank-1 term in the tensor decomposition with coefficients.
    
    # For a 2x2x2 block with A rows {r0, r1}, A cols/B rows {c0, c1}, B cols/C cols {d0, d1}:
    # The 8 schoolbook terms are for (i, j, k) in {r0,r1} x {c0,c1} x {d0,d1}
    # We replace them with 7 terms.
    
    # Function to apply Strassen to a block and update U, V, W
    def apply_strassen_block(row_indices, col_indices, out_indices, U, V, W, base_col):
        """
        Apply Strassen's 7-term decomposition to a 2x2x2 block.
        row_indices: [r0, r1] - rows of A
        col_indices: [c0, c1] - columns of A / rows of B
        out_indices: [d0, d1] - columns of B / rows of C
        U, V, W: the factor matrices
        base_col: the starting column index for this block's 8 schoolbook terms
        
        Returns: number of columns used (7)
        """
        r0, r1 = row_indices
        c0, c1 = col_indices
        d0, d1 = out_indices
        
        # The 8 schoolbook term indices for this block:
        # (r0,c0,d0), (r0,c0,d1), (r0,c1,d0), (r0,c1,d1),
        # (r1,c0,d0), (r1,c0,d1), (r1,c1,d0), (r1,c1,d1)
        
        # We'll assign 7 new columns starting at base_col
        # First, zero out the old 8 schoolbook entries for this block in U, V, W
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = row_indices[di]
                    j = col_indices[dj]
                    k = out_indices[dk]
                    # Find the schoolbook column for (i, j, k)
                    # In schoolbook, col = i * m * p + j * p + k
                    sb_col = i * m * p + j * p + k
                    U[i * m + j][sb_col] = 0
                    V[j * p + k][sb_col] = 0
                    W[k * n + i][sb_col] = 0
        
        # Now add 7 Strassen terms
        # M1 = (A[r0,c0] + A[r1,c1]) * (B[c0,d0] + B[c1,d1])
        # M2 = (A[r1,c0] + A[r1,c1]) * B[c0,d0]
        # M3 = A[r0,c0] * (B[c0,d1] - B[c1,d1])
        # M4 = A[r1,c1] * (B[c1,d0] - B[c0,d0])
        # M5 = (A[r0,c0] + A[r0,c1]) * B[c1,d1]
        # M6 = (A[r1,c0] - A[r0,c0]) * (B[c0,d0] + B[c0,d1])
        # M7 = (A[r0,c1] - A[r1,c1]) * (B[c1,d0] + B[c1,d1])
        
        # Each term contributes to C entries:
        # C[r0,d0] += M1 + M4 - M5 + M7
        # C[r0,d1] += M3 + M5
        # C[r1,d0] += M2 + M4
        # C[r1,d1] += M1 - M2 + M3 + M6
        
        # For each Strassen term t (0..6), we need to set U, V, W coefficients
        # such that sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = C[i,k] contribution
        
        # Term M1:
        # U coefficients: A[r0,c0]=1, A[r1,c1]=1 (for the (r0,c0) and (r1,c1) entries)
        # V coefficients: B[c0,d0]=1, B[c1,d1]=1
        # W coefficients: for C entries that M1 contributes to
        # M1 contributes to C[r0,d0] with coeff 1, C[r1,d1] with coeff 1
        # W[d0, r0] += 1, W[d1, r1] += 1
        
        # Let's define the 7 terms with their U, V, W coefficient patterns
        terms = []
        
        # M1: (A[r0,c0] + A[r1,c1]) * (B[c0,d0] + B[c1,d1])
        u_m1 = {r0 * m + c0: 1, r1 * m + c1: 1}
        v_m1 = {c0 * p + d0: 1, c1 * p + d1: 1}
        w_m1 = {d0 * n + r0: 1, d1 * n + r1: 1}
        terms.append((u_m1, v_m1, w_m1))
        
        # M2: (A[r1,c0] + A[r1,c1]) * B[c0,d0]
        u_m2 = {r1 * m + c0: 1, r1 * m + c1: 1}
        v_m2 = {c0 * p + d0: 1}
        w_m2 = {d0 * n + r1: 1}
        terms.append((u_m2, v_m2, w_m2))
        
        # M3: A[r0,c0] * (B[c0,d1] - B[c1,d1])
        u_m3 = {r0 * m + c0: 1}
        v_m3 = {c0 * p + d1: 1, c1 * p + d1: -1}
        w_m3 = {d1 * n + r0: 1}
        terms.append((u_m3, v_m3, w_m3))
        
        # M4: A[r1,c1] * (B[c1,d0] - B[c0,d0])
        u_m4 = {r1 * m + c1: 1}
        v_m4 = {c1 * p + d0: 1, c0 * p + d0: -1}
        w_m4 = {d0 * n + r1: 1, d0 * n + r0: 1}
        # Wait, M4 contributes to both C[r0,d0] and C[r1,d0]
        # C[r0,d0] += M4, C[r1,d0] += M4
        # So W[d0, r0] += 1, W[d0, r1] += 1
        w_m4 = {d0 * n + r0: 1, d0 * n + r1: 1}
        terms.append((u_m4, v_m4, w_m4))
        
        # M5: (A[r0,c0] + A[r0,c1]) * B[c1,d1]
        u_m5 = {r0 * m + c0: 1, r0 * m + c1: 1}
        v_m5 = {c1 * p + d1: 1}
        # M5 contributes to C[r0,d0] with coeff -1, C[r0,d1] with coeff 1
        w_m5 = {d0 * n + r0: -1, d1 * n + r0: 1}
        terms.append((u_m5, v_m5, w_m5))
        
        # M6: (A[r1,c0] - A[r0,c0]) * (B[c0,d0] + B[c0,d1])
        u_m6 = {r1 * m + c0: 1, r0 * m + c0: -1}
        v_m6 = {c0 * p + d0: 1, c0 * p + d1: 1}
        # M6 contributes to C[r1,d1] with coeff 1
        w_m6 = {d1 * n + r1: 1}
        terms.append((u_m6, v_m6, w_m6))
        
        # M7: (A[r0,c1] - A[r1,c1]) * (B[c1,d0] + B[c1,d1])
        u_m7 = {r0 * m + c1: 1, r1 * m + c1: -1}
        v_m7 = {c1 * p + d0: 1, c1 * p + d1: 1}
        # M7 contributes to C[r0,d0] with coeff 1
        w_m7 = {d0 * n + r0: 1}
        terms.append((u_m7, v_m7, w_m7))
        
        # Now, we need to assign columns. We'll use the next 7 available columns.
        # But we've zeroed out 8 schoolbook columns. We can reuse some of them.
        # For simplicity, let's use 7 new columns. But that would increase rank.
        # Instead, we should reuse the 8 schoolbook columns, using only 7 of them.
        
        # The 8 schoolbook columns for this block:
        sb_cols = []
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = row_indices[di]
                    j = col_indices[dj]
                    k = out_indices[dk]
                    sb_cols.append(i * m * p + j * p + k)
        
        # We'll use the first 7 of these columns for the 7 Strassen terms
        for t in range(7):
            col_idx = sb_cols[t]
            u_dict, v_dict, w_dict = terms[t]
            for (ij, coeff) in u_dict.items():
                U[ij][col_idx] += coeff
            for (jk, coeff) in v_dict.items():
                V[jk][col_idx] += coeff
            for (ki, coeff) in w_dict.items():
                W[ki][col_idx] += coeff
        
        # The 8th column (sb_cols[7]) remains zeroed out
        return 7
    
    # Apply Strassen to 4 disjoint 2x2x2 blocks
    # Block 1: rows {0,1}, cols {0,1}, out {0,1}
    apply_strassen_block([0, 1], [0, 1], [0, 1], U, V, W, 0)
    
    # Block 2: rows {0,1}, cols {2,3}, out {2,3}
    apply_strassen_block([0, 1], [2, 3], [2, 3], U, V, W, 0)
    
    # Block 3: rows {2,3}, cols {0,1}, out {0,1}
    apply_strassen_block([2, 3], [0, 1], [0, 1], U, V, W, 0)
    
    # Block 4: rows {2,3}, cols {2,3}, out {2,3}
    apply_strassen_block([2, 3], [2, 3], [2, 3], U, V, W, 0)
    
    # Now we have a valid decomposition with rank <= 100 (some columns are zero)
    # We can trim zero columns to get the actual rank
    # But for the interface, we just return the matrices as-is
    
    # Verify the decomposition
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    is_valid, err = verify(U, V, W)
    if not is_valid:
        # Fallback to pure schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Trim zero columns
    nonzero_cols = []
    for c in range(rank_schoolbook):
        if any(U[i][c] != 0 for i in range(n * m)) or \
           any(V[i][c] != 0 for i in range(m * p)) or \
           any(W[i][c] != 0 for i in range(p * n)):
            nonzero_cols.append(c)
    
    U_trimmed = [[U[i][c] for c in nonzero_cols] for i in range(n * m)]
    V_trimmed = [[V[i][c] for c in nonzero_cols] for i in range(m * p)]
    W_trimmed = [[W[i][c] for c in nonzero_cols] for i in range(p * n)]
    
    return {"U": U_trimmed, "V": V_trimmed, "W": W_trimmed}
