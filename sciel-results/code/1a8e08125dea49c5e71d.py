"""Program for matrix multiplication 3x4 by 4x5 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Use Strassen's algorithm for the 2x2x2 blocks where possible
    # The matrix multiplication tensor for (n,m,p) = (3,4,5) has 60 entries.
    # We can decompose this into blocks.
    
    # Strategy: Use Strassen for 2x2 blocks where we have 2x2 submatrices
    # For a 3x4 by 4x5 multiplication, we can think of it as:
    # - The 3 rows of A, 5 columns of B
    # - We can group rows of A and columns of B into 2-element groups
    
    # Let's use a block decomposition approach:
    # Group i (rows of A) into: [0,1] and [2]
    # Group k (cols of B) into: [0,1], [2,3], [4]
    # For each pair (i_group, k_group) of size 2x2, use Strassen (rank 7)
    # For remaining, use schoolbook (rank 1 per entry)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_schoolbook(i, j, k):
        """Add a schoolbook term for entry (i,j,k)"""
        U = [0.0] * (n * m)
        V = [0.0] * (m * p)
        W = [0.0] * (p * n)
        U[i * m + j] = 1.0
        V[j * p + k] = 1.0
        W[k * n + i] = 1.0
        U_cols.append(U)
        V_cols.append(V)
        W_cols.append(W)
    
    def add_strassen_block(i0, k0, j_start, j_end):
        """
        Add Strassen's rank-7 decomposition for a 2x2 block.
        i0: starting row index for A (2 rows: i0, i0+1)
        k0: starting col index for B (2 cols: k0, k0+1)
        j_start, j_end: range of j indices to process (all m columns)
        """
        # Strassen's 7 products for 2x2 matrix multiplication
        # For each j in the range, we compute M1..M7
        
        # The Strassen algorithm computes:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # In our case, A is 2xm (rows i0, i0+1, cols j_start..j_end-1)
        # B is m x 2 (rows j_start..j_end-1, cols k0, k0+1)
        
        # For Strassen, we need to handle each column of A and row of B
        # Actually, Strassen works on 2x2 matrices, so we need to be careful.
        
        # Let me reconsider. For the full m columns, we can't directly apply
        # Strassen unless we also group the j dimension.
        
        # Alternative: Use Strassen only when we can form 2x2x2 sub-problems.
        # For a 2x2 block in (i,k), we still have m columns in j.
        # So we'd need to also group j into pairs.
        
        # Let's use a different approach: enumerate all possible 2x2x2 sub-blocks
        # and apply Strassen to them, then fill the rest with schoolbook.
        pass
    
    # Let's use a cleaner approach:
    # For each combination of:
    #   - 2 consecutive i values (or 1 if at boundary)
    #   - 2 consecutive j values (or 1 if at boundary)  
    #   - 2 consecutive k values (or 1 if at boundary)
    # If all three are size 2, use Strassen (7 terms instead of 8)
    # Otherwise, use schoolbook for each entry.
    
    # Define which (i,j,k) triples are covered by Strassen blocks
    strassen_covered = set()
    
    # Possible i groups: [0,1], [1,2] (overlapping) or just [0,1]
    # To avoid overlap, let's use non-overlapping groups:
    # i: [0,1], [2]
    # j: [0,1], [2,3]
    # k: [0,1], [2,3], [4]
    
    # So Strassen blocks for (i_group, j_group, k_group) where all are size 2:
    # i_group [0,1], j_group [0,1], k_group [0,1]
    # i_group [0,1], j_group [0,1], k_group [2,3]
    # i_group [0,1], j_group [2,3], k_group [0,1]
    # i_group [0,1], j_group [2,3], k_group [2,3]
    
    i_groups = [[0, 1]]  # only one group of size 2
    j_groups = [[0, 1], [2, 3]]  # two groups of size 2
    k_groups = [[0, 1], [2, 3]]  # two groups of size 2
    
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                if len(ig) == 2 and len(jg) == 2 and len(kg) == 2:
                    # Apply Strassen to this 2x2x2 block
                    i0, i1 = ig
                    j0, j1 = jg
                    k0, k1 = kg
                    
                    # Mark these 8 entries as covered
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                strassen_covered.add((i0+ii, j0+jj, k0+kk))
                    
                    # Add Strassen's 7 terms
                    # Local indices: A[0,0]=A[i0,j0], A[0,1]=A[i0,j1]
                    #                A[1,0]=A[i1,j0], A[1,1]=A[i1,j1]
                    # B[0,0]=B[j0,k0], B[0,1]=B[j0,k1]
                    #         B[1,0]=B[j1,k0], B[1,1]=B[j1,k1]
                    # C[0,0]=C[i0,k0], C[0,1]=C[i0,k1]
                    #         C[1,0]=C[i1,k0], C[1,1]=C[i1,k1]
                    
                    # M1 = (A00 + A11)(B00 + B11)
                    # U: A[i0,j0]=1, A[i1,j1]=1
                    # V: B[j0,k0]=1, B[j1,k1]=1
                    # W: C[i0,k0]=1, C[i1,k1]=1
                    U = [0.0] * (n * m)
                    V = [0.0] * (m * p)
                    W = [0.0] * (p * n)
                    U[i0 * m + j0] = 1.0
                    U[i1 * m + j1] = 1.0
                    V[j0 * p + k0] = 1.0
                    V[j1 * p + k1] = 1.0
                    W[k0 * n + i0] = 1.0
                    W[k1 * n + i1] = 1.0
                    U_cols.append(U)
                    V_cols.append(V)
                    W_cols.append(W)
                    
                    # M2 = (A10 + A11)B00
                    # U: A[i1,j0]=1, A[i1,j1]=1
                    # V: B[j0,k0]=1
                    # W: C[i1,k0]=1
                    U = [0.0] * (n * m)
                    V = [0.0] * (m * p)
                    W = [0.0] * (p * n)
                    U[i1 * m + j0] = 1.0
                    U[i1 * m + j1] = 1.0
                    V[j0 * p + k0] = 1.0
                    W[k0 * n + i1] = 1.0
                    U_cols.append(U)
                    V_cols.append(V)
                    W_cols.append(W)
                    
                    # M3 = A00(B01 - B11)
                    # U: A[i0,j0]=1
                    # V: B[j0,k1]=1, B[j1,k1]=-1
                    # W: C[i0,k1]=1
                    U = [0.0] * (n * m)
                    V = [0.0] * (m * p)
                    W = [0.0] * (p * n)
                    U[i0 * m + j0] = 1.0
                    V[j0 * p + k1] = 1.0
                    V[j1 * p + k1] = -1.0
                    W[k1 * n + i0] = 1.0
                    U_cols.append(U)
                    V_cols.append(V)
                    W_cols.append(W)
                    
                    # M4 = A11(B10 - B00)
                    # U: A[i1,j1]=1
                    # V: B[j1,k0]=1, B[j0,k0]=-1
                    # W: C[i1,k0]=1
                    U = [0.0] * (n * m)
                    V = [0.0] * (m * p)
                    W = [0.0] * (p * n)
                    U[i1 * m + j1] = 1.0
                    V[j1 * p + k0] = 1.0
                    V[j0 * p + k0] = -1.0
                    W[k0 * n + i1] = 1.0
                    U_cols.append(U)
                    V_cols.append(V)
                    W_cols.append(W)
                    
                    # M5 = (A00 + A01)B11
                    # U: A[i0,j0]=1, A[i0,j1]=1
                    # V: B[j1,k1]=1
                    # W: C[i0,k1]=1
                    U = [0.0] * (n * m)
                    V = [0.0] * (m * p)
                    W = [0.0] * (p * n)
                    U[i0 * m + j0] = 1.0
                    U[i0 * m + j1] = 1.0
                    V[j1 * p + k1] = 1.0
                    W[k1 * n + i0] = 1.0
                    U_cols.append(U)
                    V_cols.append(V)
                    W_cols.append(W)
                    
                    # M6 = (A10 - A00)(B00 + B01)
                    # U: A[i1,j0]=1, A[i0,j0]=-1
                    # V: B[j0,k0]=1, B[j0,k1]=1
                    # W: C[i1,k1]=1
                    U = [0.0] * (n * m)
                    V = [0.0] * (m * p)
                    W = [0.0] * (p * n)
                    U[i1 * m + j0] = 1.0
                    U[i0 * m + j0] = -1.0
                    V[j0 * p + k0] = 1.0
                    V[j0 * p + k1] = 1.0
                    W[k1 * n + i1] = 1.0
                    U_cols.append(U)
                    V_cols.append(V)
                    W_cols.append(W)
                    
                    # M7 = (A01 - A11)(B10 + B11)
                    # U: A[i0,j1]=1, A[i1,j1]=-1
                    # V: B[j1,k0]=1, B[j1,k1]=1
                    # W: C[i0,k0]=1
                    U = [0.0] * (n * m)
                    V = [0.0] * (m * p)
                    W = [0.0] * (p * n)
                    U[i0 * m + j1] = 1.0
                    U[i1 * m + j1] = -1.0
                    V[j1 * p + k0] = 1.0
                    V[j1 * p + k1] = 1.0
                    W[k0 * n + i0] = 1.0
                    U_cols.append(U)
                    V_cols.append(V)
                    W_cols.append(W)
    
    # Now add schoolbook terms for all entries not covered by Strassen
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    add_schoolbook(i, j, k)
    
    rank = len(U_cols)
    
    # Convert to 2D lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for r in range(n * m):
            U[r][c] = U_cols[c][r]
        for r in range(m * p):
            V[r][c] = V_cols[c][r]
        for r in range(p * n):
            W[r][c] = W_cols[c][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
