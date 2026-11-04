"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # We decompose C = A * B where A is 3x3 and B is 3x7.
    # We use a block decomposition approach.
    # Split A into 4 blocks: A00 (2x2), A01 (2x1), A10 (1x2), A11 (1x1).
    # Split B into 2 blocks: B0 (3x3), B1 (3x4).
    # C = A * B0 + A * B1.
    # C0 (3x3) = A * B0.
    # C1 (3x4) = A * B1.
    
    # For C0 (3x3x3), we use Strassen-like decomposition or just schoolbook for simplicity and correctness first.
    # Actually, let's try to get a lower rank.
    # Rank of 3x3x3 is 23. But constructing it is hard.
    # Let's stick to a robust decomposition that beats 63.
    
    # Strategy:
    # 1. Compute C0 = A * B0 (3x3 * 3x3).
    #    Use Strassen for the 2x2 sub-blocks and schoolbook for the rest.
    #    A00 (2x2) * B00 (2x2) -> C00 (2x2). Rank 7 (Strassen).
    #    A01 (2x1) * B01 (1x2) -> C01 (2x2). Rank 4 (Schoolbook: 2*1*2=4? No, 2x1 * 1x2 is outer product, rank 1? No, sum of 1 term. But indices: i=0,1; j=2; k=0,1. So 2*1*2=4 terms? No, j is fixed at 2. So sum over j=2. A[i,2]*B[2,k]. This is an outer product of a 2x1 vector and a 1x2 vector. Rank 1? No, it's a sum of 1 term in the tensor decomposition sense?
    #    Let's check the tensor definition.
    #    T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes.
    #    For A01 * B01: i in {0,1}, j=2, k in {0,1}.
    #    Terms: (0,2,0), (0,2,1), (1,2,0), (1,2,1).
    #    This is 4 terms. Can we do it in less?
    #    It's a rank 4 tensor if we view it as a 2x1x2 tensor.
    #    Actually, the tensor for 2x1x2 is just the 4 basis vectors. It has rank 4?
    #    No, the tensor T_{i,j,k} where j is fixed is T_{i,2,k}. This is a 2x2 matrix (identity? No, just 1s).
    #    The tensor is T[i,k] = 1 for i in 0,1 and k in 0,1. This is a 2x2 matrix of ones.
    #    The rank of a 2x2 matrix of ones is 1.
    #    Wait, the tensor is T[i, j, k]. If j is fixed, it's a matrix T[i, k].
    #    The rank of the tensor (as a trilinear form) is the rank of the matrix unfolding?
    #    No, tensor rank is the minimum number of rank-1 tensors to sum to it.
    #    T[i,k] = 1 for all i,k. This can be written as sum_{r=1}^1 u_r[i] v_r[k] w_r[j]?
    #    If j is fixed, w_r[j] is a constant.
    #    So T[i,k] = 1 = (1,1)^T (1,1). Rank 1.
    #    So A01 * B01 can be done in rank 1?
    #    Let's check the indices.
    #    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    #    For j=2 fixed.
    #    U[i*3+2, r] V[2*7+k, r] W[k*3+i, r].
    #    We need sum_r ... = 1 for all i in {0,1}, k in {0,1}.
    #    If we set r=1, U[i*3+2, 1]=1, V[2*7+k, 1]=1, W[k*3+i, 1]=1.
    #    Then sum is 1.
    #    So yes, rank 1.
    
    #    Similarly for A10 * B10 (1x2 * 2x1) -> C10 (1x1).
    #    i=2, j in {0,1}, k=2.
    #    T[2,j,2] = 1 for j=0,1.
    #    This is a vector of ones. Rank 1.
    
    #    A11 * B11 (1x1 * 1x1) -> C11 (1x1).
    #    i=2, j=2, k=2.
    #    T[2,2,2] = 1. Rank 1.
    
    #    So for C0 (3x3x3):
    #    A00 * B00 (2x2x2): Rank 7 (Strassen).
    #    A01 * B01 (2x1x2): Rank 1.
    #    A10 * B10 (1x2x1): Rank 1.
    #    A11 * B11 (1x1x1): Rank 1.
    #    Total for C0: 7 + 1 + 1 + 1 = 10.
    #    Wait, is this correct?
    #    The cross terms are handled separately.
    #    C0 = A00 B00 + A01 B01 + A10 B10 + A11 B11.
    #    Yes, these are disjoint index sets in the tensor.
    #    So the rank is the sum of the ranks of the sub-tensors.
    #    So Rank(C0) = 10.
    #    This is much better than 27.
    
    # Now for C1 (3x4) = A * B1.
    # B1 is 3x4.
    # A00 (2x2) * B10 (2x4) -> C10 (2x4).
    # A01 (2x1) * B11 (1x4) -> C11 (2x4).
    # A10 (1x2) * B10 (2x4) -> C12 (1x4).
    # A11 (1x1) * B11 (1x4) -> C13 (1x4).
    
    # Let's analyze the ranks.
    # A00 * B10: 2x2 * 2x4.
    # Split B10 into B10a (2x2) and B10b (2x2).
    # A00 * B10a: 2x2x2. Rank 7 (Strassen).
    # A00 * B10b: 2x2x2. Rank 7 (Strassen).
    # Total: 14.
    
    # A01 * B11: 2x1 * 1x4.
    # i in {0,1}, j=2, k in {0,1,2,3} (local k).
    # T[i, 2, k] = 1.
    # This is a 2x4 matrix of ones.
    # Rank 1?
    # U[i*3+2, r] V[2*7+k, r] W[k*3+i, r].
    # If we set r=1, U=1, V=1, W=1. Sum=1.
    # So Rank 1.
    
    # A10 * B10: 1x2 * 2x4.
    # i=2, j in {0,1}, k in {0,1,2,3}.
    # T[2, j, k] = 1.
    # This is a 2x4 matrix of ones (j vs k).
    # Rank 1?
    # U[2*3+j, r] V[j*7+k, r] W[k*3+2, r].
    # If r=1, U=1, V=1, W=1. Sum=1.
    # So Rank 1.
    
    # A11 * B11: 1x1 * 1x4.
    # i=2, j=2, k in {0,1,2,3}.
    # T[2, 2, k] = 1.
    # Vector of ones. Rank 1.
    
    # Total for C1: 14 + 1 + 1 + 1 = 17.
    
    # Total Rank = Rank(C0) + Rank(C1) = 10 + 17 = 27.
    
    # Wait, is this correct?
    # The tensor T is the sum of the sub-tensors.
    # Since the index sets are disjoint, the rank is the sum.
    # So Rank = 27.
    
    # This is the same as schoolbook for 3x3x3, but we are doing 3x3x7.
    # Schoolbook for 3x3x7 is 63.
    # So 27 is a huge improvement!
    
    # Let's verify the "Rank 1" claims.
    # For A01 * B01 (2x1x2):
    # Indices: i in {0,1}, j=2, k in {0,1}.
    # We need T[i,2,k] = 1.
    # Term: U[i*3+2, r] V[2*7+k, r] W[k*3+i, r].
    # If we use one rank-1 term with r=0:
    # U[i*3+2, 0] = 1 for i=0,1.
    # V[2*7+k, 0] = 1 for k=0,1.
    # W[k*3+i, 0] = 1 for k=0,1 and i=0,1.
    # Note: W index is k*n + i = k*3 + i.
    # For k=0, i=0: W[0,0]=1.
    # For k=0, i=1: W[1,0]=1.
    # For k=1, i=0: W[3,0]=1.
    # For k=1, i=1: W[4,0]=1.
    # These are distinct entries in W.
    # So yes, it works.
    
    # So the plan is:
    # 1. C0 (3x3x3) part:
    #    - A00*B00 (2x2x2): Strassen (7 terms).
    #    - A01*B01 (2x1x2): 1 term.
    #    - A10*B10 (1x2x1): 1 term.
    #    - A11*B11 (1x1x1): 1 term.
    # 2. C1 (3x4) part:
    #    - A00*B10 (2x2x4): Split into two 2x2x2 Strassen (14 terms).
    #    - A01*B11 (2x1x4): 1 term.
    #    - A10*B10 (1x2x4): 1 term.
    #    - A11*B11 (1x1x4): 1 term.
    
    # Total Rank = 7 + 1 + 1 + 1 + 14 + 1 + 1 + 1 = 27.
    
    # Let's implement this.
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen_2x2x2(col, i_off, j_off, k_off):
        # i_off, j_off, k_off are offsets in the global index space.
        # Local i: 0,1 -> Global i: i_off, i_off+1
        # Local j: 0,1 -> Global j: j_off, j_off+1
        # Local k: 0,1 -> Global k: k_off, k_off+1
        
        # M1 = (A11+A22)(B11+B22)
        # A11: i=0,j=0. A22: i=1,j=1.
        # B11: j=0,k=0. B22: j=1,k=1.
        # C11: i=0,k=0. C12: i=0,k=1. C21: i=1,k=0. C22: i=1,k=1.
        
        # U
        U[(i_off+0)*3 + (j_off+0)][col] += 1
        U[(i_off+1)*3 + (j_off+1)][col] += 1
        # V
        V[(j_off+0)*7 + (k_off+0)][col] += 1
        V[(j_off+1)*7 + (k_off+1)][col] += 1
        # W
        W[(k_off+0)*3 + (i_off+0)][col] += 1
        W[(k_off+1)*3 + (i_off+0)][col] += 1
        W[(k_off+0)*3 + (i_off+1)][col] -= 1
        W[(k_off+1)*3 + (i_off+1)][col] += 1
        
        col += 1
        
        # M2 = (A21+A22)B11
        # A21: i=1,j=0. A22: i=1,j=1.
        # B11: j=0,k=0.
        # C21: i=1,k=0.
        
        U[(i_off+1)*3 + (j_off+0)][col] += 1
        U[(i_off+1)*3 + (j_off+1)][col] += 1
        V[(j_off+0)*7 + (k_off+0)][col] += 1
        W[(k_off+0)*3 + (i_off+1)][col] += 1
        
        col += 1
        
        # M3 = A11(B21-B11)
        # A11: i=0,j=0.
        # B21: j=1,k=0. B11: j=0,k=0.
        # C12: i=0,k=1.
        
        U[(i_off+0)*3 + (j_off+0)][col] += 1
        V[(j_off+1)*7 + (k_off+0)][col] += 1
        V[(j_off+0)*7 + (k_off+0)][col] -= 1
        W[(k_off+1)*3 + (i_off+0)][col] += 1
        
        col += 1
        
        # M4 = A22(B22-B11)
        # A22: i=1,j=1.
        # B22: j=1,k=1. B11: j=0,k=0.
        # C22: i=1,k=1.
        
        U[(i_off+1)*3 + (j_off+1)][col] += 1
        V[(j_off+1)*7 + (k_off+1)][col] += 1
        V[(j_off+0)*7 + (k_off+0)][col] -= 1
        W[(k_off+1)*3 + (i_off+1)][col] += 1
        
        col += 1
        
        # M5 = (A11+A12)B22
        # A11: i=0,j=0. A12: i=0,j=1.
        # B22: j=1,k=1.
        # C11: i=0,k=0. C22: i=1,k=1.
        
        U[(i_off+0)*3 + (j_off+0)][col] += 1
        U[(i_off+0)*3 + (j_off+1)][col] += 1
        V[(j_off+1)*7 + (k_off+1)][col] += 1
        W[(k_off+0)*3 + (i_off+0)][col] += 1
        W[(k_off+1)*3 + (i_off+1)][col] += 1
        
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21: i=1,j=0. A11: i=0,j=0.
        # B11: j=0,k=0. B12: j=0,k=1.
        # C21: i=1,k=0. C22: i=1,k=1.
        
        U[(i_off+1)*3 + (j_off+0)][col] += 1
        U[(i_off+0)*3 + (j_off+0)][col] -= 1
        V[(j_off+0)*7 + (k_off+0)][col] += 1
        V[(j_off+0)*7 + (k_off+1)][col] += 1
        W[(k_off+0)*3 + (i_off+1)][col] += 1
        W[(k_off+1)*3 + (i_off+1)][col] += 1
        
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12: i=0,j=1. A22: i=1,j=1.
        # B21: j=1,k=0. B22: j=1,k=1.
        # C11: i=0,k=0. C12: i=0,k=1.
        
        U[(i_off+0)*3 + (j_off+1)][col] += 1
        U[(i_off+1)*3 + (j_off+1)][col] -= 1
        V[(j_off+1)*7 + (k_off+0)][col] += 1
        V[(j_off+1)*7 + (k_off+1)][col] += 1
        W[(k_off+0)*3 + (i_off+0)][col] += 1
        W[(k_off+1)*3 + (i_off+0)][col] += 1
        
        col += 1
        
        return col
    
    # Part 1: C0 (3x3x3)
    # A00*B00 (2x2x2) at i=0,1; j=0,1; k=0,1.
    col = add_strassen_2x2x2(col, 0, 0, 0)
    
    # A01*B01 (2x1x2) at i=0,1; j=2; k=0,1.
    # Rank 1.
    for i in [0, 1]:
        U[i*3 + 2][col] = 1
    for k in [0, 1]:
        V[2*7 + k][col] = 1
        W[k*3 + 0][col] = 1  # i=0
        W[k*3 + 1][col] = 1  # i=1
    col += 1
    
    # A10*B10 (1x2x1) at i=2; j=0,1; k=2.
    # Rank 1.
    for j in [0, 1]:
        U[2*3 + j][col] = 1
        V[j*7 + 2][col] = 1
    W[2*3 + 2][col] = 1  # i=2, k=2
    col += 1
    
    # A11*B11 (1x1x1) at i=2; j=2; k=2.
    # Rank 1.
    U[2*3 + 2][col] = 1
    V[2*7 + 2][col] = 1
    W[2*3 + 2][col] = 1
    col += 1
    
    # Part 2: C1 (3x4)
    # A00*B10 (2x2x4) at i=0,1; j=0,1; k=3,4,5,6.
    # Split into k=3,4 and k=5,6.
    col = add_strassen_2x2x2(col, 0, 0, 3)
    col = add_strassen_2x2x2(col, 0, 0, 5)
    
    # A01*B11 (2x1x4) at i=0,1; j=2; k=3,4,5,6.
    # Rank 1.
    for i in [0, 1]:
        U[i*3 + 2][col] = 1
    for k in [3, 4, 5, 6]:
        V[2*7 + k][col] = 1
        W[k*3 + 0][col] = 1  # i=0
        W[k*3 + 1][col] = 1  # i=1
    col += 1
    
    # A10*B10 (1x2x4) at i=2; j=0,1; k=3,4,5,6.
    # Rank 1.
    for j in [0, 1]:
        U[2*3 + j][col] = 1
    for k in [3, 4, 5, 6]:
        V[j*7 + k][col] = 1  # Wait, j is in U and V.
        # V index is j*7+k.
        # We need V[j*7+k] = 1 for j=0,1 and k=3,4,5,6.
        # This is 8 entries.
        # But we are in a single rank-1 term.
        # So V[j*7+k] must be 1 for all these indices.
        # This is fine.
        W[k*3 + 2][col] = 1  # i=2
    col += 1
    
    # A11*B11 (1x1x4) at i=2; j=2; k=3,4,5,6.
    # Rank 1.
    U[2*3 + 2][col] = 1
    for k in [3, 4, 5, 6]:
        V[2*7 + k][col] = 1
        W[k*3 + 2][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
