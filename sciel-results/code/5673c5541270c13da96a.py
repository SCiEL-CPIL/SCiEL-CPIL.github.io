"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-96 exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Strategy:
    - Partition the 4x5 matrix A into four 2x2 blocks (rows 0-1,2-3; cols 0-1,2-4)
      and the 5x5 matrix B into four 2x2 blocks (rows 0-1,2-4; cols 0-1,2-4).
      Wait, this doesn't work cleanly because B has 5 rows and 5 cols.
      
    Better approach: Use the knowledge from the previous observation.
    The 4x5x5 tensor can be decomposed using 4 disjoint 2x2x2 sub-problems.
    But 4x5x5 = 100 scalar multiplications in schoolbook.
    
    Actually, let me think more carefully. The matrix multiplication tensor for
    (n,m,p) has rank at most n*m*p (schoolbook). For 4,5,5 that's 100.
    
    The key insight from the previous measurement: Replacing 4 disjoint 2x2x2 
    standard sub-problems with Strassen's 7-term decomposition reduces rank from 
    100 to 96.
    
    How does this work? We need to identify 4 disjoint 2x2x2 sub-tensors in the 
    4x5x5 multiplication tensor.
    
    The 4x5 matrix A has entries A[i,j] for i in 0..3, j in 0..4.
    The 5x5 matrix B has entries B[j,k] for j in 0..4, k in 0..4.
    The 4x5 output C has entries C[i,k] for i in 0..3, k in 0..4.
    
    C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    We can partition this into blocks. Let's use:
    - For i in {0,1}, j in {0,1}, k in {0,1}: one 2x2x2 sub-tensor
    - For i in {0,1}, j in {2,3}, k in {0,1}: another 2x2x2 sub-tensor (but j only goes to 4)
    
    Actually, let me reconsider. The standard approach would be:
    Partition A into 2x2 blocks along rows and columns, but the dimensions don't divide evenly.
    
    Alternative: Use a hybrid approach. 
    - 4 rows x 5 columns for A
    - 5 rows x 5 columns for B  
    - 4 rows x 5 columns for C
    
    We can view this as:
    - First 2 rows of A (2x5) times first 2 rows of B (2x5) -> contributes to first 2 rows of C
    - etc.
    
    Let me try a different partition. Consider:
    - Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    - Block 2: i in {0,1}, j in {2,3}, k in {0,1}  
    - Block 3: i in {2,3}, j in {0,1}, k in {0,1}
    - Block 4: i in {2,3}, j in {2,3}, k in {0,1}
    
    That's only 4 blocks covering j in {0,1,2,3} and k in {0,1}, missing j=4 and k in {2,3,4}.
    
    This is getting complicated. Let me use a simpler approach:
    
    Actually, the cleanest way is to note that the tensor T_{i,j,k} = delta for 
    (n,m,p) matrix mult can be decomposed. For 4x5x5, we can use:
    
    Rank = 4*5*5 = 100 (schoolbook)
    
    To reduce to 96, we need to find 4 instances of 2x2x2 where we can save 1 rank each.
    
    Let me try: Partition the computation into blocks where each block involves 
    a 2x2 submatrix of A, a 2x2 submatrix of B, and produces a 2x2 submatrix of C.
    
    For A (4x5): rows 0-3, cols 0-4
    For B (5x5): rows 0-4, cols 0-4
    For C (4x5): rows 0-3, cols 0-4
    
    Let's define 4 blocks:
    Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    Block 3: A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    Block 4: A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]
    
    That's only 4 blocks but they overlap in C[0:2, 0:2] and C[2:4, 0:2].
    
    This is messy. Let me just implement the Strassen-based decomposition directly 
    for the known structure.
    
    Actually, I'll use a more systematic approach: build the full 100-term schoolbook 
    decomposition, then for 4 specific 2x2x2 sub-tensors, replace the 8 terms with 
    7 Strassen terms.
    
    The 4 sub-tensors correspond to:
    (i0,i1) x (j0,j1) x (k0,k1) where:
    - Sub-tensor 1: i in {0,1}, j in {0,1}, k in {0,1}
    - Sub-tensor 2: i in {0,1}, j in {2,3}, k in {2,3}
    - Sub-tensor 3: i in {2,3}, j in {0,1}, k in {2,3}
    - Sub-tensor 4: i in {2,3}, j in {2,3}, k in {0,1}
    
    Wait, these need to be "disjoint" in the sense that they don't share any (i,j,k) 
    triple. Let me check:
    - ST1: i in {0,1}, j in {0,1}, k in {0,1}
    - ST2: i in {0,1}, j in {2,3}, k in {2,3}
    - ST3: i in {2,3}, j in {0,1}, k in {2,3}
    - ST4: i in {2,3}, j in {2,3}, k in {0,1}
    
    These are disjoint! No (i,j,k) appears in more than one.
    
    So total terms = 100 - 4*8 + 4*7 = 100 - 32 + 28 = 96.
    
    Now I need to build this.
    """
    n, m, p = 4, 5, 5  # A is n x m, B is m x p, C is n x p
    
    # Start with schoolbook decomposition
    U = np.zeros((n * m, n * m * p))  # U[i*m+j, c]
    V = np.zeros((m * p, n * m * p))  # V[j*p+k, c]
    W = np.zeros((p * n, n * m * p))  # W[k*n+i, c]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now replace 4 specific 2x2x2 sub-tensors with Strassen
    # Sub-tensor 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Sub-tensor 2: i in {0,1}, j in {2,3}, k in {2,3}
    # Sub-tensor 3: i in {2,3}, j in {0,1}, k in {2,3}
    # Sub-tensor 4: i in {2,3}, j in {2,3}, k in {0,1}
    
    sub_tensors = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [2, 3]),
        ([2, 3], [0, 1], [2, 3]),
        ([2, 3], [2, 3], [0, 1]),
    ]
    
    for (i_vals, j_vals, k_vals) in sub_tensors:
        i0, i1 = i_vals
        j0, j1 = j_vals
        k0, k1 = k_vals
        
        # Find the 8 columns corresponding to these (i,j,k) triples
        cols_to_remove = []
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    # Find column where U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
                    for c in range(n * m * p):
                        if (U[i * m + j, c] == 1 and 
                            V[j * p + k, c] == 1 and 
                            W[k * n + i, c] == 1):
                            cols_to_remove.append(c)
                            break
        
        # Clear these 8 columns
        for c in cols_to_remove:
            U[:, c] = 0
            V[:, c] = 0
            W[:, c] = 0
        
        # Now add 7 Strassen terms in the freed columns
        # We'll use the first 7 of the cleared columns
        strassen_cols = cols_to_remove[:7]
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        c = strassen_cols[0]
        U[i0 * m + j0, c] = 1
        U[i1 * m + j1, c] = 1
        V[j0 * p + k0, c] = 1
        V[j1 * p + k1, c] = 1
        W[k0 * n + i0, c] = 1
        W[k1 * n + i1, c] = 1
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        c = strassen_cols[1]
        U[i1 * m + j0, c] = 1
        U[i1 * m + j1, c] = 1
        V[j0 * p + k0, c] = 1
        W[k0 * n + i1, c] = 1
        
        # M3 = A[i0,j1] * B[j1,k0]
        c = strassen_cols[2]
        U[i0 * m + j1, c] = 1
        V[j1 * p + k0, c] = 1
        W[k0 * n + i0, c] = 1
        
        # M4 = A[i0,j0] * B[j0,k1]
        c = strassen_cols[3]
        U[i0 * m + j0, c] = 1
        V[j0 * p + k1, c] = 1
        W[k1 * n + i0, c] = 1
        
        # M5 = (A[i0,j0] + A[i1,j1]) * B[j1,k1]
        c = strassen_cols[4]
        U[i0 * m + j0, c] = 1
        U[i1 * m + j1, c] = 1
        V[j1 * p + k1, c] = 1
        W[k1 * n + i1, c] = 1
        
        # M6 = (A[i0,j1] - A[i1,j1]) * (B[j0,k1] + B[j1,k1])
        c = strassen_cols[5]
        U[i0 * m + j1, c] = 1
        U[i1 * m + j1, c] = -1
        V[j0 * p + k1, c] = 1
        V[j1 * p + k1, c] = 1
        W[k1 * n + i0, c] = 1
        
        # M7 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j1,k0])
        c = strassen_cols[6]
        U[i1 * m + j0, c] = 1
        U[i0 * m + j0, c] = -1
        V[j0 * p + k0, c] = 1
        V[j1 * p + k0, c] = 1
        W[k0 * n + i1, c] = 1
    
    # Convert to lists as expected
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
