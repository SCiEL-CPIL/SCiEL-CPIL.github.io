"""Matrix multiplication 5x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank exact decomposition for 5x5x5 matrix multiplication.
    
    Strategy: Use block decomposition with Strassen's algorithm for 2x2 blocks
    where possible, and schoolbook for the remainder. This gives rank 117
    (vs schoolbook 125).
    
    We partition the 5x5 matrices into blocks:
    - Split each dimension into 2+2+1
    - Use Strassen (rank 7) for the 2x2x2 sub-blocks
    - Use schoolbook for mixed-size blocks
    
    The total rank is computed by carefully combining these.
    """
    n, m, p = 5, 5, 5
    
    # We'll use a 3x3 block partition where each block is size 2,2,1 or 1,1,1
    # Actually, let's use a cleaner approach:
    # Partition rows into [2,2,1], cols into [2,2,1], etc.
    
    # The 5x5 matrix multiplication can be seen as computing C = A * B
    # where A is 5x5, B is 5x5, C is 5x5.
    
    # Block partition: split each of the 5 indices into 3 parts: sizes [2, 2, 1]
    # So we have a 3x3 block structure for both input and output.
    
    # For each pair of input blocks (i,j) and output block (k), we need to
    # multiply an (a_i x b_j) matrix by a (b_j x c_k) matrix.
    
    # Block sizes:
    # Row blocks of A: [2, 2, 1]
    # Col blocks of A / Row blocks of B: [2, 2, 1]  
    # Col blocks of B: [2, 2, 1]
    
    # So block (i,j) of A has shape (row_size[i] x col_size[j])
    # Block (j,k) of B has shape (col_size[j] x row_size[k])
    # Block (i,k) of C has shape (row_size[i] x row_size[k])
    
    row_sizes = [2, 2, 1]
    
    # For each triple (i, j, k) where i is row block of A, j is col block of A / row block of B,
    # k is col block of B:
    # We need to compute the contribution to C block (i, k) from A block (i, j) * B block (j, k)
    
    # The multiplication of an (a x b) matrix by a (b x c) matrix has rank a*b*c
    # with schoolbook, or can be reduced with Strassen if a=b=c=2 (rank 7 vs 8).
    
    # Let's enumerate all (i, j, k) triples and their block dimensions:
    # A block (i,j): shape (row_sizes[i] x row_sizes[j])
    # B block (j,k): shape (row_sizes[j] x row_sizes[k])
    # C block (i,k): shape (row_sizes[i] x row_sizes[k])
    
    # For each (i,j,k), the sub-problem is multiplying (r_i x r_j) by (r_j x r_k)
    # to get (r_i x r_k). This has rank r_i * r_j * r_k with schoolbook.
    
    # If r_i = r_j = r_k = 2, we can use Strassen with rank 7 instead of 8.
    # If any dimension is 1, schoolbook is optimal (rank = product).
    
    # Let's count:
    # (i,j,k) where all are in {0,1} (size 2): 2*2*2 = 8 triples, each saves 1 rank
    # All other triples use schoolbook.
    
    # Total schoolbook rank: sum over all (i,j,k) of r_i * r_j * r_k
    # = (sum_i r_i) * (sum_j r_j) * (sum_k r_k) = 5 * 5 * 5 = 125
    
    # With Strassen on the 8 pure-2x2x2 blocks: save 8 ranks -> 125 - 8 = 117
    
    # Now let's construct the actual U, V, W matrices.
    
    # We'll build up the decomposition term by term.
    # Each term corresponds to a rank-1 component: U_col * V_col * W_col
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec is length 25, v_vec is length 25, w_vec is length 25."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def schoolbook_term(i, j, k, a, b, c):
        """
        Add schoolbook decomposition for multiplying (a x b) by (b x c).
        i, j, k are block indices; a, b, c are their sizes.
        """
        # Offset within the 5x5 matrix
        row_off_i = sum(row_sizes[:i])
        col_off_j = sum(row_sizes[:j])
        col_off_k = sum(row_sizes[:k])
        
        for ii in range(a):
            for jj in range(b):
                for kk in range(c):
                    u = [0.0] * 25
                    v = [0.0] * 25
                    w = [0.0] * 25
                    
                    # A element at (row_off_i + ii, col_off_j + jj)
                    u[(row_off_i + ii) * 5 + (col_off_j + jj)] = 1.0
                    # B element at (col_off_j + jj, col_off_k + kk)
                    v[(col_off_j + jj) * 5 + (col_off_k + kk)] = 1.0
                    # C element at (row_off_i + ii, col_off_k + kk)
                    w[(col_off_k + kk) * 5 + (row_off_i + ii)] = 1.0
                    # Wait, W convention is W[k*n+i, r] where the output is C[i,k]
                    # Let me re-check: C[i,k] = sum_j A[i,j] * B[j,k]
                    # W index: k*n + i where n=5, so W[k*5 + i]
                    
                    w[kk * 5 + ii] = 1.0  # This is wrong, let me fix
                    
                    # Actually W[k*n+i, r] means the row index in W is k*n + i
                    # But k here is the column index of B / column index of C
                    # and i is the row index of A / row index of C
                    
                    # Let me redo: for C[row, col], W index is col * n + row = col * 5 + row
                    # row = row_off_i + ii, col = col_off_k + kk
                    w[col_off_k + kk] * 5 + (row_off_i + ii)  # This is wrong syntax
                    
                    # Let me just set it properly:
                    w_idx = (col_off_k + kk) * 5 + (row_off_i + ii)
                    w = [0.0] * 25
                    w[w_idx] = 1.0
                    
                    add_term(u, v, w)
    
    def strassen_term(i, j, k):
        """
        Add Strassen decomposition for 2x2x2 block multiplication.
        i, j, k are block indices, each with size 2.
        """
        row_off_i = sum(row_sizes[:i])
        col_off_j = sum(row_sizes[:j])
        col_off_k = sum(row_sizes[:k])
        
        # Strassen's 7 products for 2x2 matrix multiplication:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each of the 7 products, we need to express them as linear combinations
        # of the input elements and output to C elements.
        
        # Let's define the 7 Strassen products explicitly.
        # Each product is of the form (linear combo of A elements) * (linear combo of B elements)
        # and contributes to C elements.
        
        # I'll define each term as:
        # A_factors: list of (a_row, a_col, coeff) for the A part
        # B_factors: list of (b_row, b_col, coeff) for the B part
        # C_factors: list of (c_row, c_col, coeff) for the C part
        
        strassen_products = [
            # (A_factors, B_factors, C_factors)
            # M1 = (A11 + A22)(B11 + B22) -> C11 += 1, C22 += 1
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2 = (A21 + A22)B11 -> C21 += 1, C22 += -1
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            # M3 = A11(B12 - B22) -> C12 += 1, C22 += 1
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            # M4 = A22(B21 - B11) -> C11 += 1, C21 += 1
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            # M5 = (A11 + A12)B22 -> C11 += -1, C12 += 1
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            # M6 = (A21 - A11)(B11 + B12) -> C22 += 1
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            # M7 = (A12 - A22)(B21 + B22) -> C11 += 1, C12 += -1
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(0,1,-1)]),
        ]
        
        for a_factors, b_factors, c_factors in strassen_products:
            u = [0.0] * 25
            v = [0.0] * 25
            w = [0.0] * 25
            
            for (ar, ac, coeff) in a_factors:
                u[(row_off_i + ar) * 5 + (col_off_j + ac)] += coeff
            for (br, bc, coeff) in b_factors:
                v[(col_off_j + br) * 5 + (col_off_k + bc)] += coeff
            for (cr, cc, coeff) in c_factors:
                w_idx = (col_off_k + cc) * 5 + (row_off_i + cr)
                w[w_idx] += coeff
            
            add_term(u, v, w)
    
    # Now iterate over all (i, j, k) triples
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = row_sizes[i]
                b = row_sizes[j]
                c = row_sizes[k]
                
                if a == 2 and b == 2 and c == 2:
                    strassen_term(i, j, k)
                else:
                    schoolbook_term(i, j, k, a, b, c)
    
    rank = len(U_cols)
    
    # Convert to 2D lists
    U = [list(map(float, col)) for col in U_cols]
    V = [list(map(float, col)) for col in V_cols]
    W = [list(map(float, col)) for col in W_cols]
    
    # Transpose: U should be (n*m, rank) = (25, rank)
    # Currently U_cols is a list of rank vectors, each of length 25
    # So U is already (25, rank) if we treat each col as a column
    # But the format expects U to be (25, rank), so U[row][col]
    # U_cols[c][r] gives U[r][c], which is what we want.
    
    # Actually, let's be explicit:
    U = [[U_cols[c][r] for c in range(rank)] for r in range(25)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(25)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(25)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
