"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx is (i, j)
        # v_idx is (j, k)
        # w_idx is (k, i)
        # We need to map to linear indices
        # U: i*m + j
        # V: j*p + k
        # W: k*n + i
        pass # Will be implemented inside construct or as closure

    # We will build the matrices row by row, or just append columns.
    # Since rank is variable, we build lists of columns.
    # U_col, V_col, W_col
    
    def add_rank1(i, j, j_prime, k, k_prime, i_prime, val=1):
        # The term is U[i,j] * V[j',k] * W[k',i']
        # For matrix mult, we need j==j' and k==k' and i==i'?
        # No, the tensor decomposition is:
        # T = sum u_r v_r w_r
        # u_r is a matrix in R^{n x m}
        # v_r is a matrix in R^{m x p}
        # w_r is a matrix in R^{p x n} (indexed by k, i)
        # The product A B is recovered by:
        # C_{ik} = sum_r (sum_{j} U_{i,j,r} A_{ij}) (sum_{j} V_{j,k,r} B_{jk}) W_{k,i,r} ?
        # Wait, if W is just a scalar per term, then W_{k,i,r} must be 1 only for the matching i,k.
        # In the seed code, W[k*n+i] = 1 for the specific i,k.
        # So each term in the sum corresponds to a specific (i,j) for U, (j,k) for V, and (k,i) for W.
        # And the value is 1.
        # So we just need to add a column to U, V, W with 1s at the appropriate positions.
        
        u_lin = i * m + j
        v_lin = j * p + k
        w_lin = k * n + i
        
        # Append to U, V, W
        # We are building the matrices column by column.
        # So we need to keep track of the current rank.
        pass

    # Let's use a list of tuples (u_lin, v_lin, w_lin)
    terms = []
    
    def add_term(i, j, k, val=1):
        # Adds a term corresponding to A[i,j] * B[j,k] contributing to C[i,k]
        # In the tensor decomposition, this is a rank-1 tensor with 1 at (i,j), (j,k), (k,i)
        terms.append((i * m + j, j * p + k, k * n + i, val))

    # We need to add linear combinations for Strassen.
    # Strassen M1 = (A11 + A22)(B11 + B22)
    # This expands to A11 B11 + A11 B22 + A22 B11 + A22 B22
    # But we want to compute M1 as a single multiplication of sums.
    # This means we need a term where U has 1s at A11 and A22 positions?
    # No, U is a matrix.
    # If we want to compute (A11 + A22) * (B11 + B22), we need:
    # U has 1 at (0,0) and 1 at (1,1).
    # V has 1 at (0,0) and 1 at (1,1).
    # W has 1 at (0,0) and 1 at (1,1)?
    # No, the result is a matrix C.
    # C11 = M1 + ...
    # C12 = M3 + ...
    # So W must distribute the result to the correct positions.
    
    # Let's define a function to add a Strassen block for 2x2x2.
    # Inputs:
    # i0, i1: row indices for A (2 rows)
    # j0, j1: col indices for A (2 cols)
    # k0, k1: col indices for B (2 cols)
    # The B indices are j0, j1.
    # The C indices are i0, i1 and k0, k1.
    
    def add_strassen_2x2(i0, i1, j0, j1, k0, k1):
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # U: 1 at (i0,j0), 1 at (i1,j1)
        # V: 1 at (j0,k0), 1 at (j1,k1)
        # W: 1 at (k0,i0) with +1, 1 at (k1,i0) with +1?
        # No, M1 contributes to C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1].
        # Specifically:
        # C[i0,k0] += M1
        # C[i0,k1] += M1
        # C[i1,k0] += M1
        # C[i1,k1] += M1
        # Wait, M1 is a scalar.
        # C[i0,k0] = M1 + M4 - M5 + M7
        # C[i0,k1] = M3 + M5
        # C[i1,k0] = M2 + M4
        # C[i1,k1] = M1 - M2 + M3 + M6
        
        # So for M1, W should have:
        # W[k0, i0] += 1
        # W[k1, i0] += 1  (Wait, M1 is added to C[i0,k1] too? No, M1 is added to C[i0,k0] and C[i1,k1] and C[i0,k1]?
        # Let's check Strassen formulas.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1 is in C11 and C22.
        # So W for M1 should have 1 at (k0, i0) and 1 at (k1, i1).
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        # U: 1 at (i1,j0), 1 at (i1,j1)
        # V: 1 at (j0,k0)
        # W: 1 at (k0, i1)
        
        # M3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
        # U: 1 at (i0,j0)
        # V: 1 at (j0,k1), -1 at (j1,k1)
        # W: 1 at (k1, i0)
        
        # M4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
        # U: 1 at (i1,j1)
        # V: 1 at (j1,k0), -1 at (j0,k0)
        # W: 1 at (k0, i0)
        
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        # U: 1 at (i0,j0), 1 at (i0,j1)
        # V: 1 at (j1,k1)
        # W: 1 at (k1, i1)
        
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        # U: 1 at (i1,j0), -1 at (i0,j0)
        # V: 1 at (j0,k0), 1 at (j0,k1)
        # W: 1 at (k1, i1)
        
        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        # U: 1 at (i0,j1), -1 at (i1,j1)
        # V: 1 at (j1,k0), 1 at (j1,k1)
        # W: 1 at (k0, i0)
        
        # We need to add these as columns.
        # Each M is a separate rank-1 term in the decomposition of the sub-block.
        # So we add 7 terms.
        
        # M1
        add_term_comb([(i0,j0, 1), (i1,j1, 1)], [(j0,k0, 1), (j1,k1, 1)], [(k0,i0, 1), (k1,i1, 1)])
        
        # M2
        add_term_comb([(i1,j0, 1), (i1,j1, 1)], [(j0,k0, 1)], [(k0,i1, 1)])
        
        # M3
        add_term_comb([(i0,j0, 1)], [(j0,k1, 1), (j1,k1, -1)], [(k1,i0, 1)])
        
        # M4
        add_term_comb([(i1,j1, 1)], [(j1,k0, 1), (j0,k0, -1)], [(k0,i0, 1)])
        
        # M5
        add_term_comb([(i0,j0, 1), (i0,j1, 1)], [(j1,k1, 1)], [(k1,i1, 1)])
        
        # M6
        add_term_comb([(i1,j0, 1), (i0,j0, -1)], [(j0,k0, 1), (j0,k1, 1)], [(k1,i1, 1)])
        
        # M7
        add_term_comb([(i0,j1, 1), (i1,j1, -1)], [(j1,k0, 1), (j1,k1, 1)], [(k0,i0, 1)])

    def add_term_comb(u_list, v_list, w_list):
        # u_list: list of (i, j, val)
        # v_list: list of (j, k, val)
        # w_list: list of (k, i, val)
        # This adds one column to U, V, W.
        # U column has vals at i*m+j
        # V column has vals at j*p+k
        # W column has vals at k*n+i
        
        u_col = [0.0] * (n*m)
        v_col = [0.0] * (m*p)
        w_col = [0.0] * (p*n)
        
        for i, j, val in u_list:
            u_col[i*m + j] = val
        for j, k, val in v_list:
            v_col[j*p + k] = val
        for k, i, val in w_list:
            w_col[k*n + i] = val
            
        terms.append((u_col, v_col, w_col))

    # Now build the full decomposition
    
    # Block 1: 4x2x6 (Top part of A, Top part of B)
    # A rows 0-3, cols 0-1
    # B rows 0-1, cols 0-5
    # C rows 0-3, cols 0-5
    
    # Split C into 3 blocks of 2 cols
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        # A is 4x2, B is 2x2
        # Split A into two 2x2
        # A1: rows 0-1
        # A2: rows 2-3
        add_strassen_2x2(0, 1, 0, 1, k0, k1)
        add_strassen_2x2(2, 3, 0, 1, k0, k1)
        
    # Block 2: 4x3x6 (Bottom part of A, Bottom part of B)
    # A rows 0-3, cols 2-4
    # B rows 2-4, cols 0-5
    # C rows 0-3, cols 0-5
    
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        # A is 4x3, B is 3x2
        # Split A into two 2x3
        # A1: rows 0-1
        # A2: rows 2-3
        # For each 2x3x2, we use 3 terms of 2x1x2 (rank 4 each)
        
        # A1 (rows 0-1, cols 2-4)
        # B_part (rows 2-4, cols k0-k1)
        
        # Term 1: A[0..1, 2] * B[2, k0..k1]
        # U: 1 at (0,2), 1 at (1,2)
        # V: 1 at (2,k0), 1 at (2,k1)
        # W: 1 at (k0,0), 1 at (k1,0), 1 at (k0,1), 1 at (k1,1)
        # Wait, 2x1x2 rank 4.
        # C[i,k] = A[i,2] * B[2,k]
        # This is 4 terms:
        # A[0,2] B[2,k0] -> C[0,k0]
        # A[0,2] B[2,k1] -> C[0,k1]
        # A[1,2] B[2,k0] -> C[1,k0]
        # A[1,2] B[2,k1] -> C[1,k1]
        
        add_term_comb([(0,2, 1)], [(2,k0, 1)], [(k0,0, 1)])
        add_term_comb([(0,2, 1)], [(2,k1, 1)], [(k1,0, 1)])
        add_term_comb([(1,2, 1)], [(2,k0, 1)], [(k0,1, 1)])
        add_term_comb([(1,2, 1)], [(2,k1, 1)], [(k1,1, 1)])
        
        # Term 2: A[0..1, 3] * B[3, k0..k1]
        add_term_comb([(0,3, 1)], [(3,k0, 1)], [(k0,0, 1)])
        add_term_comb([(0,3, 1)], [(3,k1, 1)], [(k1,0, 1)])
        add_term_comb([(1,3, 1)], [(3,k0, 1)], [(k0,1, 1)])
        add_term_comb([(1,3, 1)], [(3,k1, 1)], [(k1,1, 1)])
        
        # Term 3: A[0..1, 4] * B[4, k0..k1]
        add_term_comb([(0,4, 1)], [(4,k0, 1)], [(k0,0, 1)])
        add_term_comb([(0,4, 1)], [(4,k1, 1)], [(k1,0, 1)])
        add_term_comb([(1,4, 1)], [(4,k0, 1)], [(k0,1, 1)])
        add_term_comb([(1,4, 1)], [(4,k1, 1)], [(k1,1, 1)])
        
        # A2 (rows 2-3, cols 2-4)
        add_term_comb([(2,2, 1)], [(2,k0, 1)], [(k0,2, 1)])
        add_term_comb([(2,2, 1)], [(2,k1, 1)], [(k1,2, 1)])
        add_term_comb([(3,2, 1)], [(2,k0, 1)], [(k0,3, 1)])
        add_term_comb([(3,2, 1)], [(2,k1, 1)], [(k1,3, 1)])
        
        add_term_comb([(2,3, 1)], [(3,k0, 1)], [(k0,2, 1)])
        add_term_comb([(2,3, 1)], [(3,k1, 1)], [(k1,2, 1)])
        add_term_comb([(3,3, 1)], [(3,k0, 1)], [(k0,3, 1)])
        add_term_comb([(3,3, 1)], [(3,k1, 1)], [(k1,3, 1)])
        
        add_term_comb([(2,4, 1)], [(4,k0, 1)], [(k0,2, 1)])
        add_term_comb([(2,4, 1)], [(4,k1, 1)], [(k1,2, 1)])
        add_term_comb([(3,4, 1)], [(4,k0, 1)], [(k0,3, 1)])
        add_term_comb([(3,4, 1)], [(4,k1, 1)], [(k1,3, 1)])

    # Construct matrices
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_col, v_col, w_col = terms[r]
        for i in range(n * m):
            U[i][r] = u_col[i]
        for j in range(m * p):
            V[j][r] = v_col[j]
        for k in range(p * n):
            W[k][r] = w_col[k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
