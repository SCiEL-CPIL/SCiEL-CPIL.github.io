"""Matrix multiplication 4x5x5 - Strassen-based decomposition."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 120.0
    
    # Partition:
    # A is 4x5, B is 5x5
    # Split A rows into two 2x5 blocks
    # Split B cols into 2x3, 2x3, 1x3 blocks
    # Split A cols into 2x3, 2x3, 1x3 blocks
    # Split B rows into 2x3, 2x3, 1x3 blocks
    
    # For pairs where A_col_block and B_row_block are both size 2,
    # we can use Strassen to compute 2x2 products.
    
    # A_col_blocks: [{0,1}, {2,3}, {4}] - indices into 5 columns
    # B_row_blocks: [{0,1}, {2,3}, {4}] - indices into 5 rows
    # Matching size-2 pairs: ({0,1},{0,1}) and ({2,3},{2,3})
    
    # A_row_blocks: [{0,1}, {2,3}] - indices into 4 rows
    # B_col_blocks: [{0,1}, {2,3}, {4}] - indices into 5 cols
    
    # For each (A_row_block, B_col_block) pair where both are size 2,
    # and for each matching (A_col_block, B_row_block) pair where both are size 2,
    # apply Strassen.
    
    # A_row_blocks of size 2: 2 blocks
    # B_col_blocks of size 2: 2 blocks ({0,1}, {2,3})
    # Matching A_col/B_row size-2 pairs: 2 pairs
    
    # Total Strassen applications: 2 * 2 * 2 = 8
    # Each Strassen uses 7 terms
    # Total from Strassen: 8 * 7 = 56
    
    # Remaining terms use schoolbook:
    # Cases where A_col_block and B_row_block don't both have size 2:
    # (size2, size1) pairs: 2*1 = 2
    # (size1, size2) pairs: 1*2 = 2  
    # (size1, size1) pairs: 1*1 = 1
    # Total mismatched A_col/B_row pairs: 5
    # Each has 2*2*1 = 4 schoolbook terms (2 A_row x 2 B_col x 1 middle)
    # But wait, need to be more careful.
    
    # Let me think differently. Total output entries: 4*5 = 20
    # Each output entry C[i][k] = sum_j A[i][j]*B[j][k]
    
    # Let me just implement the Strassen-based approach carefully.
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Build the full decomposition using Strassen where applicable.
    
    rank = 0
    U = {}
    V = {}
    W = {}
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a single rank-1 term. u_entries, v_entries, w_entries are dicts mapping index->coeff."""
        nonlocal rank
        col = rank
        rank += 1
        U[col] = dict(u_entries)
        V[col] = dict(v_entries)
        W[col] = dict(w_entries)
    
    # Strassen 2x2 product: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # where A rows are given by a_rows, B cols by b_cols, A cols by a_cols, B rows by b_rows
    def strassen_2x2(a_rows, b_cols, a_cols, b_rows):
        """
        a_rows: list of 2 row indices of A (and C)
        b_cols: list of 2 col indices of B (and C)  
        a_cols: list of 2 col indices of A (and row indices of B)
        b_rows: list of 2 row indices of B (and col indices of A for middle)
        
        Computes C[a_rows[0:2], b_cols[0:2]] = sum over a_cols/b_rows
        """
        # Strassen terms
        # M1 = (A00+A11)(B00+B11)
        u0 = {a_rows[0]*m+a_cols[0]: 1, a_rows[1]*m+a_cols[1]: 1}
        v0 = {b_rows[0]*p+b_cols[0]: 1, b_rows[1]*p+b_cols[1]: 1}
        w0 = {b_cols[0]*n+a_rows[0]: 1, b_cols[1]*n+a_rows[0]: 1, 
              b_cols[0]*n+a_rows[1]: 1, b_cols[1]*n+a_rows[1]: 1}
        add_term(u0, v0, w0)
        
        # M2 = (A10+A11)B00
        u1 = {a_rows[1]*m+a_cols[0]: 1, a_rows[1]*m+a_cols[1]: 1}
        v1 = {b_rows[0]*p+b_cols[0]: 1}
        w1 = {b_cols[0]*n+a_rows[1]: 1, b_cols[1]*n+a_rows[1]: 1}
        add_term(u1, v1, w1)
        
        # M3 = A00(B10-B11)
        u2 = {a_rows[0]*m+a_cols[0]: 1}
        v2 = {b_rows[1]*p+b_cols[0]: 1, b_rows[1]*p+b_cols[1]: -1}
        w2 = {b_cols[0]*n+a_rows[0]: 1, b_cols[1]*n+a_rows[1]: 1}
        add_term(u2, v2, w2)
        
        # M4 = A11(B01-B00)
        u3 = {a_rows[1]*m+a_cols[1]: 1}
        v3 = {b_rows[0]*p+b_cols[1]: 1, b_rows[0]*p+b_cols[0]: -1}
        w3 = {b_cols[0]*n+a_rows[0]: -1, b_cols[1]*n+a_rows[1]: 1}
        add_term(u3, v3, w3)
        
        # M5 = (A00+A01)B11
        u4 = {a_rows[0]*m+a_cols[0]: 1, a_rows[0]*m+a_cols[1]: 1}
        v4 = {b_rows[1]*p+b_cols[1]: 1}
        w4 = {b_cols[1]*n+a_rows[0]: -1, b_cols[1]*n+a_rows[1]: 1}
        add_term(u4, v4, w4)
        
        # M6 = (A10-A00)(B00+B01)
        u5 = {a_rows[1]*m+a_cols[0]: 1, a_rows[0]*m+a_cols[0]: -1}
        v5 = {b_rows[0]*p+b_cols[0]: 1, b_rows[0]*p+b_cols[1]: 1}
        w5 = {b_cols[0]*n+a_rows[0]: 1, b_cols[1]*n+a_rows[1]: 1}
        add_term(u5, v5, w5)
        
        # M7 = (A01-A11)(B10+B11)
        u6 = {a_rows[0]*m+a_cols[1]: 1, a_rows[1]*m+a_cols[1]: -1}
        v6 = {b_rows[1]*p+b_cols[0]: 1, b_rows[1]*p+b_cols[1]: 1}
        w6 = {b_cols[0]*n+a_rows[1]: -1, b_cols[1]*n+a_rows[1]: 1}
        add_term(u6, v6, w6)
    
    # Schoolbook for remaining terms
    def schoolbook_single(a_row, b_col, mid):
        """Single schoolbook term for C[a_row][b_col] += A[a_row][mid]*B[mid][b_col]"""
        u = {a_row*m+mid: 1}
        v = {mid*p+b_col: 1}
        w = {b_col*n+a_row: 1}
        add_term(u, v, w)
    
    # A_col partitions: [{0,1}, {2,3}, {4}]
    # B_row partitions: [{0,1}, {2,3}, {4}]
    # A_row partitions: [{0,1}, {2,3}]
    # B_col partitions: [{0,1}, {2,3}, {4}]
    
    a_col_parts = [[0,1], [2,3], [4]]
    b_row_parts = [[0,1], [2,3], [4]]
    a_row_parts = [[0,1], [2,3]]
    b_col_parts = [[0,1], [2,3], [4]]
    
    for ac_idx, ac in enumerate(a_col_parts):
        for br_idx, br in enumerate(b_row_parts):
            for ar_idx, ar in enumerate(a_row_parts):
                for bc_idx, bc in enumerate(b_col_parts):
                    if len(ac) == 2 and len(br) == 2:
                        # Use Strassen
                        strassen_2x2(ar, bc, ac, br)
                    else:
                        # Schoolbook for each combination
                        for a_row in ar:
                            for b_col in bc:
                                for mid in ac:  # ac == br in size, use ac
                                    if mid in br:
                                        schoolbook_single(a_row, b_col, mid)
    
    # Convert to final format
    nU, nV, nW = n*m, m*p, p*n
    U_mat = [[0]*rank for _ in range(nU)]
    V_mat = [[0]*rank for _ in range(nV)]
    W_mat = [[0]*rank for _ in range(nW)]
    
    for col in range(rank):
        for idx, val in U[col].items():
            U_mat[idx][col] = val
        for idx, val in V[col].items():
            V_mat[idx][col] = val
        for idx, val in W[col].items():
            W_mat[idx][col] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
