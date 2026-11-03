import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 88.
    # Structure:
    # n=3 split into 1, 2
    # m=4 split into 2, 2
    # p=8 split into 4, 4
    
    # Sub-problems:
    # C11: 1x4 = A11(1x2)B11(2x4) + A12(1x2)B21(2x4) -> Rank 8 + 8 = 16
    # C12: 1x4 = A11(1x2)B12(2x4) + A12(1x2)B22(2x4) -> Rank 8 + 8 = 16
    # C21: 2x4 = A21(2x2)B11(2x4) + A22(2x2)B21(2x4) -> Rank 14 + 14 = 28
    # C22: 2x4 = A21(2x2)B12(2x4) + A22(2x2)B22(2x4) -> Rank 14 + 14 = 28
    # Total Rank = 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook 1x2x4 block
    # A is 1x2, B is 2x4, C is 1x4
    # Indices in U: row i*m + j. Here i is fixed relative to block, j in 0..1
    # Indices in V: row j*p + k. Here j in 0..1, k in 0..3
    # Indices in W: row k*n + i. Here k in 0..3, i fixed
    
    def add_1x2x4(U, V, W, rank_offset, i_start, j_start, k_start, n, m, p):
        # i_start: row index in A (0-based in global n)
        # j_start: col index in A (0-based in global m)
        # k_start: col index in C (0-based in global p)
        # A is 1x2, B is 2x4
        # A indices: (i_start, j_start), (i_start, j_start+1)
        # B indices: (j_start, k_start), (j_start, k_start+1), (j_start+1, k_start), (j_start+1, k_start+1)
        # C indices: (i_start, k_start), (i_start, k_start+1), (i_start, k_start+2), (i_start, k_start+3)
        
        # Rank 8
        # Terms: A[i, j] * B[j, k]
        # j=0, k=0..3
        # j=1, k=0..3
        
        for j in range(2):
            for k in range(4):
                # U index: (i_start, j_start + j)
                u_idx = i_start * m + (j_start + j)
                # V index: (j_start + j, k_start + k)
                v_idx = (j_start + j) * p + (k_start + k)
                # W index: (k_start + k, i_start)
                w_idx = (k_start + k) * n + i_start
                
                U[u_idx][rank_offset] = 1
                V[v_idx][rank_offset] = 1
                W[w_idx][rank_offset] = 1
                rank_offset += 1
        return rank_offset

    # Helper to add a Strassen 2x2x4 block
    # A is 2x2, B is 2x4, C is 2x4
    # Split B into B_left (2x2) and B_right (2x2)
    # A * B_left uses Strassen (Rank 7)
    # A * B_right uses Strassen (Rank 7)
    
    def add_2x2x4_strassen(U, V, W, rank_offset, i_start, j_start, k_start, n, m, p):
        # A indices: (i_start, j_start), (i_start, j_start+1), (i_start+1, j_start), (i_start+1, j_start+1)
        # B indices: 
        #   B_left: (j_start, k_start), (j_start, k_start+1), (j_start+1, k_start), (j_start+1, k_start+1)
        #   B_right: (j_start, k_start+2), (j_start, k_start+3), (j_start+1, k_start+2), (j_start+1, k_start+3)
        # C indices: (i_start, k_start..k_start+3), (i_start+1, k_start..k_start+3)
        
        # Strassen for 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map these to U, V, W.
        # Each M is a product of a linear combo of A and a linear combo of B.
        # This results in a rank-1 tensor for the product?
        # No, Strassen computes the product matrix C.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # The "rank" in tensor decomposition counts the number of scalar multiplications.
        # Each M corresponds to 1 scalar multiplication in the abstract algebra,
        # but in the tensor decomposition of the full matrix product,
        # each M is itself a rank-1 tensor?
        # No. M1 = (A11+A22)(B11+B22).
        # This is a scalar.
        # The contribution to C is M1 * (1, 0, 0, 1) in the C matrix?
        # Actually, the tensor decomposition expresses the bilinear map.
        # The 7 products are the "intermediate" values.
        # The final C is a linear combination of these 7 values.
        # In the U, V, W representation:
        # Each column c in U, V, W corresponds to one rank-1 term.
        # The rank-1 term is U[:, c] * V[:, c] * W[:, c]^T?
        # No, the decomposition is C = sum_c U[:, c] V[:, c]^T W[:, c]?
        # The standard form is:
        # C_{i, k} = sum_{j, c} A_{i, j} U_{j, c} V_{c, k} ... ?
        # The interface is U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # The tensor is T_{i, j, k} = sum_c U_{i, j, c} V_{j, k, c} W_{k, i, c}.
        # Here U_{i, j, c} is U[i*m+j][c].
        
        # For Strassen, the 7 products are:
        # M1 = (A11+A22) * (B11+B22)
        # This is a product of two linear forms.
        # In the tensor decomposition, this corresponds to a column c where:
        # U has non-zeros at (1,1) and (2,2) with value 1.
        # V has non-zeros at (1,1) and (2,2) with value 1.
        # W has non-zeros at the output positions corresponding to M1.
        # M1 contributes to C11 and C22.
        # C11 += M1. C22 += M1.
        # So W[1, 1] = 1, W[2, 2] = 1.
        
        # Let's implement this logic.
        
        def add_strassen_2x2x2(U, V, W, rank_offset, i0, j0, k0):
            # i0, j0, k0 are top-left indices of the 2x2x2 block.
            # A: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
            # B: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
            # C: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
            
            # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
            U[i0*m + j0][rank_offset] += 1
            U[(i0+1)*m + (j0+1)][rank_offset] += 1
            V[j0*p + k0][rank_offset] += 1
            V[(j0+1)*p + (k0+1)][rank_offset] += 1
            W[k0*n + i0][rank_offset] += 1
            W[(k0+1)*n + (i0+1)][rank_offset] += 1
            rank_offset += 1
            
            # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
            U[(i0+1)*m + j0][rank_offset] += 1
            U[(i0+1)*m + (j0+1)][rank_offset] += 1
            V[j0*p + k0][rank_offset] += 1
            W[k0*n + (i0+1)][rank_offset] += 1
            W[(k0+1)*n + (i0+1)][rank_offset] -= 1
            rank_offset += 1
            
            # M3 = A11(B21-B22) -> C12 += M3, C22 += M3
            U[i0*m + j0][rank_offset] += 1
            V[(j0+1)*p + k0][rank_offset] += 1
            V[(j0+1)*p + (k0+1)][rank_offset] -= 1
            W[k0*n + i0][rank_offset] += 1
            W[(k0+1)*n + (i0+1)][rank_offset] += 1
            rank_offset += 1
            
            # M4 = A22(B12-B11) -> C11 += M4, C21 += M4
            U[(i0+1)*m + (j0+1)][rank_offset] += 1
            V[j0*p + (k0+1)][rank_offset] += 1
            V[j0*p + k0][rank_offset] -= 1
            W[k0*n + i0][rank_offset] += 1
            W[k0*n + (i0+1)][rank_offset] += 1
            rank_offset += 1
            
            # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
            U[i0*m + j0][rank_offset] += 1
            U[i0*m + (j0+1)][rank_offset] += 1
            V[(j0+1)*p + (k0+1)][rank_offset] += 1
            W[k0*n + i0][rank_offset] -= 1
            W[(k0+1)*n + i0][rank_offset] += 1
            rank_offset += 1
            
            # M6 = (A21-A11)(B11+B12) -> C22 += M6
            U[(i0+1)*m + j0][rank_offset] += 1
            U[i0*m + j0][rank_offset] -= 1
            V[j0*p + k0][rank_offset] += 1
            V[j0*p + (k0+1)][rank_offset] += 1
            W[(k0+1)*n + (i0+1)][rank_offset] += 1
            rank_offset += 1
            
            # M7 = (A12-A22)(B21+B22) -> C11 += M7, C12 += M7? 
            # Formula: C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            # Wait, standard Strassen:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            # Let's re-verify M7 contribution.
            # M7 = (A12-A22)(B21+B22).
            # It contributes to C11?
            # Yes, C11 += M7.
            # Does it contribute to others?
            # In standard Strassen, M7 is only in C11.
            # Let's check:
            # C11 = A11B11 + A12B21
            # M1 + M4 - M5 + M7 = 
            # (A11+A22)(B11+B22) + A22(B12-B11) - (A11+A12)B22 + (A12-A22)(B21+B22)
            # = A11B11 + A11B22 + A22B11 + A22B22 + A22B12 - A22B11 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
            # = A11B11 + A12B21. Correct.
            
            U[i0*m + (j0+1)][rank_offset] += 1
            U[(i0+1)*m + (j0+1)][rank_offset] -= 1
            V[(j0+1)*p + k0][rank_offset] += 1
            V[(j0+1)*p + (k0+1)][rank_offset] += 1
            W[k0*n + i0][rank_offset] += 1
            rank_offset += 1
            
            return rank_offset

        # Apply Strassen to B_left (k0, k0+1)
        rank_offset = add_strassen_2x2x2(U, V, W, rank_offset, i_start, j_start, k_start)
        # Apply Strassen to B_right (k0+2, k0+3)
        rank_offset = add_strassen_2x2x2(U, V, W, rank_offset, i_start, j_start, k_start + 2)
        
        return rank_offset

    # C11: i=0, j=0..1, k=0..3
    # A11 is (0, 0..1), B11 is (0..1, 0..3)
    col = add_1x2x4(U, V, W, col, 0, 0, 0, n, m, p)
    # A12 is (0, 2..3), B21 is (2..3, 0..3)
    col = add_1x2x4(U, V, W, col, 0, 2, 0, n, m, p)
    
    # C12: i=0, j=0..1, k=4..7
    # A11 is (0, 0..1), B12 is (0..1, 4..7)
    col = add_1x2x4(U, V, W, col, 0, 0, 4, n, m, p)
    # A12 is (0, 2..3), B22 is (2..3, 4..7)
    col = add_1x2x4(U, V, W, col, 0, 2, 4, n, m, p)
    
    # C21: i=1..2, j=0..1, k=0..3
    # A21 is (1..2, 0..1), B11 is (0..1, 0..3)
    col = add_2x2x4_strassen(U, V, W, col, 1, 0, 0, n, m, p)
    # A22 is (1..2, 2..3), B21 is (2..3, 0..3)
    col = add_2x2x4_strassen(U, V, W, col, 1, 2, 0, n, m, p)
    
    # C22: i=1..2, j=0..1, k=4..7
    # A21 is (1..2, 0..1), B12 is (0..1, 4..7)
    col = add_2x2x4_strassen(U, V, W, col, 1, 0, 4, n, m, p)
    # A22 is (1..2, 2..3), B22 is (2..3, 4..7)
    col = add_2x2x4_strassen(U, V, W, col, 1, 2, 4, n, m, p)
    
    if col != rank:
        print(f"Warning: Rank mismatch. Expected {rank}, got {col}")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
