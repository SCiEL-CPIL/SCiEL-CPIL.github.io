import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We decompose 2x4x5 into two 2x2x5 products.
    # Each 2x2x5 is decomposed into 2x2x2 (Strassen, rank 7) and 2x2x1 (schoolbook, rank 4) three times.
    # Total rank per 2x2x5 is 7 + 4 + 4 + 4 = 19.
    # Total rank is 2 * 19 = 38.
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Indices in U: i*m + j. Here m=4.
    # We are handling sub-blocks.
    # Let's map the global indices.
    # A is 2x4. Rows 0,1. Cols 0,1,2,3.
    # B is 4x5. Rows 0,1,2,3. Cols 0,1,2,3,4.
    # C is 2x5. Rows 0,1. Cols 0,1,2,3,4.
    
    # Split A into A1 (cols 0,1) and A2 (cols 2,3).
    # Split B into B1 (rows 0,1) and B2 (rows 2,3).
    # C = A1 B1 + A2 B2.
    
    # For A1 B1:
    # A1 is 2x2 (cols 0,1 of A).
    # B1 is 2x5 (rows 0,1 of B).
    # Split B1 into B1_1 (cols 0,1) and B1_2 (cols 2,3,4).
    # C1 = A1 B1_1 + A1 B1_2.
    
    # A1 B1_1 is 2x2x2. Use Strassen.
    # A1 B1_2 is 2x2x3. Use 3x (2x2x1).
    
    # Indices for A1: U[0, :], U[1, :] (rows 0,1 of A). Cols 0,1 of A are 0,1.
    # Indices for B1: V[0, :], V[1, :] (rows 0,1 of B).
    # Indices for C1: W[0, :], W[1, :] (cols 0,1 of C). W index is k*n+i.
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        # a_row_offset, a_col_offset: top-left of 2x2 A block
        # b_row_offset, b_col_offset: top-left of 2x2 B block
        # c_row_offset, c_col_offset: top-left of 2x2 C block
        # Returns next column index
        # U indices: (a_row_offset + i) * m + (a_col_offset + j)
        # V indices: (b_row_offset + i) * p + (b_col_offset + j)
        # W indices: (c_col_offset + k) * n + (c_row_offset + i)
        
        def u_idx(i, j):
            return (a_row_offset + i) * m + (a_col_offset + j)
        def v_idx(i, j):
            return (b_row_offset + i) * p + (b_col_offset + j)
        def w_idx(k, i):
            return (c_col_offset + k) * n + (c_row_offset + i)
            
        # M1 = (x11+x22)(y11+y22)
        U[u_idx(0,0)][start_col] = 1
        U[u_idx(1,1)][start_col] = 1
        V[v_idx(0,0)][start_col] = 1
        V[v_idx(1,1)][start_col] = 1
        W[w_idx(0,0)][start_col] = 1
        W[w_idx(1,1)][start_col] = 1
        # z11 += M1, z22 += M1
        # Handled by W entries? No, W entries are for the final sum.
        # In this interface, U, V, W are such that sum_c U[i,c] V[j,c] W[k,c] = C[i,k]
        # Wait, the standard decomposition is C[i,k] = sum_c U[i,c] V[k,c]?
        # No, the task says:
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # C[i,k] = sum_c U[i*m + j, c] V[j*p + l, c] W[l*n + i, c]?
        # Let's check the seed code.
        # U[i*m+j][c], V[j*p+k][c], W[k*n+i][c].
        # Sum over j, k, c?
        # The tensor is T[i,j,k]. C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
        # Decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]?
        # No, usually it's U[i,c] V[j,c] W[k,c] for T[i,j,k].
        # But here dimensions are mixed.
        # Seed code:
        # U[i*m+j][c] = 1
        # V[j*p+k][c] = 1
        # W[k*n+i][c] = 1
        # This implies the term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        # Summing over c, j, k?
        # No, the sum is over c. The indices i,j,k are free?
        # If we sum over c, we get a tensor T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
        # Then C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k]?
        # No, the decomposition is for the multiplication tensor itself.
        # The multiplication tensor T has entries T[i,j,k] = 1 if we multiply A[i,j] and B[j,k].
        # The rank is the number of terms c.
        # So we need to represent T[i,j,k] as sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
        # Wait, W index is k*n+i.
        # In the seed, W[k*n+i] is used.
        # So for a specific c, we have a triple (i,j,k) such that U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
        # This corresponds to a rank-1 tensor.
        
        # So for Strassen, we need to express the 2x2x2 multiplication tensor as sum of 7 rank-1 tensors.
        # The 7 terms are:
        # 1. (x11+x22)(y11+y22) -> contributes to z11, z22.
        # This means we need to add 1 to U[0,0], V[0,0], W[0,0]?
        # No, the bilinear form is (x11+x22)(y11+y22).
        # This is not a rank-1 tensor in (i,j,k).
        # The tensor T[i,j,k] is 1 if (i,j,k) is a valid index.
        # Strassen's algorithm computes z using 7 products.
        # Each product is of the form (linear combo of x)(linear combo of y).
        # This corresponds to a rank-1 tensor in the space of bilinear forms?
        # Yes.
        # So we need to map the 7 Strassen products to the U, V, W matrices.
        
        # Product 1: (x11+x22)(y11+y22)
        # x11 is A[0,0], x22 is A[1,1].
        # y11 is B[0,0], y22 is B[1,1].
        # This product contributes to z11 and z22.
        # z11 = ... + M1 ...
        # z22 = ... + M1 ...
        # In the tensor decomposition, we add a term for each "multiplication" in the algorithm.
        # But the algorithm's products are not directly the tensor entries.
        # The tensor entries are the coefficients of A[i,j]B[j,k] in the final sum.
        # Strassen's algorithm is a change of basis.
        # It is known that the rank of the tensor is 7.
        # The decomposition is:
        # T = sum_{c=1}^7 u_c \otimes v_c \otimes w_c.
        # We need to find u_c, v_c, w_c.
        
        # For Strassen 2x2x2:
        # u_1: A[0,0]=1, A[1,1]=1. Others 0.
        # v_1: B[0,0]=1, B[1,1]=1. Others 0.
        # w_1: C[0,0]=1, C[1,1]=1. Others 0.
        # This works for M1.
        
        # u_2: A[1,0]=1, A[1,1]=1.
        # v_2: B[0,0]=1.
        # w_2: C[1,0]=1.
        # Check: A[1,0]B[0,0] + A[1,1]B[0,0] = (x21+x22)y11.
        # Contributes to z21? z21 = x21y11 + x22y21.
        # M2 = (x21+x22)y11.
        # z21 = M2 + ...
        # So w_2 should have C[1,0]=1.
        
        # This mapping is standard. I will implement the 7 terms for 2x2x2.
        
        # Term 1
        U[u_idx(0,0)][start_col] += 1
        U[u_idx(1,1)][start_col] += 1
        V[v_idx(0,0)][start_col] += 1
        V[v_idx(1,1)][start_col] += 1
        W[w_idx(0,0)][start_col] += 1
        W[w_idx(1,1)][start_col] += 1
        start_col += 1
        
        # Term 2
        U[u_idx(1,0)][start_col] += 1
        U[u_idx(1,1)][start_col] += 1
        V[v_idx(0,0)][start_col] += 1
        W[w_idx(1,0)][start_col] += 1
        start_col += 1
        
        # Term 3
        U[u_idx(0,0)][start_col] += 1
        V[v_idx(0,1)][start_col] += 1
        V[v_idx(1,1)][start_col] -= 1
        W[w_idx(0,1)][start_col] += 1
        start_col += 1
        
        # Term 4
        U[u_idx(1,1)][start_col] += 1
        V[v_idx(1,0)][start_col] += 1
        V[v_idx(0,0)][start_col] -= 1
        W[w_idx(1,0)][start_col] += 1
        start_col += 1
        
        # Term 5
        U[u_idx(0,0)][start_col] += 1
        U[u_idx(0,1)][start_col] += 1
        V[v_idx(1,1)][start_col] += 1
        W[w_idx(0,1)][start_col] += 1
        start_col += 1
        
        # Term 6
        U[u_idx(1,0)][start_col] += 1
        U[u_idx(0,0)][start_col] -= 1
        V[v_idx(0,0)][start_col] += 1
        V[v_idx(0,1)][start_col] += 1
        W[w_idx(1,1)][start_col] += 1
        start_col += 1
        
        # Term 7
        U[u_idx(0,1)][start_col] += 1
        U[u_idx(1,1)][start_col] -= 1
        V[v_idx(1,0)][start_col] += 1
        V[v_idx(1,1)][start_col] += 1
        W[w_idx(1,1)][start_col] += 1
        start_col += 1
        
        return start_col

    def add_schoolbook_2x2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        # 2x2 by 2x1.
        # z1 = x11y11 + x12y21
        # z2 = x21y11 + x22y21
        # 4 terms.
        # T1: x11 y11 -> z1
        U[u_idx(0,0)][start_col] += 1
        V[v_idx(0,0)][start_col] += 1
        W[w_idx(0,0)][start_col] += 1
        start_col += 1
        
        # T2: x12 y21 -> z1
        U[u_idx(0,1)][start_col] += 1
        V[v_idx(1,0)][start_col] += 1
        W[w_idx(0,0)][start_col] += 1
        start_col += 1
        
        # T3: x21 y11 -> z2
        U[u_idx(1,0)][start_col] += 1
        V[v_idx(0,0)][start_col] += 1
        W[w_idx(1,0)][start_col] += 1
        start_col += 1
        
        # T4: x22 y21 -> z2
        U[u_idx(1,1)][start_col] += 1
        V[v_idx(1,0)][start_col] += 1
        W[w_idx(1,0)][start_col] += 1
        start_col += 1
        
        return start_col

    # Block 1: A1 (rows 0,1, cols 0,1) * B1 (rows 0,1, cols 0,4)
    # B1 is 2x5.
    # Split B1 into B1_1 (cols 0,1) and B1_2 (cols 2,3,4).
    # A1 B1_1: 2x2x2. Strassen.
    col = add_strassen(0, 0, 0, 0, 0, 0, col)
    
    # A1 B1_2: 2x2x3.
    # Split into 3 x 2x2x1.
    # Col 2:
    col = add_schoolbook_2x2x1(0, 0, 0, 2, 0, 2, col)
    # Col 3:
    col = add_schoolbook_2x2x1(0, 0, 0, 3, 0, 3, col)
    # Col 4:
    col = add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4, col)
    
    # Block 2: A2 (rows 0,1, cols 2,3) * B2 (rows 2,3, cols 0,4)
    # B2 is 2x5.
    # Split B2 into B2_1 (cols 0,1) and B2_2 (cols 2,3,4).
    # A2 B2_1: 2x2x2. Strassen.
    col = add_strassen(0, 2, 2, 0, 0, 0, col)
    
    # A2 B2_2: 2x2x3.
    # Col 2:
    col = add_schoolbook_2x2x1(0, 2, 2, 2, 0, 2, col)
    # Col 3:
    col = add_schoolbook_2x2x1(0, 2, 2, 3, 0, 3, col)
    # Col 4:
    col = add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4, col)
    
    return {"U": U, "V": V, "W": W}
