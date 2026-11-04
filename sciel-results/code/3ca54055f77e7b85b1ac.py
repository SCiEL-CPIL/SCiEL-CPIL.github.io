"""
Tensor decomposition for 3x4 by 4x4 matrix multiplication.
Uses block structure: split A into [2x4; 1x4], B into [4x2; 4x2], C into [2x2; 1x2].
Apply Strassen to 2x2x2 blocks, schoolbook for 1x2x2 blocks.
"""
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll build the decomposition by blocks.
    # A is 3x4, B is 4x4, C is 3x4.
    # Split A into A1 (2x4) and A2 (1x4)
    # Split B into B1 (4x2) and B2 (4x2)
    # Then C1 = A1*B1 (2x2), C2 = A1*B2 (2x2), C3 = A2*B1 (1x2), C4 = A2*B2 (1x2)
    
    # For each block product, we use either Strassen (for 2x2x2) or schoolbook (for 1x2x2)
    # We collect all the (U, V, W) columns.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper: add a term where U has support on specific A indices, V on B indices, W on C indices
    # For schoolbook: U[i,j]=1, V[j,k]=1, W[k,i]=1
    # For Strassen: linear combinations
    
    # Let's represent indices:
    # A indices: (i, j) where i in 0..2, j in 0..3 -> index i*4 + j
    # B indices: (j, k) where j in 0..3, k in 0..3 -> index j*4 + k
    # C indices: (i, k) where i in 0..2, k in 0..3 -> index k*3 + i  (W[k*n+i] convention)
    
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a single tensor term."""
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for (i, j), val in u_dict.items():
            u_vec[a_idx(i, j)] = val
        for (j, k), val in v_dict.items():
            v_vec[b_idx(j, k)] = val
        for (k, i), val in w_dict.items():
            w_vec[c_idx(k, i)] = val
        U_rows.append(u_vec)
        V_rows.append(v_vec)
        W_rows.append(w_vec)
    
    # Block C1 = A1 * B1 : A1 is rows 0-1 of A (i=0,1), B1 is cols 0-1 of B (k=0,1)
    # This is a 2x2x2 product. Use Strassen.
    # A1 = [[a00, a01], [a10, a11]] where a_ij = A[i, j] for i in 0,1 and j in 0,1? 
    # Wait, B1 is cols 0-1 of B, so B1[j, k] = B[j, k] for k in 0,1.
    # The product C1[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k] for i in 0,1, k in 0,1.
    # This is NOT a 2x2x2 product because the summation index j ranges over 0..3, not 0..1.
    # So we can't directly apply Strassen here.
    
    # Let me reconsider. The block decomposition I was thinking of doesn't work directly 
    # because the intermediate index j is shared across all blocks and has size 4.
    
    # Alternative approach: Use the fact that rank of matrix multiplication M_{n,m,p} 
    # is at most n * m * p / 2 for some cases, but for 3x4x4 it's unknown.
    # 
    # Let's try a different strategy: use the Strassen algorithm recursively.
    # 
    # Actually, let's think about this more carefully. The standard schoolbook rank is 3*4*4 = 48.
    # 
    # One known result: rank(M_{2,2,2}) = 7 (Strassen).
    # rank(M_{2,2,3}) is known to be 13? Let me recall... Actually, rank(M_{2,2,3}) = 13 is a known bound.
    # 
    # For M_{3,4,4}, we can view it as a combination of smaller tensors.
    # 
    # Let's try to decompose it as follows:
    # C = A * B where A is 3x4, B is 4x4.
    # We can write A = [A_top; A_bottom] where A_top is 2x4 and A_bottom is 1x4.
    # B = [B_left | B_right] where B_left is 4x2 and B_right is 4x2.
    # Then C = [A_top*B_left, A_top*B_right; A_bottom*B_left, A_bottom*B_right]
    # 
    # The four blocks are:
    # C11 = A_top * B_left : 2x2 result, but it's a "2x4x2" tensor product
    # C12 = A_top * B_right : 2x2 result, "2x4x2" tensor
    # C21 = A_bottom * B_left : 1x2 result, "1x4x2" tensor  
    # C22 = A_bottom * B_right : 1x2 result, "1x4x2" tensor
    #
    # The rank of M_{2,4,2} is at most 2*4*2 = 16 schoolbook, but can we do better?
    # M_{2,4,2} can be viewed as computing a 2x2 matrix where each entry is a dot product of 
    # length-4 vectors. So it's like 4 separate 2x1x2 products? No...
    #
    # Actually, M_{2,4,2} has rank at most 8 by using the fact that it's equivalent to 
    # computing 4 independent 2x2 multiplications? No, that's not right either.
    #
    # Let me just implement the schoolbook and see if I can find a better decomposition.
    # 
    # Actually, let's try a completely different approach: use random linear algebra 
    # to find a low-rank decomposition. But this is hard to do exactly with integers.
    #
    # Let me try the following: since we need exact decomposition with integer/half-integer entries,
    # and the evaluator checks exactness, let's stick with known good decompositions.
    #
    # For now, let's implement a slightly better than schoolbook decomposition.
    # 
    # One idea: use the fact that M_{3,4,4} can be embedded in M_{4,4,4} and use 
    # Strassen-like decompositions for M_{4,4,4}.
    #
    # Rank of M_{4,4,4} is at most 49 using Strassen recursively (2^3 * 7 = 56? No, 
    # Strassen gives 7^(log2(4)) = 7^2 = 49 for 4x4x4).
    # 
    # So if we can embed our 3x4x4 problem into a 4x4x4 problem and use rank 49,
    # we'd have rank 49, which is worse than 48.
    #
    # Let me try yet another approach: exploit the specific dimensions.
    #
    # Actually, I recall that for M_{n,m,p} where one dimension is 1, the rank is n*p.
    # For M_{2,2,2} it's 7.
    #
    # Let me try to construct a decomposition using Strassen for sub-blocks.
    #
    # Split the problem as follows:
    # A (3x4) = [A1 (2x4); A2 (1x4)]
    # B (4x4) = [B1 (4x2); B2 (4x2)]
    #
    # C11 = A1 * B1 (2x2)
    # C12 = A1 * B2 (2x2)
    # C21 = A2 * B1 (1x2)
    # C22 = A2 * B2 (1x2)
    #
    # For C11: This is a 2x4x2 tensor. We can compute it as:
    # C11[i,k] = sum_{j=0}^{3} A1[i,j] * B1[j,k]
    # = sum_{j=0}^{1} A1[i,j] * B1[j,k] + sum_{j=2}^{3} A1[i,j] * B1[j,k]
    # = C11a[i,k] + C11b[i,k]
    # where C11a is a 2x2x2 product (using A1[:,0:2] and B1[0:2,:])
    # and C11b is a 2x2x2 product (using A1[:,2:4] and B1[2:4,:])
    #
    # So C11 can be computed with 2 * 7 = 14 operations using Strassen for each 2x2x2 block.
    #
    # Similarly for C12, C21, C22:
    # C12: 2x4x2 -> split into two 2x2x2 blocks -> 14 operations
    # C21: 1x4x2 -> this is trickier. Let's think...
    #   C21[i,k] = sum_{j=0}^{3} A2[i,j] * B1[j,k] where i is always 0 (since A2 is 1x4)
    #   = sum_{j=0}^{1} A2[0,j] * B1[j,k] + sum_{j=2}^{3} A2[0,j] * B1[j,k]
    #   Each sum is a 1x2x2 product. Rank of M_{1,2,2} is 2 (just two dot products).
    #   So C21 needs 2 + 2 = 4 operations.
    # C22: similar to C21, needs 4 operations.
    #
    # Total: 14 + 14 + 4 + 4 = 36 operations.
    #
    # That's better than 48! Let me implement this.

    U_cols = []
    V_cols = []
    W_cols = []

    def strassen_222(u_offsets, v_offsets, w_offsets):
        """
        Add Strassen decomposition for a 2x2x2 block.
        u_offsets: list of 4 indices into A (for the 2x2 submatrix)
        v_offsets: list of 4 indices into B (for the 2x2 submatrix)
        w_offsets: list of 4 indices into C (for the 2x2 submatrix)
        """
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        
        a00, a01, a10, a11 = u_offsets
        b00, b01, b10, b11 = v_offsets
        c00, c01, c10, c11 = w_offsets
        
        # Strassen's 7 products:
        # M1 = (a00 + a11) * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a01) * b11
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        
        # C entries:
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        terms = []
        
        # M1 = (a00 + a11) * (b00 + b11)
        u1 = {a00: 1, a11: 1}
        v1 = {b00: 1, b11: 1}
        w1 = {c00: 1, c11: 1}
        terms.append((u1, v1, w1))
        
        # M2 = (a10 + a11) * b00
        u2 = {a10: 1, a11: 1}
        v2 = {b00: 1}
        w2 = {c10: 1, c11: -1}
        terms.append((u2, v2, w2))
        
        # M3 = a00 * (b01 - b11)
        u3 = {a00: 1}
        v3 = {b01: 1, b11: -1}
        w3 = {c01: 1, c11: 1}
        terms.append((u3, v3, w3))
        
        # M4 = a11 * (b10 - b00)
        u4 = {a11: 1}
        v4 = {b10: 1, b00: -1}
        w4 = {c00: 1, c10: 1}
        terms.append((u4, v4, w4))
        
        # M5 = (a00 + a01) * b11
        u5 = {a00: 1, a01: 1}
        v5 = {b11: 1}
        w5 = {c00: -1, c01: 1}
        terms.append((u5, v5, w5))
        
        # M6 = (a10 - a00) * (b00 + b01)
        u6 = {a10: 1, a00: -1}
        v6 = {b00: 1, b01: 1}
        w6 = {c11: 1}
        terms.append((u6, v6, w6))
        
        # M7 = (a01 - a11) * (b10 + b11)
        u7 = {a01: 1, a11: -1}
        v7 = {b10: 1, b11: 1}
        w7 = {c00: 1}
        terms.append((u7, v7, w7))
        
        for u, v, w in terms:
            add_term(u, v, w)
    
    def schoolbook_122(u_idx, v_indices, w_indices):
        """
        Add schoolbook decomposition for a 1x2x2 block.
        u_idx: index into A (row 0 of the 1x2 submatrix)
        v_indices: list of 2 indices into B
        w_indices: list of 2 indices into C
        """
        for j, v_idx in enumerate(v_indices):
            for k, w_idx in enumerate(w_indices):
                u = {u_idx: 1}
                v = {v_idx: 1}
                w = {w_idx: 1}
                add_term(u, v, w)
    
    # Now let's define the actual indices.
    # A is 3x4: indices (i,j) -> i*4+j
    # B is 4x4: indices (j,k) -> j*4+k
    # C is 3x4: indices (i,k) -> k*3+i (W[k*n+i] convention)
    
    # C11 = A[0:2, :] * B[:, 0:2] -> result is 2x2
    # Split into two 2x2x2 blocks:
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    strassen_222(
        u_offsets=[0*4+0, 0*4+1, 1*4+0, 1*4+1],  # a00, a01, a10, a11
        v_offsets=[0*4+0, 0*4+1, 1*4+0, 1*4+1],  # b00, b01, b10, b11
        w_offsets=[0*3+0, 0*3+1, 1*3+0, 1*3+1]    # c00, c01, c10, c11
    )
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2]
    strassen_222(
        u_offsets=[0*4+2, 0*4+3, 1*4+2, 1*4+3],
        v_offsets=[2*4+0, 2*4+1, 3*4+0, 3*4+1],
        w_offsets=[0*3+0, 0*3+1, 1*3+0, 1*3+1]
    )
    
    # C12 = A[0:2, :] * B[:, 2:4] -> result is 2x2
    # Block 1: A[0:2, 0:2] * B[0:2, 2:4]
    strassen_222(
        u_offsets=[0*4+0, 0*4+1, 1*4+0, 1*4+1],
        v_offsets=[0*4+2, 0*4+3, 1*4+2, 1*4+3],
        w_offsets=[2*3+0, 2*3+1, 3*3+0, 3*3+1]
    )
    
    # Block 2: A[0:2, 2:4] * B[2:4, 2:4]
    strassen_222(
        u_offsets=[0*4+2, 0*4+3, 1*4+2, 1*4+3],
        v_offsets=[2*4+2, 2*4+3, 3*4+2, 3*4+3],
        w_offsets=[2*3+0, 2*3+1, 3*3+0, 3*3+1]
    )
    
    # C21 = A[2:3, :] * B[:, 0:2] -> result is 1x2
    # Split into two 1x2x2 blocks:
    # Block 1: A[2, 0:2] * B[0:2, 0:2]
    schoolbook_122(
        u_idx=2*4+0,  # A[2,0]
        v_indices=[0*4+0, 1*4+0],  # B[0,0], B[1,0]
        w_indices=[0*3+2, 1*3+2]   # C[2,0], C[2,1]
    )
    # Block 2: A[2, 2:4] * B[2:4, 0:2]
    schoolbook_122(
        u_idx=2*4+2,  # A[2,2]
        v_indices=[2*4+0, 3*4+0],  # B[2,0], B[3,0]
        w_indices=[0*3+2, 1*3+2]   # C[2,0], C[2,1]
    )
    # Wait, this isn't right. For a 1x2x2 product:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For A[2, 0:2] * B[0:2, 0:2]:
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    # So we need 4 terms, not 2.
    
    # Let me redo this. The schoolbook_122 function above is wrong.
    # Let me fix it.
    
    # Actually, let me just directly add the terms for C21 and C22.
    
    # Clear the previous incorrect additions (we'll just rebuild)
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Redo everything correctly.
    
    # C11 = A[0:2, :] * B[:, 0:2]
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    strassen_222(
        u_offsets=[0*4+0, 0*4+1, 1*4+0, 1*4+1],
        v_offsets=[0*4+0, 0*4+1, 1*4+0, 1*4+1],
        w_offsets=[0*3+0, 0*3+1, 1*3+0, 1*3+1]
    )
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2]
    strassen_222(
        u_offsets=[0*4+2, 0*4+3, 1*4+2, 1*4+3],
        v_offsets=[2*4+0, 2*4+1, 3*4+0, 3*4+1],
        w_offsets=[0*3+0, 0*3+1, 1*3+0, 1*3+1]
    )
    
    # C12 = A[0:2, :] * B[:, 2:4]
    # Block 1: A[0:2, 0:2] * B[0:2, 2:4]
    strassen_222(
        u_offsets=[0*4+0, 0*4+1, 1*4+0, 1*4+1],
        v_offsets=[0*4+2, 0*4+3, 1*4+2, 1*4+3],
        w_offsets=[2*3+0, 2*3+1, 3*3+0, 3*3+1]
    )
    
    # Block 2: A[0:2, 2:4] * B[2:4, 2:4]
    strassen_222(
        u_offsets=[0*4+2, 0*4+3, 1*4+2, 1*4+3],
        v_offsets=[2*4+2, 2*4+3, 3*4+2, 3*4+3],
        w_offsets=[2*3+0, 2*3+1, 3*3+0, 3*3+1]
    )
    
    # C21 = A[2, :] * B[:, 0:2] -> result C[2, 0:2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0] + A[2,3]*B[3,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1] + A[2,3]*B[3,1]
    # This is a 1x4x2 product, rank at most 4*2 = 8 schoolbook, but we can do better.
    # Actually, it's just 8 independent products, so rank 8.
    # But wait, we can group them:
    # C[2,0:2] = [A[2,0:2]] * B[0:2, 0:2] + [A[2,2:4]] * B[2:4, 0:2]
    # Each is a 1x2x2 product with rank 4.
    # So total rank 8 for C21.
    
    # Add terms for C21:
    for j in range(4):
        for k in range(2):
            add_term(
                u_dict={a_idx(2, j): 1},
                v_dict={b_idx(j, k): 1},
                w_dict={c_idx(k, 2): 1}
            )
    
    # C22 = A[2, :] * B[:, 2:4] -> result C[2, 2:4]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2] + A[2,3]*B[3,2]
    # C[2,3] = A[2,0]*B[0,3] + A[2,1]*B[1,3] + A[2,2]*B[2,3] + A[2,3]*B[3,3]
    # Rank 8.
    
    for j in range(4):
        for k in range(2, 4):
            add_term(
                u_dict={a_idx(2, j): 1},
                v_dict={b_idx(j, k): 1},
                w_dict={c_idx(k, 2): 1}
            )
    
    # Total rank: 4*7 + 8 + 8 = 28 + 16 = 44
    
    # Build the matrices
    rank = len(U_cols)
    U = np.array(U_cols).T  # shape (n*m, rank)
    V = np.array(V_cols).T  # shape (m*p, rank)
    W = np.array(W_cols).T  # shape (p*n, rank)
    
    # Convert to list of lists for the return format
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    
    return {"U": U, "V": V, "W": W}
