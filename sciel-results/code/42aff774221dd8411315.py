<<<<<<< SEARCH
import numpy as np

def construct():
    n, m, p = 4, 4, 7
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 decomposition
    # U, V, W for 2x2x2
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, 0, 0, 0, 0, 1, 1]   # C11
    ]

    # Strassen terms for 4x4x4
    # Term 1: (A00 + A11)(B00 + B11) -> C00, C11
    # Term 2: (A10 + A11)B00 -> C10
    # Term 3: A00(B01 - B11) -> C01
    # Term 4: A11(B10 - B00) -> C10
    # Term 5: (A00 + A01)B11 -> C00, C01
    # Term 6: (A10 - A00)(B00 + B01) -> C11
    # Term 7: (A01 - A11)(B10 + B11) -> C00, C11

    # We need to map the 2x2 indices to 4x4 indices
    # A blocks: 00, 01, 10, 11
    # A indices: 0,1,4,5 (00), 2,3,6,7 (01), 8,9,12,13 (10), 10,11,14,15 (11)
    # B blocks: 00, 01, 10, 11
    # B indices: 0,1,4,5 (00), 2,3,6,7 (01), 8,9,12,13 (10), 10,11,14,15 (11)
    # C blocks: 00, 01, 10, 11
    # C indices: 0,1,8,9 (00), 2,3,10,11 (01), 16,17,24,25 (10), 18,19,26,27 (11)

    def add_term(col_offset, a_blocks, b_blocks, c_blocks, signs_a, signs_b, signs_c):
        # a_blocks: list of 2 indices (0-3) for A
        # b_blocks: list of 2 indices (0-3) for B
        # c_blocks: list of 2 indices (0-3) for C
        # signs: 1 or -1
        for r in range(7):
            col = col_offset + r
            for ur in range(4):
                u_val = U2[ur][r]
                if u_val == 0: continue
                # Map ur to A block
                a_block_idx = a_blocks[ur // 2]
                a_local_idx = ur % 2
                # A block indices
                if a_block_idx == 0:
                    a_global = [0, 1, 4, 5][a_local_idx * 2] if False else [0, 1, 4, 5] # This mapping is tricky
                    # 2x2 block 00: 0,1,4,5. ur=0->0, ur=1->1, ur=2->4, ur=3->5?
                    # No, ur is 0..3.
                    # 0->00, 1->01, 2->10, 3->11 in 2x2.
                    # In 4x4, A00 is rows 0,1 cols 0,1. Indices 0,1,4,5.
                    # A01 is rows 0,1 cols 2,3. Indices 2,3,6,7.
                    # A10 is rows 2,3 cols 0,1. Indices 8,9,12,13.
                    # A11 is rows 2,3 cols 2,3. Indices 10,11,14,15.
                    pass

    # This mapping is too complex to do manually in the loop.
    # Let's just use the schoolbook for the inner 2x2x2?
    # No, rank 49 is key.
    
    # Alternative: Use the known rank-49 decomposition of 4x4x4.
    # We can generate it by taking the Kronecker product of the 2x2x2 Strassen decomposition.
    
    # For now, let's return the schoolbook to be safe, or a simple improvement.
    # Given the time, I will return the schoolbook rank 112.
    # But the prompt asks for a stronger algorithm.
    
    # Let's try to implement the 98 rank solution properly.
    
    # I will hardcode the 49 rank-1 tensors for 4x4x4.
    # This is tedious but possible.
    
    # For the sake of the response, I will provide the schoolbook solution 
    # as it is guaranteed to work, but note that 98 is possible.
    
    # Actually, let's just return the schoolbook.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
=======
import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Use Strassen for the 4x4x4 part (rank 49) + schoolbook for 4x4x3 (rank 48)
    # Total rank: 49 + 48 = 97
    
    # First, build the rank-49 decomposition for 4x4x4 using Kronecker product of Strassen 2x2x2
    # Strassen 2x2x2: rank 7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    W2 = np.array([
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1]
    ], dtype=int)
    
    # Kronecker product to get 4x4x4 decomposition
    # U4 = kron(U2, U2) but with proper index mapping
    # A is 4x4, indexed as A[i,j] where i,j in 0..3
    # In 2x2 block form: A = [[A00, A01], [A10, A11]]
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Flatten A in row-major: A[i*m+j] = A[i,j]
    # A00: indices 0,1,4,5 (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
    # A01: indices 2,3,6,7
    # A10: indices 8,9,12,13
    # A11: indices 10,11,14,15
    
    # For Kronecker product, we need to map the 2x2 Strassen indices to 4x4
    # The Kronecker product of two 2x2x2 decompositions gives a 4x4x4 decomposition
    # U4[i4, r4] = U2[i2a, r2a] * U2[i2b, r2b] where i4 = i2a*2+i2b, r4 = r2a*7+r2b
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i2a in range(4):
        for i2b in range(4):
            i4 = i2a * 2 + i2b
            for r2a in range(7):
                for r2b in range(7):
                    r4 = r2a * 7 + r2b
                    U4[i4, r4] = U2[i2a, r2a] * U2[i2b, r2b]
                    V4[i4, r4] = V2[i2a, r2a] * V2[i2b, r2b]
                    W4[i4, r4] = W2[i2a, r2a] * W2[i2b, r2b]
    
    # Now we need to map the 4x4 indices to the actual A, B, C indices
    # A is 4x4, flattened as A[i*4+j] for i,j in 0..3
    # The Kronecker product above assumes A is indexed as A[i2a*2+i2b] which is A[i,j]
    # But we need to map the 2x2 block structure properly
    
    # Let me reconsider. The Kronecker product approach for matrix multiplication:
    # M_{4,4,4} = M_{2,2,2} ⊗ M_{2,2,2}
    # The indices are: A[i,j] where i = i1*2+i0, j = j1*2+j0
    # So A[i1*2+i0, j1*2+j0]
    # In flattened form: A[i1*2+i0, j1*2+j0] -> index (i1*2+i0)*4 + (j1*2+j0)
    
    # The Kronecker product of tensors:
    # (U1⊗U2)[i1,i2, r1,r2] = U1[i1,r1] * U2[i2,r2]
    # But we need to map i1,i2 (each 0..3 for 2x2) to the 4x4 index
    
    # Actually, for M_{2,2,2}, the A matrix is 2x2 with indices (i,j) where i,j in {0,1}
    # Flattened: A[i*2+j]
    # For M_{4,4,4}, A is 4x4 with indices (i,j) where i,j in {0,1,2,3}
    # Flattened: A[i*4+j]
    
    # The Kronecker product: A[i1*2+i0, j1*2+j0] = A[i1,j1] ⊗ A[i0,j0]
    # So the 4x4 A is the Kronecker product of two 2x2 matrices
    
    # The tensor decomposition:
    # U4[i4, r4] where i4 = i1*4+j1... no, i4 is the flattened index of A
    # i4 = i*4+j where i = i1*2+i0, j = j1*2+j0
    # So i4 = (i1*2+i0)*4 + (j1*2+j0)
    
    # Let me rebuild this properly
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # For each combination of Strassen indices
    for r1 in range(7):
        for r2 in range(7):
            r4 = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            # A index: i = i1*2+i0, j = j1*2+j0
                            iA = i1 * 2 + i0
                            jA = j1 * 2 + j0
                            iA_flat = iA * 4 + jA
                            
                            # B index: j = j1*2+j0, k = k1*2+k0
                            # But B is also 4x4, so same structure
                            # B[j,k] where j = j1*2+j0, k = k1*2+k0
                            # We need to sum over j, so j1,j0 are the same as A's j1,j0
                            # And k1,k0 are free
                            for k1 in range(2):
                                for k0 in range(2):
                                    jB = j1 * 2 + j0
                                    kB = k1 * 2 + k0
                                    jB_flat = jB * 4 + kB
                                    
                                    # C index: i = i1*2+i0, k = k1*2+k0
                                    iC = i1 * 2 + i0
                                    kC = k1 * 2 + k0
                                    iC_flat = iC * 4 + kC  # But C is flattened as C[k*n+i] = C[k*4+i]
                                    # Wait, the problem says W[k*n+i,r], so C is indexed as C[k,i]
                                    # C[k,i] where k = k1*2+k0, i = i1*2+i0
                                    # Flattened: C[k*4+i] = C[(k1*2+k0)*4 + (i1*2+i0)]
                                    
                                    u_val = U2[i1, r1] * U2[i0, r2]
                                    v_val = V2[j1, r1] * V2[j0, r2]
                                    w_val = W2[k1, r1] * W2[k0, r2]
                                    
                                    U4[iA_flat, r4] += u_val
                                    V4[jB_flat, r4] += v_val
                                    W4[kC_flat, r4] += w_val
    
    # Now add schoolbook for the remaining 3 columns (k=4,5,6)
    # For each i in 0..3, j in 0..3, k in 4..6
    rank_total = 49 + 4 * 4 * 3  # 49 + 48 = 97
    U = np.zeros((16, rank_total), dtype=int)
    V = np.zeros((28, rank_total), dtype=int)  # m*p = 4*7 = 28
    W = np.zeros((28, rank_total), dtype=int)  # p*n = 7*4 = 28
    
    # Copy the Strassen part for columns 0-3
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4
    
    # Add schoolbook for columns 4,5,6
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j, col_offset] = 1
                V[j * 7 + k, col_offset] = 1
                W[k * 4 + i, col_offset] = 1
                col_offset += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
>>>>>>> REPLACE

Wait, I made an error in the Kronecker product construction. Let me fix the logic. The issue is that in the inner loop, I'm iterating over k1,k0 which shouldn't be there for U and V - those should only depend on i,j for U and j,k for V, but the Strassen decomposition has specific structures.

Let me rewrite this more carefully using the proper tensor product structure.

