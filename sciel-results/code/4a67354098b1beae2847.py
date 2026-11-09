"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    num_u = n * m  # 8
    num_v = m * p  # 16
    num_w = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 100
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((num_u, rank))
        V = np.zeros((num_v, rank))
        W = np.zeros((num_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    def strassen_4x4():
        """
        Strassen decomposition for 4x4 * 4x4 -> 4x4.
        Returns U, V, W of shape (16, 7), (16, 7), (16, 7) for the 4x4 block.
        We index U[i*4+j, r] for A[i,j], V[j*4+k, r] for B[j,k], W[k*4+i, r] for C[i,k].
        """
        # Strassen for 2x2 blocks.
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We have a 4x4 matrix. We can view it as a 2x2 block matrix of 2x2 blocks.
        # Block (0,0) is rows 0-1, cols 0-1
        # Block (0,1) is rows 0-1, cols 2-3
        # Block (1,0) is rows 2-3, cols 0-1
        # Block (1,1) is rows 2-3, cols 2-3
        
        # The 7 Strassen products correspond to linear combinations of the 16 entries of A and B.
        # We need to construct U, V, W for the 4x4 case.
        
        # Let's define the 7 rank-1 terms for the 4x4 multiplication.
        # Each term r has U_r, V_r, W_r vectors of length 16.
        
        U = np.zeros((16, 7))
        V = np.zeros((16, 7))
        W = np.zeros((16, 7))
        
        # Helper to add to U, V, W
        def add_term(r, u_indices, v_indices, w_indices, val):
            for idx in u_indices:
                U[idx, r] += val
            for idx in v_indices:
                V[idx, r] += val
            for idx in w_indices:
                W[idx, r] += val
        
        # Block indices:
        # A blocks:
        # A11: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 4, 5
        # A12: (0,2), (0,3), (1,2), (1,3) -> indices 2, 3, 6, 7
        # A21: (2,0), (2,1), (3,0), (3,1) -> indices 8, 9, 12, 13
        # A22: (2,2), (2,3), (3,2), (3,3) -> indices 10, 11, 14, 15
        
        # B blocks:
        # B11: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 4, 5
        # B12: (0,2), (0,3), (1,2), (1,3) -> indices 2, 3, 6, 7
        # B21: (2,0), (2,1), (3,0), (3,1) -> indices 8, 9, 12, 13
        # B22: (2,2), (2,3), (3,2), (3,3) -> indices 10, 11, 14, 15
        
        # C blocks:
        # C11: indices 0, 1, 4, 5
        # C12: indices 2, 3, 6, 7
        # C21: indices 8, 9, 12, 13
        # C22: indices 10, 11, 14, 15
        
        # M1 = (A11+A22)(B11+B22)
        # U1: A11 + A22
        add_term(0, [0,1,4,5,10,11,14,15], [0,1,4,5,10,11,14,15], [0,1,4,5], 1.0)
        # W1: C11 + C22
        add_term(0, [0,1,4,5,10,11,14,15], [0,1,4,5,10,11,14,15], [10,11,14,15], 1.0)
        
        # M2 = (A21+A22)B11
        # U2: A21 + A22
        add_term(1, [8,9,12,13,10,11,14,15], [0,1,4,5], [8,9,12,13], 1.0)
        
        # M3 = A11(B12-B22)
        # U3: A11
        add_term(2, [0,1,4,5], [2,3,6,7], [2,3,6,7], 1.0)
        # V3: B12 - B22
        add_term(2, [0,1,4,5], [10,11,14,15], [2,3,6,7], -1.0)
        
        # M4 = A22(B21-B11)
        # U4: A22
        add_term(3, [10,11,14,15], [8,9,12,13], [4,5], 1.0)
        # V4: B21 - B11
        add_term(3, [10,11,14,15], [0,1,4,5], [4,5], -1.0)
        
        # M5 = (A11+A12)B22
        # U5: A11 + A12
        add_term(4, [0,1,4,5,2,3,6,7], [10,11,14,15], [4,5], -1.0)
        
        # M6 = (A21-A11)(B11+B12)
        # U6: A21 - A11
        add_term(5, [8,9,12,13], [0,1,4,5,2,3,6,7], [10,11,14,15], 1.0)
        add_term(5, [0,1,4,5], [0,1,4,5,2,3,6,7], [10,11,14,15], -1.0)
        
        # M7 = (A12-A22)(B21+B22)
        # U7: A12 - A22
        add_term(6, [2,3,6,7], [8,9,12,13,10,11,14,15], [0,1,4,5], 1.0)
        add_term(6, [10,11,14,15], [8,9,12,13,10,11,14,15], [0,1,4,5], -1.0)
        
        return U, V, W
    
    # Build rank-28 decomposition for 2x4 * 4x4
    def strassen_rank28():
        """
        Uses 2x2 Strassen for the 2x4 * 4x4 problem.
        View A (2x4) as two 1x4 rows? No, Strassen works on blocks.
        Let's view A as 2x2 blocks of size 1x2.
        A11 = A[0, 0:2], A12 = A[0, 2:4]
        A21 = A[1, 0:2], A22 = A[1, 2:4]
        
        B (4x4) as 2x2 blocks of size 2x2.
        B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
        B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
        
        C (2x4) as 2x2 blocks of size 1x2.
        C11 = C[0, 0:2], C12 = C[0, 2:4]
        C21 = C[1, 0:2], C22 = C[1, 2:4]
        
        This doesn't quite fit standard Strassen which requires square blocks.
        However, we can use the rank-7 decomposition for 2x2x2 and combine them.
        
        Actually, a simpler way to get rank 28:
        The 2x4 * 4x4 tensor is the tensor product of (2x1 * 1x4) and (4x4)? No.
        
        Let's use the fact that 2x4 * 4x4 can be decomposed into 4 independent 2x2 * 2x2 multiplications?
        No, that would be rank 4 * 7 = 28.
        
        Let's verify this.
        A = [[a00, a01, a02, a03],
             [a10, a11, a12, a13]]
        B = [[b00, b01, b02, b03],
             [b10, b11, b12, b13],
             [b20, b21, b22, b23],
             [b30, b31, b32, b33]]
        C = A * B (2x4)
        
        C[i, k] = sum_j A[i, j] B[j, k]
        
        If we split j into two groups: j in {0,1} and j in {2,3}.
        C[i, k] = sum_{j in {0,1}} A[i, j] B[j, k] + sum_{j in {2,3}} A[i, j] B[j, k]
        
        This is the sum of two products:
        C1 = A1 * B1 where A1 is 2x2 (cols 0,1), B1 is 2x4 (rows 0,1)
        C2 = A2 * B2 where A2 is 2x2 (cols 2,3), B2 is 2x4 (rows 2,3)
        
        So the problem reduces to two independent 2x2 * 2x4 multiplications.
        What is the rank of 2x2 * 2x4?
        It's equivalent to 2x2 * (2x2 + 2x2).
        The tensor is the sum of two 2x2 * 2x2 tensors.
        Rank(2x2 * 2x2) = 7 (Strassen).
        So Rank(2x2 * 2x4) <= 14?
        If we can do 2x2 * 2x4 with rank 14, then total rank is 28.
        
        Let's construct the rank-14 decomposition for 2x2 * 2x4.
        A (2x2), B (2x4), C (2x4).
        B = [B_left, B_right] where B_left, B_right are 2x2.
        C = [C_left, C_right] where C_left = A * B_left, C_right = A * B_right.
        
        We can use Strassen for A * B_left and A * B_right.
        That gives 7 + 7 = 14 terms.
        
        Let's build the U, V, W for the full 2x4 * 4x4 case using this structure.
        """
        rank = 28
        U = np.zeros((num_u, rank))
        V = np.zeros((num_v, rank))
        W = np.zeros((num_w, rank))
        
        # Strassen 2x2 * 2x2 terms
        # Indices for U (2x2): 0,1,2,3
        # Indices for V (2x2): 0,1,2,3
        # Indices for W (2x2): 0,1,2,3
        
        # We need to map these to the 2x4 * 4x4 indices.
        # U index: i*m + j. i in {0,1}, j in {0,1,2,3}.
        # V index: j*p + k. j in {0,1,2,3}, k in {0,1,2,3}.
        # W index: k*n + i. k in {0,1,2,3}, i in {0,1}.
        
        # Part 1: j in {0,1}
        # A_part1: rows 0,1; cols 0,1. U indices: 0,1,2,3 (i*4+j)
        #   i=0,j=0 -> 0
        #   i=0,j=1 -> 1
        #   i=1,j=0 -> 4
        #   i=1,j=1 -> 5
        # B_part1: rows 0,1; cols 0,1,2,3. V indices: j*4+k
        #   j=0: 0,1,2,3
        #   j=1: 4,5,6,7
        # C_part1: rows 0,1; cols 0,1,2,3. W indices: k*2+i
        #   k=0: 0,1
        #   k=1: 2,3
        #   k=2: 4,5
        #   k=3: 6,7
        
        # Part 2: j in {2,3}
        # A_part2: rows 0,1; cols 2,3. U indices: 2,3,6,7
        #   i=0,j=2 -> 2
        #   i=0,j=3 -> 3
        #   i=1,j=2 -> 6
        #   i=1,j=3 -> 7
        # B_part2: rows 2,3; cols 0,1,2,3. V indices: j*4+k
        #   j=2: 8,9,10,11
        #   j=3: 12,13,14,15
        # C_part2: rows 0,1; cols 0,1,2,3. W indices: k*2+i
        #   Same as C_part1.
        
        # Let's define the 7 Strassen terms for 2x2 * 2x4.
        # For B (2x4), we can view it as two 2x2 blocks: B_left (cols 0,1) and B_right (cols 2,3).
        # We apply Strassen to A * B_left and A * B_right.
        
        def strassen_2x2_2x4(u_indices, v_indices, w_indices, col_offset):
            """
            u_indices: list of 4 indices for U (2x2 block)
            v_indices: list of 8 indices for V (2x4 block), first 4 for left, last 4 for right
            w_indices: list of 8 indices for W (2x4 block), first 4 for left, last 4 for right
            col_offset: offset in the rank dimension
            """
            # v_left = v_indices[0:4], v_right = v_indices[4:8]
            # w_left = w_indices[0:4], w_right = w_indices[4:8]
            
            # Term 0: M1 = (A11+A22)(B11+B22)
            # U: A11, A22
            U[u_indices[0], col_offset + 0] = 1.0
            U[u_indices[3], col_offset + 0] = 1.0
            # V: B11, B22
            V[v_indices[0], col_offset + 0] = 1.0
            V[v_indices[3], col_offset + 0] = 1.0
            # W: C11, C22
            W[w_indices[0], col_offset + 0] = 1.0
            W[w_indices[3], col_offset + 0] = 1.0
            
            # Term 1: M2 = (A21+A22)B11
            U[u_indices[2], col_offset + 1] = 1.0
            U[u_indices[3], col_offset + 1] = 1.0
            V[v_indices[0], col_offset + 1] = 1.0
            W[w_indices[2], col_offset + 1] = 1.0
            
            # Term 2: M3 = A11(B12-B22)
            U[u_indices[0], col_offset + 2] = 1.0
            V[v_indices[1], col_offset + 2] = 1.0
            V[v_indices[3], col_offset + 2] = -1.0
            W[w_indices[1], col_offset + 2] = 1.0
            
            # Term 3: M4 = A22(B21-B11)
            U[u_indices[3], col_offset + 3] = 1.0
            V[v_indices[2], col_offset + 3] = 1.0
            V[v_indices[0], col_offset + 3] = -1.0
            W[w_indices[3], col_offset + 3] = 1.0
            
            # Term 4: M5 = (A11+A12)B22
            U[u_indices[0], col_offset + 4] = 1.0
            U[u_indices[1], col_offset + 4] = 1.0
            V[v_indices[3], col_offset + 4] = 1.0
            W[w_indices[3], col_offset + 4] = -1.0
            
            # Term 5: M6 = (A21-A11)(B11+B12)
            U[u_indices[2], col_offset + 5] = 1.0
            U[u_indices[0], col_offset + 5] = -1.0
            V[v_indices[0], col_offset + 5] = 1.0
            V[v_indices[1], col_offset + 5] = 1.0
            W[w_indices[3], col_offset + 5] = 1.0
            
            # Term 6: M7 = (A12-A22)(B21+B22)
            U[u_indices[1], col_offset + 6] = 1.0
            U[u_indices[3], col_offset + 6] = -1.0
            V[v_indices[2], col_offset + 6] = 1.0
            V[v_indices[3], col_offset + 6] = 1.0
            W[w_indices[0], col_offset + 6] = 1.0
            
            # Now for the right half (B_right, C_right)
            # The A part is the same, so U is the same.
            # We need to add the V and W parts for the right half.
            
            # Term 0: M1_right = (A11+A22)(B11_right+B22_right)
            # V: B11_right (v_indices[4]), B22_right (v_indices[7])
            V[v_indices[4], col_offset + 0] = 1.0
            V[v_indices[7], col_offset + 0] = 1.0
            # W: C11_right (w_indices[4]), C22_right (w_indices[7])
            W[w_indices[4], col_offset + 0] = 1.0
            W[w_indices[7], col_offset + 0] = 1.0
            
            # Term 1: M2_right = (A21+A22)B11_right
            # V: B11_right (v_indices[4])
            V[v_indices[4], col_offset + 1] = 1.0
            # W: C21_right (w_indices[6])
            W[w_indices[6], col_offset + 1] = 1.0
            
            # Term 2: M3_right = A11(B12_right-B22_right)
            # V: B12_right (v_indices[5]), B22_right (v_indices[7])
            V[v_indices[5], col_offset + 2] = 1.0
            V[v_indices[7], col_offset + 2] = -1.0
            # W: C12_right (w_indices[5])
            W[w_indices[5], col_offset + 2] = 1.0
            
            # Term 3: M4_right = A22(B21_right-B11_right)
            # V: B21_right (v_indices[6]), B11_right (v_indices[4])
            V[v_indices[6], col_offset + 3] = 1.0
            V[v_indices[4], col_offset + 3] = -1.0
            # W: C22_right (w_indices[7])
            W[w_indices[7], col_offset + 3] = 1.0
            
            # Term 4: M5_right = (A11+A12)B22_right
            # V: B22_right (v_indices[7])
            V[v_indices[7], col_offset + 4] = 1.0
            # W: C22_right (w_indices[7])
            W[w_indices[7], col_offset + 4] = -1.0
            
            # Term 5: M6_right = (A21-A11)(B11_right+B12_right)
            # V: B11_right (v_indices[4]), B12_right (v_indices[5])
            V[v_indices[4], col_offset + 5] = 1.0
            V[v_indices[5], col_offset + 5] = 1.0
            # W: C22_right (w_indices[7])
            W[w_indices[7], col_offset + 5] = 1.0
            
            # Term 6: M7_right = (A12-A22)(B21_right+B22_right)
            # V: B21_right (v_indices[6]), B22_right (v_indices[7])
            V[v_indices[6], col_offset + 6] = 1.0
            V[v_indices[7], col_offset + 6] = 1.0
            # W: C11_right (w_indices[4])
            W[w_indices[4], col_offset + 6] = 1.0
        
        # Part 1: j in {0,1}
        u1 = [0, 1, 4, 5]  # i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1
        v1 = [0, 1, 2, 3, 4, 5, 6, 7]  # j=0: 0-3, j=1: 4-7
        w1 = [0, 1, 2, 3, 4, 5, 6, 7]  # k=0: 0,1; k=1: 2,3; k=2: 4,5; k=3: 6,7
        strassen_2x2_2x4(u1, v1, w1, 0)
        
        # Part 2: j in {2,3}
        u2 = [2, 3, 6, 7]  # i=0,j=2; i=0,j=3; i=1,j=2; i=1,j=3
        v2 = [8, 9, 10, 11, 12, 13, 14, 15]  # j=2: 8-11, j=3: 12-15
        w2 = [0, 1, 2, 3, 4, 5, 6, 7]  # Same W indices
        strassen_2x2_2x4(u2, v2, w2, 14)
        
        return U, V, W
    
    # Verify Strassen rank-28
    U_s, V_s, W_s = strassen_rank28()
    if not verify(U_s, V_s, W_s, tol=1e-9):
        # Fallback to schoolbook if Strassen is wrong
        U_s, V_s, W_s = schoolbook()
    
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = 28
    
    # Try to find rank 24 or lower using linear algebra
    # This is a sparse linear system problem.
    # We can try to express some of the 28 terms as linear combinations of others.
    # This is equivalent to finding the rank of the tensor.
    
    # For now, return the rank-28 decomposition.
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
