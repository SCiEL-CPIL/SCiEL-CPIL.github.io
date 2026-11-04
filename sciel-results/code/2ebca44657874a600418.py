"""Tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def strassen_2x2x2():
        """Strassen's 7-product decomposition for 2x2 * 2x2 = 2x2.
        Returns U (4x7), V (4x7), W (4x7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A: A11=0, A12=1, A21=2, A22=3
        # B: B11=0, B12=1, B21=2, B22=3
        # C: C11=0, C12=1, C21=2, C22=3
        
        # M1 = (A11+A22)*(B11+B22)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1
        
        # M2 = (A21+A22)*B11
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=1
        
        # M3 = A11*(B12-B22)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        
        # M4 = A22*(B21-B11)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        
        # M5 = (A11+A12)*B22
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        
        # M6 = (A21-A11)*(B11+B12)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        # M7 = (A12-A22)*(B21+B22)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        
        return U, V, W

    def strassen_4x4x4():
        """Recursive Strassen for 4x4 * 4x4 = 4x4. Rank 49.
        Returns U (16x49), V (16x49), W (16x49)."""
        # Get base 2x2x2 decomposition
        U2, V2, W2 = strassen_2x2x2()
        
        # Define the 7 linear combinations for the 2x2 blocks
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Coefficients for A blocks (indices 0:11, 1:12, 2:21, 3:22)
        # Coefficients for B blocks
        # Coefficients for C blocks (C11, C12, C21, C22)
        
        # We need to map the 2x2x2 decomposition to the 4x4x4 context.
        # Each M_k is a 2x2 matrix product.
        # The inputs to this product are linear combinations of the 2x2 blocks of A and B.
        
        # Let's define the linear combinations for A and B blocks for each M_k
        # a_coeffs[k][block_idx] = coeff
        # b_coeffs[k][block_idx] = coeff
        # c_coeffs[block_idx][k] = coeff (for C = sum k c_block_k * M_k)
        
        a_coeffs = [
            [1, 0, 0, 1], # M1: A11 + A22
            [0, 0, 1, 1], # M2: A21 + A22
            [1, 0, 0, 0], # M3: A11
            [0, 0, 0, 1], # M4: A22
            [1, 1, 0, 0], # M5: A11 + A12
            [0, 0, 1, -1], # M6: A21 - A11 (wait, M6 = (A21-A11)) -> [ -1, 0, 1, 0 ]? 
                           # Standard Strassen: M6 = (A21 - A11)(B11 + B12). 
                           # My previous code: U[2,5]=1; U[0,5]=-1. So A21 - A11.
                           # So coeff for A11 is -1, A21 is 1.
                           # Block indices: 0:A11, 1:A12, 2:A21, 3:A22.
                           # So [-1, 0, 1, 0].
            [0, 1, 0, -1]  # M7: A12 - A22. U[1,6]=1; U[3,6]=-1. So A12 - AA22.
                           # Block indices: 0:A11, 1:A12, 2:A21, 3:A22.
                           # So [0, 1, 0, -1].
        ]
        # Correcting M6:
        a_coeffs[5] = [-1, 0, 1, 0]
        
        b_coeffs = [
            [1, 0, 0, 1], # M1: B11 + B22
            [1, 0, 0, 0], # M2: B11
            [0, 1, 0, -1], # M3: B12 - B22. V[1,2]=1; V[3,2]=-1. So B12 - B22.
            [0, 0, 1, -1], # M4: B21 - B11. V[2,3]=1; V[0,3]=-1. So B21 - B11.
            [0, 0, 0, 1], # M5: B22
            [1, 1, 0, 0], # M6: B11 + B12. V[0,5]=1; V[1,5]=1.
            [0, 0, 1, 1]  # M7: B21 + B22. V[2,6]=1; V[3,6]=1.
        ]
        
        # C = [[C11, C12], [C21, C22]]
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # c_coeffs[block_idx][k]
        c_coeffs = [
            [1, 0, 0, -1, 1, 0, -1], # C11
            [0, 0, 1, 0, 0, 0, 1],   # C12
            [0, 1, 0, 1, 0, 0, 0],   # C21
            [1, -1, 1, 0, 0, 1, 0]   # C22
        ]
        
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # Map block indices to global indices
        # Block 0 (A11): rows 0,1; cols 0,1 -> global rows 0,1; cols 0,1
        # Block 1 (A12): rows 0,1; cols 2,3 -> global rows 0,1; cols 2,3
        # Block 2 (A21): rows 2,3; cols 0,1 -> global rows 2,3; cols 0,1
        # Block 3 (A22): rows 2,3; cols 2,3 -> global rows 2,3; cols 2,3
        
        # Global index for A[i,j] is i*4+j.
        # Block b has top-left at (2*br, 2*bc).
        # Local index (r,c) in 2x2 block maps to global (2*br+r, 2*bc+c).
        
        block_offsets_a = [
            [0, 1, 4, 5],   # A11
            [2, 3, 6, 7],   # A12
            [8, 9, 12, 13], # A21
            [10, 11, 14, 15] # A22
        ]
        
        block_offsets_b = [
            [0, 1, 4, 5],   # B11
            [2, 3, 6, 7],   # B12
            [8, 9, 12, 13], # B21
            [10, 11, 14, 15] # B22
        ]
        
        block_offsets_c = [
            [0, 1, 4, 5],   # C11
            [2, 3, 6, 7],   # C12
            [8, 9, 12, 13], # C21
            [10, 11, 14, 15] # C22
        ]
        
        for k in range(7):
            # For each Strassen product k
            # We have a 2x2x2 decomposition U2, V2, W2
            # U2 is 4x7. Rows correspond to the 4 elements of the 2x2 matrix L_k(A).
            # The elements of L_k(A) are linear combinations of the elements of the blocks.
            
            # Construct the mapping from local 2x2 elements to global 4x4 elements for U
            # Local element l (0..3) corresponds to block element (l//2, l%2) in the 2x2 matrix L_k(A).
            # L_k(A)[r,c] = sum_b a_coeffs[k][b] * Block_b[r,c]
            # Block_b[r,c] is at global index block_offsets_a[b][l] where l is local index in block.
            
            # So, for a given local index l in the 2x2 matrix L_k(A):
            # It receives contributions from all blocks b where a_coeffs[k][b] != 0.
            # Specifically, L_k(A)[l] = sum_b a_coeffs[k][b] * A_global[block_offsets_a[b][l]]
            
            # The row l of U2 gives the weight of L_k(A)[l] in the products.
            # So the contribution to U4[global_idx, k*7 + p] is:
            # sum_l U2[l, p] * (coefficient of A_global[global_idx] in L_k(A)[l])
            
            # Let's iterate over global indices for A
            for g in range(16):
                # Determine which block and local index g corresponds to
                # g = i*4+j
                i = g // 4
                j = g % 4
                br = i // 2
                bc = j // 2
                b = br * 2 + bc
                lr = i % 2
                lc = j % 2
                l = lr * 2 + lc
                
                # The coefficient of A[g] in L_k(A)[l] is a_coeffs[k][b]
                # Note: L_k(A)[l] is the element at local index l.
                # If A[g] is in block b, it contributes to L_k(A)[l] with weight a_coeffs[k][b].
                
                coeff = a_coeffs[k][b]
                if coeff != 0:
                    # Add contribution to U4
                    U4[g, k*7:(k+1)*7] += coeff * U2[l, :]
            
            # Similarly for V
            for g in range(16):
                i = g // 4
                j = g % 4
                br = i // 2
                bc = j // 2
                b = br * 2 + bc
                lr = i % 2
                lc = j % 2
                l = lr * 2 + lc
                
                coeff = b_coeffs[k][b]
                if coeff != 0:
                    V4[g, k*7:(k+1)*7] += coeff * V2[l, :]
            
            # For W
            # The output of product k is a 2x2 matrix P_k.
            # P_k[l] is the l-th element.
            # The global C is formed by summing P_k with coefficients c_coeffs[block][k].
            # C_block[b][l] = sum_k c_coeffs[b][k] * P_k[l]
            # C_global[g] is in block b, local index l.
            # C_global[g] = sum_k c_coeffs[b][k] * P_k[l]
            # P_k[l] is the l-th row of W2? No, W2 maps products to output elements.
            # W2[l, p] is the weight of product p in output element l of P_k.
            # So P_k[l] = sum_p W2[l, p] * M_p (where M_p is the scalar product).
            # Thus C_global[g] = sum_k c_coeffs[b][k] * sum_p W2[l, p] * M_p
            # = sum_p (sum_k c_coeffs[b][k] * W2[l, p]) * M_p
            
            for g in range(16):
                i = g // 4
                j = g % 4
                br = i // 2
                bc = j // 2
                b = br * 2 + bc
                lr = i % 2
                lc = j % 2
                l = lr * 2 + lc
                
                # Sum over k
                for k in range(7):
                    coeff = c_coeffs[b][k]
                    if coeff != 0:
                        W4[g, k*7:(k+1)*7] += coeff * W2[l, :]
        
        return U4, V4, W4

    def schoolbook_4x4x3():
        """Schoolbook decomposition for 4x4 * 4x3 = 4x3. Rank 48.
        Returns U (16x48), V (12x48), W (12x48)."""
        U = np.zeros((16, 48))
        V = np.zeros((12, 48))
        W = np.zeros((12, 48))
        
        # A is 4x4, B is 4x3, C is 4x3
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W

    def decompose_4x4x7():
        U4, V4, W4 = strassen_4x4x4()
        U3, V3, W3 = schoolbook_4x4x3()
        
        total_rank = 49 + 48
        U = np.zeros((16, total_rank))
        V = np.zeros((28, total_rank))
        W = np.zeros((28, total_rank))
        
        # Part 1: 4x4x4 (cols 0-3 of B, cols 0-3 of C)
        U[:, 0:49] = U4
        # V rows 0-15 correspond to B cols 0-3
        V[0:16, 0:49] = V4
        # W rows corresponding to C cols 0-3
        # C is 4x7, row-major k*4+i
        # Cols 0-3 -> k=0,1,2,3
        # Indices: k*4+i for k in 0..3, i in 0..3 -> 0..15
        W[0:16, 0:49] = W4
        
        # Part 2: 4x4x3 (cols 4-6 of B, cols 4-6 of C)
        U[:, 49:97] = U3
        # V rows 16-27 correspond to B cols 4-6
        # B is 4x7. Rows j=0..3, Cols k=4,5,6
        # Indices: j*7+k
        # j=0: 4,5,6
        # j=1: 11,12,13
        # j=2: 18,19,20
        # j=3: 25,26,27
        b_indices = [4,5,6, 11,12,13, 18,19,20, 25,26,27]
        V[b_indices, 49:97] = V3
        
        # W rows corresponding to C cols 4-6
        # C is 4x7. Rows i=0..3, Cols k=4,5,6
        # Indices: k*4+i
        # k=4: 16,17,18,19
        # k=5: 20,21,22,23
        # k=6: 24,25,26,27
        c_indices = [16,17,18,19, 20,21,22,23, 24,25,26,27]
        W[c_indices, 49:97] = W3
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank

    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return float(np.max(np.abs(result - 1.0)))

    best_result = None
    best_rank = n * m * p

    try:
        result, rank = decompose_4x4x7()
        err = verify(result["U"], result["V"], result["W"])
        if err < 1e-9:
            best_result = result
            best_rank = rank
    except Exception as e:
        pass

    if best_result is None:
        best_result = make_schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
