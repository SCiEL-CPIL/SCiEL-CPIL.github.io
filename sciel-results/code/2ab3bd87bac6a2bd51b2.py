"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    # The matrix multiplication tensor T[i,j,k] = delta (i.e., T[i,j,k] = 1 if we think of it as the standard matmul tensor)
    # We want to find a rank-r decomposition: T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # For 2x4x6 matmul, the tensor has dimensions (2,4,6) with 48 non-zero entries (all 1s in the standard pattern)
    # Actually, the matmul tensor T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's a 3-way tensor where each entry is 1)
    # Wait, no. The tensor for matrix multiplication is defined as:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid i,j,k
    # 
    # Actually, let me reconsider. The matrix multiplication tensor is:
    # mu_{i,j,k} where the product is sum_{j} A_{i,j} B_{j,k} = C_{i,k}
    # The tensor T has entries T[i,j,k] and the multiplication is:
    # C[i,k] = sum_{j,r} U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k]
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta (which is 1 for the matmul tensor)
    # 
    # So the tensor we're decomposing has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Known lower bound: rank >= n*m*p / max(n,m,p) ... not quite
    # For 2x4x6, the standard algorithm has rank 2*4*6 = 48
    # Strassen-based: split into 2x2x2 blocks
    # A is 2x4, B is 4x6
    # We can split A into two 2x2 blocks (A0, A1) and B into three 2x2 blocks (B0, B1, B2)
    # C = A0*B0 + A1*B1 (where each product is 2x2)
    # Wait, that's not right either.
    
    # Let me think more carefully:
    # A is 2x4: A = [A_left | A_right] where A_left, A_right are 2x2
    # B is 4x6: B = [B_top; B_bottom] where B_top, B_bottom are 2x6
    # C = A*B = A_left * B_top + A_right * B_bottom (each is 2x2 * 2x6 = 2x6)
    
    # Now split B_top into three 2x2 blocks: B_top = [B_top_0 | B_top_1 | B_top_2]
    # Similarly B_bottom = [B_bottom_0 | B_bottom_1 | B_bottom_2]
    
    # C[:, 0:2] = A_left * B_top_0 + A_right * B_bottom_0
    # C[:, 2:4] = A_left * B_top_1 + A_right * B_bottom_1
    # C[:, 4:6] = A_left * B_top_2 + A_right * B_bottom_2
    
    # Each of these is a sum of two 2x2 * 2x2 products.
    # We can use Strassen for each 2x2*2x2 product (rank 7), so total rank = 6 * 7 = 42
    
    # But can we do better? Let's think about using a shared Strassen decomposition.
    # 
    # Actually, let me try a different approach. Let's use the fact that we can combine the computations.
    # 
    # For C[:, 0:2]: we need A_left * B_top_0 + A_right * B_bottom_0
    # This is a 2x4 * 4x2 = 2x2 product, which can be done with rank at most 4 (schoolbook) or 7 (Strassen on 2x2x2... wait no)
    # Actually 2x4 * 4x2 is just a 2x2 result with 4 multiplications total (schoolbook).
    # Using Strassen: split into 2x2 * 2x2 blocks, so 4 sub-products of 2x2*2x2, each with rank 7, total 28. That's worse.
    
    # So the Strassen approach with 6 blocks of 2x2*2x2 giving rank 42 is good.
    # Can we do better than 42?
    
    # Lower bound considerations:
    # The border rank of the 2x4x6 matmul tensor... 
    # For m x n x p matmul, the rank is at least max(m*n, n*p, m*p) / min(m,n,p) ... not a clean bound.
    # 
    # Actually, for 2x4x6:
    # - The "slice rank" lower bound: rank >= max(n*m, m*p, p*n) / min(n,m,p) = max(8, 24, 12)/2 = 12
    # - But this is a weak bound.
    # 
    # Let me just implement the rank-42 Strassen-based solution and verify it works.
    # Then I'll try to see if I can find something better.
    
    def strassen_2x2x2():
        """
        Return U, V, W for 2x2x2 Strassen multiplication.
        U: 4x7 (rows: A00, A01, A10, A11), V: 4x7 (rows: B00, B01, B10, B11), W: 4x7 (rows: C00, C01, C10, C11)
        """
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (A00+A10)*(B00+B10)
        U_s[0, 0] = 1; U_s[2, 0] = 1
        V_s[0, 0] = 1; V_s[2, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1
        
        # M3 = A00*(B01-B11)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1
        
        # M4 = A11*(B10-B00)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1
        
        return U_s, V_s, W_s
    
    U_s, V_s, W_s = strassen_2x2x2()
    
    # Build the 2x4x6 decomposition
    # A: 2x4, indexed as A[i, a_col] where i in {0,1}, a_col in {0,1,2,3}
    # B: 4x6, indexed as B[b_row, b_col] where b_row in {0,1,2,3}, b_col in {0,1,2,3,4,5}
    # C: 2x6, indexed as C[i, c_col] where i in {0,1}, c_col in {0,1,2,3,4,5}
    # W index: k*n + i where k is the C column, i is the C row
    
    rank = 42
    U = np.zeros((n*m, rank))  # 8 x 42
    V = np.zeros((m*p, rank))  # 24 x 42
    W = np.zeros((p*n, rank))  # 12 x 42
    
    # 6 blocks: for each (a_half, b_col_block)
    # a_half in {0, 1}: which half of A's columns (0: cols 0-1, 1: cols 2-3)
    # b_col_block in {0, 1, 2}: which third of B's columns (0: cols 0-1, 1: cols 2-3, 2: cols 4-5)
    # b_row_half in {0, 1}: which half of B's rows (0: rows 0-1, 1: rows 2-3)
    # 
    # For block (a_half, b_col_block):
    #   A submatrix: A[:, a_half*2 : a_half*2+2] (2x2)
    #   B submatrix: B[a_half*2 : a_half*2+2, b_col_block*2 : b_col_block*2+2] (2x2)
    #   C submatrix: C[:, b_col_block*2 : b_col_block*2+2] (2x2)
    
    for a_half in range(2):
        for b_col_block in range(3):
            b_row_half = a_half  # A_left pairs with B_top, A_right pairs with B_bottom
            col_offset = (a_half * 3 + b_col_block) * 7
            
            for j in range(4):
                for k in range(7):
                    val = U_s[j, k]
                    if val != 0:
                        # U index: i*m + a_col where i is A row, a_col is A column
                        a_row = j // 2
                        a_col = a_half * 2 + (j % 2)
                        U[a_row * m + a_col, col_offset + k] = val
                    
                    val = V_s[j, k]
                    if val != 0:
                        # V index: b_row*p + b_col
                        b_row = b_row_half * 2 + (j // 2)
                        b_col = b_col_block * 2 + (j % 2)
                        V[b_row * p + b_col, col_offset + k] = val
                    
                    val = W_s[j, k]
                    if val != 0:
                        # W index: c_col*n + c_row
                        c_row = j // 2
                        c_col = b_col_block * 2 + (j % 2)
                        W[c_col * n + c_row, col_offset + k] = val
    
    # Verify
    def verify_decomposition(U, V, W, n, m, p, rank):
        random.seed(12345)
        for trial in range(5):
            A = np.random.randint(-3, 4, size=(n, m))
            B = np.random.randint(-3, 4, size=(m, p))
            C_direct = A @ B
            
            C_decomp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    total = 0
                    for j in range(m):
                        for r in range(rank):
                            total += U[i*m+j, r] * A[i, j] * V[j*p+k, r] * B[j, k] * W[k*n+i, r]
                    C_decomp[i, k] = total
            
            if not np.allclose(C_decomp, C_direct, atol=1e-10):
                return False
        return True
    
    if verify_decomposition(U, V, W, n, m, p, rank):
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook rank 48
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
