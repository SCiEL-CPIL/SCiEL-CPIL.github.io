"""Winograd-based decomposition for 2x4 * 4x5 matrix multiplication."""

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    
    # Winograd F2T2 algorithm for 2x2 matrix multiply
    # F = [[1, 0], [1, 1], [1, -1], [0, 1]]
    # G = [[1, 0, 0], [1, 1, 1], [1, -1, -1], [0, 1, 1]]
    # M = diag([1, 1, 1, 1])
    # F^T = [[1, 1, 1, 0], [0, 1, -1, 1]]
    
    # For 2x2 * 2x2: rank 4
    # For 2x2 * 2x1: rank 2 (only need first two columns of B)
    
    # We decompose 2x4 * 4x5 as:
    # Split A into two 2x2 blocks: A_left (cols 0,1), A_right (cols 2,3)
    # Split B into two 4x2 blocks and one 4x1 block:
    #   B_left (cols 0,1), B_mid (cols 2,3), B_right (col 4)
    # C = A_left * B_left + A_right * B_mid + (A_left[:,0]*B[:,4] + A_right[:,0]*B[:,4])
    # Wait, that's not right. Let me reconsider.
    
    # Actually: C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Group by pairs of j:
    # = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    # For each pair, this is a 2x2 * 2x? operation where the "2x2" in A is 
    # actually just 2 elements (one per row), and B is 2 rows x p columns.
    
    # Let's use Winograd for the 2-element dot product with 2 rows of B.
    # For a fixed k: C[i,k] for i=0,1 from A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is like multiplying a 2x1 matrix by a 1x1 matrix... no.
    
    # Let me think differently. Use Winograd F2T2 for the j-dimension.
    # For each pair of j's (j, j+1), we compute contributions to all i,k.
    
    # The sub-tensor for j-pair {0,1}:
    # C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 2x2x5 tensor (i=0,1; j=0,1; k=0..4)
    
    # Using Winograd F2T2:
    # Transform A along j: A'[i,0] = A[i,0], A'[i,1] = A[i,0]+A[i,1], A'[i,2] = A[i,0]-A[i,1], A'[i,3] = A[i,1]
    # Transform B along j: B'[0,k] = B[0,k], B'[1,k] = B[0,k]+B[1,k], B'[2,k] = B[0,k]-B[1,k], B'[3,k] = B[1,k]
    # Multiply pointwise: M[t,k] = A'[i,t] * B'[t,k] (but A' depends on i too)
    # Actually, the Winograd algorithm for 2x2*2x2 gives rank 4.
    # For 2x2*2xp, we can use rank 4*p/2 = 2p? No, that's not right either.
    
    # Let me just use the standard Winograd F2T2 for each 2x2*2x2 block,
    # and handle the 2x2*2x1 blocks with a similar approach.
    
    # Actually, let's use a simpler approach: decompose into 2x2*2x2 and 2x2*2x1 blocks.
    
    # Rank of 2x2*2x2 via Winograd F2T2: 4
    # Rank of 2x2*2x1: We can use 2 multiplications (schoolbook) or Winograd-like.
    
    # For 2x2*2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is rank 4 by schoolbook, but we can do better.
    # Using Winograd: transform A along the 2-dim, transform B along the 2-dim.
    # A' = [A00, A00+A01, A00-A01, A01] for row 0
    #     = [A10, A10+A11, A10-A11, A11] for row 1
    # B' = [B0, B0+B1, B0-B1, B1]
    # M[t] = A'[i,t] * B'[t]
    # C[i] = sum_t F^T[t, i] * M[t] where F^T = [[1,1,1,0],[0,1,-1,1]]
    # So C[0] = M[0] + M[1] + M[2], C[1] = M[1] - M[2] + M[3]
    # This is still 4 multiplications. No gain for 2x2*2x1.
    
    # Let me just use schoolbook for 2x2*2x1 (rank 4) and Winograd for 2x2*2x2 (rank 4).
    # Total: 2 * (4 + 4 + 4) = 24? No wait.
    
    # Split B into B[0:2, :] (2x5) and B[2:4, :] (2x5)
    # A is 2x4, split into A[:, 0:2] (2x2) and A[:, 2:4] (2x2)
    # C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:]
    
    # Each term is 2x2 * 2x5.
    # For 2x2 * 2x5: split B into 2x2 + 2x2 + 2x1
    # Rank: 4 + 4 + 4 = 12 per term
    # Total: 24
    
    # But we can do better. Let me use a different decomposition.
    
    # Actually, let's just implement the Winograd F2T2 for 2x2*2x2 blocks
    # and schoolbook for the remaining 2x2*2x1 blocks.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 8 x 24
    V = [[0] * rank for _ in range(m * p)]  # 20 x 24
    W = [[0] * rank for _ in range(p * n)]  # 10 x 24
    
    col = 0
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1
    
    # Winograd F2T2 for 2x2 * 2x2
    # F = [[1,0],[1,1],[1,-1],[0,1]]
    # G = [[1,0,0],[1,1,1],[1,-1,-1],[0,1,1]]
    # F^T = [[1,1,1,0],[0,1,-1,1]]
    
    def add_winograd_2x2_2x2(a_i, a_j, b_i, b_j, c_i, c_j):
        # a_i, a_j: base indices for A block (2x2)
        # b_i, b_j: base indices for B block (2x2)
        # c_i, c_j: base indices for C block (2x2)
        
        # A local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # B local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # C local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        def map_a(li):
            return (a_i + li // 2) * m + (a_j + li % 2)
        def map_b(li):
            return (b_i + li // 2) * p + (b_j + li % 2)
        def map_c(li):
            i = c_i + li // 2
            k = c_j + li % 2
            return k * n + i
        
        # Winograd F2T2:
        # M0 = (A00)(B00)
        # M1 = (A00+A01)(B00+B01+B10+B11) ... no, that's not right.
        
        # Let me use the correct Winograd F2T2:
        # A' = F * A where F = [[1,0],[1,1],[1,-1],[0,1]]
        # So A'[0] = A00, A'[1] = A00+A01, A'[2] = A00-A01, A'[3] = A01 for row 0
        #     A'[0] = A10, A'[1] = A10+A11, A'[2] = A10-A11, A'[3] = A11 for row 1
        
        # B' = G * B where G = [[1,0,0],[1,1,1],[1,-1,-1],[0,1,1]]
        # B'[0] = B00, B'[1] = B00+B01+B10+B11, B'[2] = B00-B01-B10-B11, B'[3] = B01+B10+B11
        
        # Wait, G is 4x3, but B is 2x2. I think I'm confusing things.
        
        # Let me just use the standard Strassen-like decomposition for 2x2*2x2 with rank 4.
        # Actually, the minimal rank for 2x2*2x2 is 7 (Strassen), but Winograd F2T2 gives 4?
        # No, Winograd F2T2 is for 2x2 convolution, not matrix multiplication.
        
        # Let me just use schoolbook for 2x2*2x2 (rank 4) and see if I can do better.
        
        # Schoolbook 2x2*2x2:
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        add_rank1_term({map_a(0): 1}, {map_b(0): 1}, {map_c(0): 1})
        add_rank1_term({map_a(1): 1}, {map_b(2): 1}, {map_c(0): 1})
        add_rank1_term({map_a(0): 1}, {map_b(1): 1}, {map_c(1): 1})
        add_rank1_term({map_a(1): 1}, {map_b(3): 1}, {map_c(1): 1})
        add_rank1_term({map_a(2): 1}, {map_b(0): 1}, {map_c(2): 1})
        add_rank1_term({map_a(3): 1}, {map_b(2): 1}, {map_c(2): 1})
        add_rank1_term({map_a(2): 1}, {map_b(1): 1}, {map_c(3): 1})
        add_rank1_term({map_a(3): 1}, {map_b(3): 1}, {map_c(3): 1})
    
    def add_schoolbook_2x2_2x1(a_i, a_j, b_i, b_j, c_i, c_j):
        # A is 2x2, B is 2x1, C is 2x1
        def map_a(li):
            return (a_i + li // 2) * m + (a_j + li % 2)
        def map_b(li):
            return (b_i + li) * p + b_j
        def map_c(li):
            i = c_i + li
            k = c_j
            return k * n + i
        
        add_rank1_term({map_a(0): 1}, {map_b(0): 1}, {map_c(0): 1})
        add_rank1_term({map_a(1): 1}, {map_b(1): 1}, {map_c(0): 1})
        add_rank1_term({map_a(2): 1}, {map_b(0): 1}, {map_c(1): 1})
        add_rank1_term({map_a(3): 1}, {map_b(1): 1}, {map_c(1): 1})
    
    # Block 1: A[:,0:2] * B[0:2,:]
    # B[0:2,:] is 2x5, split into cols 0,1 (2x2), cols 2,3 (2x2), col 4 (2x1)
    add_winograd_2x2_2x2(0, 0, 0, 0, 0, 0)
    add_winograd_2x2_2x2(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x2_2x1(0, 0, 0, 4, 0, 4)
    
    # Block 2: A[:,2:4] * B[2:4,:]
    add_winograd_2x2_2x2(0, 2, 2, 0, 0, 0)
    add_winograd_2x2_2x2(0, 2, 2, 2, 0, 2)
    add_schoolbook_2x2_2x1(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
