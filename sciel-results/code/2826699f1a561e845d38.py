"""Matrix multiplication 2x4x5 tensor decomposition using Winograd-style polynomial methods."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We'll use a decomposition based on splitting the computation.
    # C (2x5) = A (2x4) * B (4x5)
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Strategy: Use a rank-36 decomposition based on block structure.
    # Split A into two 2x2 blocks: A_L (cols 0,1) and A_R (cols 2,3)
    # Split B into two 2x5 blocks: B_T (rows 0,1) and B_B (rows 2,3)
    # C = A_L * B_T + A_R * B_B
    
    # For each 2x2 * 2x5 block, we can use a decomposition.
    # 2x2 * 2x5: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1; k=0..4
    # This is 2*5 = 10 scalar multiplications if done naively.
    # But we can use Strassen-like ideas for the 2x2 part.
    
    # Actually, let's think of it as: for each output column k, we have a 2x1 = 2x2 * 2x1 product.
    # 2x2 * 2x1 can be done in rank 4 (schoolbook) or potentially less?
    # No, 2x2*2x1 is just two dot products of length 2, so rank 4 is minimal.
    
    # Alternative: Use a global approach.
    # Let's try to construct a rank-36 solution using the following:
    # For the 2x2 * 2x5 block, we can use 7 multiplications for the 2x2*2x2 part (cols 0,1)
    # and 4 multiplications for the 2x2*2x1 part (col 4)? No, that's 3 columns.
    
    # Let me reconsider. The current program uses rank 36.
    # 2 blocks * (7 + 7 + 4) = 2 * 18 = 36.
    # Each block: 2x2 * 2x5 split into 2x2*2x2 (rank 7), 2x2*2x2 (rank 7), 2x2*2x1 (rank 4).
    # Total 18 per block, 36 total.
    
    # Can we do better? The lower bound for 2x4x5 is at least max(2*4, 4*5, 2*5) = 20.
    # But that's a very loose bound.
    
    # Let me try a different approach: use a direct rank decomposition.
    # We want to find U (8xr), V (20xr), W (10xr) such that
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # where T[i,j,k,l] = 1 if i*n+k = l (output index) and the product is A[i,j]*B[j,k].
    
    # Actually, the tensor is T[a,b,c] where:
    # a indexes A (8 elements: i*m+j, i=0..1, j=0..3)
    # b indexes B (20 elements: j*p+k, j=0..3, k=0..4)
    # c indexes C (10 elements: k*n+i, k=0..4, i=0..1)
    # T[a,b,c] = 1 if a's j-component equals b's j-component AND c's (k,i) matches (b's k, a's i)
    
    # Let me build the tensor and try to find a low-rank decomposition using SVD or similar.
    # But the entries need to be integer/half-integer, so I'll construct it analytically.
    
    # Let's use the following construction:
    # For each j in 0..3, and each (i,k) pair, we have a term A[i,j]*B[j,k] contributing to C[i,k].
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # A simple decomposition: for each (i,j,k), one rank-1 term.
    # That's 2*4*5 = 40 terms. Rank 40.
    
    # Can we reduce this? 
    # For fixed j, the terms are A[i,j]*B[j,k] for i=0,1 and k=0..4.
    # That's 2*5 = 10 terms per j, 4*10 = 40 total.
    
    # Using Strassen on the 2x2 blocks:
    # Split into j-pairs: (0,1) and (2,3).
    # For j-pair (0,1): A has columns 0,1; B has rows 0,1.
    # C contribution: sum_{j in {0,1}} A[i,j]*B[j,k]
    # For each k, this is a 2x1 = 2x2 * 2x1 product.
    # 2x2 * 2x1: rank 4 (schoolbook) or can we do better?
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # These are independent for different k, so we need 4 multiplications per k-pair? No, per k.
    # For each k, we need 4 multiplications. 5 k values, so 20 per j-pair? No.
    
    # Wait, for a fixed j-pair and fixed k, we have a 2x2 * 2x1 product which is rank 4.
    # But the B[j,k] values are shared across i, and A[i,j] are shared across k.
    
    # Let me just implement the rank-40 schoolbook decomposition and see if I can improve.
    
    # Actually, let me try a smarter approach. Use the fact that we can combine terms.
    
    # For now, let me implement a rank-36 solution similar to the current one but cleaner.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_rank1(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1
    
    # Strassen 2x2 * 2x2
    def strassen_2x2(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        # A block: rows a_i0..a_i0+1, cols a_j0..a_j0+1
        # B block: rows b_i0..b_i0+1, cols b_j0..b_j0+1
        # C block: rows c_i0..c_i0+1, cols c_k0..c_k0+1
        # W index: k*n + i
        
        def a_idx(li, lj):
            return (a_i0 + li) * m + (a_j0 + lj)
        
        def b_idx(li, lj):
            return (b_i0 + li) * p + (b_j0 + lj)
        
        def c_idx(li, lk):
            # C[li, lk] -> W index = (c_k0 + lk) * n + (c_i0 + li)
            return (c_k0 + lk) * n + (c_i0 + li)
        
        # M1 = (A00+A11)(B00+B11)
        add_rank1({a_idx(0,0): 1, a_idx(1,1): 1},
                   {b_idx(0,0): 1, b_idx(1,1): 1},
                   {c_idx(0,0): 1, c_idx(1,1): 1})
        
        # M2 = (A10+A11)(B00)
        add_rank1({a_idx(1,0): 1, a_idx(1,1): 1},
                   {b_idx(0,0): 1},
                   {c_idx(1,0): 1, c_idx(1,1): -1})
        
        # M3 = A00(B01-B11)
        add_rank1({a_idx(0,0): 1},
                   {b_idx(0,1): 1, b_idx(1,1): -1},
                   {c_idx(0,1): 1, c_idx(1,1): 1})
        
        # M4 = A11(B10-B00)
        add_rank1({a_idx(1,1): 1},
                   {b_idx(1,0): 1, b_idx(0,0): -1},
                   {c_idx(0,0): 1, c_idx(1,0): 1})
        
        # M5 = (A00+A01)(B11)
        add_rank1({a_idx(0,0): 1, a_idx(0,1): 1},
                   {b_idx(1,1): 1},
                   {c_idx(0,0): -1, c_idx(0,1): 1})
        
        # M6 = (A10-A00)(B00+B01)
        add_rank1({a_idx(1,0): 1, a_idx(0,0): -1},
                   {b_idx(0,0): 1, b_idx(0,1): 1},
                   {c_idx(1,1): 1})
        
        # M7 = (A01-A11)(B10+B11)
        add_rank1({a_idx(0,1): 1, a_idx(1,1): -1},
                   {b_idx(1,0): 1, b_idx(1,1): 1},
                   {c_idx(0,0): 1})
    
    # Schoolbook 2x2 * 2x1
    def schoolbook_2x2_1(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        # A: 2x2 at (a_i0, a_j0)
        # B: 2x1 at (b_i0, b_j0)
        # C: 2x1 at (c_i0, c_k0)
        
        def a_idx(li, lj):
            return (a_i0 + li) * m + (a_j0 + lj)
        
        def b_idx(li):
            return (b_i0 + li) * p + b_j0
        
        def c_idx(li):
            return c_k0 * n + (c_i0 + li)
        
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        add_rank1({a_idx(0,0): 1}, {b_idx(0): 1}, {c_idx(0): 1})
        add_rank1({a_idx(0,1): 1}, {b_idx(1): 1}, {c_idx(0): 1})
        
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        add_rank1({a_idx(1,0): 1}, {b_idx(0): 1}, {c_idx(1): 1})
        add_rank1({a_idx(1,1): 1}, {b_idx(1): 1}, {c_idx(1): 1})
    
    # Block 1: A cols 0,1; B rows 0,1
    # Split B cols into (0,1), (2,3), (4)
    strassen_2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2(0, 0, 0, 2, 0, 2)
    schoolbook_2x2_1(0, 0, 0, 4, 0, 4)
    
    # Block 2: A cols 2,3; B rows 2,3
    strassen_2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2(0, 2, 2, 2, 0, 2)
    schoolbook_2x2_1(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
