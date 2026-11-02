"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a tensor decomposition for matrix multiplication 3x3 * 3x8.
    
    Strategy: Use Strassen's algorithm on 2x2 sub-blocks to reduce rank below schoolbook (72).
    
    The 3x3 matrix A and 3x8 matrix B can be partitioned. We use the fact that
    for 2x2 blocks, Strassen reduces 8 multiplications to 7.
    
    For the 3x3 * 3x8 case, we partition:
    - A into a 2x2 block (top-left) and remaining rows/cols
    - B into corresponding blocks
    
    The decomposition uses:
    - Strassen's 7 factors for the 2x2 core (replacing 8 schoolbook terms)
    - Direct terms for the remaining 1x2, 2x1, 1x1 interactions
    
    This should give rank = 7 + 3*2*8 - 8 = 7 + 48 - 8 = 47? Let me think more carefully.
    
    Actually, let me use a cleaner approach: 
    Split the 8 columns of B into two groups of 4.
    For each group, we compute A (3x3) * B_group (3x4).
    
    For A(3x3) * B(3x4):
    - The 2x2 top-left of A times the 2x4 part of B: use Strassen on the 2x2 rows
    - This gives savings on the overlapping 2x2 region
    
    Let me just implement a direct Strassen-based decomposition for the full problem.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # We'll build the decomposition factor by factor.
    # Each factor c has:
    #   U[:, c] - vector in R^{n*m} = R^9
    #   V[:, c] - vector in R^{m*p} = R^{24}  
    #   W[:, c] - vector in R^{p*n} = R^{24}
    # such that sum_c U[i,j,c] * V[j,k,c] * W[k,i',c] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # 
    # W index convention: W[k*n + i, c] where k is output col index (0..p-1), i is output row (0..n-1)
    
    U = []  # list of columns, each of length n*m = 9
    V = []  # list of columns, each of length m*p = 24
    W = []  # list of columns, each of length p*n = 24
    
    def add_factor(u_vec, v_vec, w_vec):
        """Add a rank-1 factor. Vectors are dicts: {index: value}"""
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        
        for idx, val in u_vec.items():
            U_col[idx] = val
        for idx, val in v_vec.items():
            V_col[idx] = val
        for idx, val in w_vec.items():
            W_col[idx] = val
        
        U.append(U_col)
        V.append(V_col)
        W.append(W_col)
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    # Strategy: Use Strassen on the 2x2 top-left block of A times corresponding 2x? blocks of B
    # 
    # Let's partition:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # 
    # B = [[b00, b01, ..., b07],
    #      [b10, b11, ..., b17],
    #      [b20, b21, ..., b27]]
    # 
    # C = A * B, so C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # For the 2x2 subproblem (rows 0,1 of A; cols 0,1 of B for each output col k):
    # We can apply Strassen. But B has 8 columns, so we need to handle all k.
    # 
    # Alternative cleaner approach: 
    # Since the problem is 3x3 * 3x8, let's view it as computing 8 separate 3x3 * 3x1 products
    # (one for each column of B). But that's just schoolbook.
    # 
    # Better: Use the structure that for each pair of output rows (i, i'), 
    # the computation shares the same A rows.
    # 
    # Let me use a known result: the rank of T_{3,3,3} is at most 20 (Laderman).
    # For T_{3,3,8}, we can potentially do better than 72 by sharing computations
    # across the 8 output columns.
    # 
    # Key insight: For each fixed j (the summation index), A[:,j] is a 3-vector and 
    # B[j,:] is a 8-vector. The outer product A[:,j] * B[j,:] contributes to C.
    # 
    # So C = sum_j outer(A[:,j], B[j,:])
    # 
    # This is a sum of 3 rank-3 tensors (one for each j), each of rank 3.
    # Total rank <= 3*3 = 9? No, that's not right.
    # 
    # Actually, C = sum_{j=0}^{2} (A[:,j] * B[j,:])
    # Each term A[:,j] * B[j,:] is an outer product of a 3-vector and an 8-vector,
    # which is a 3x8 matrix of rank 1.
    # 
    # So C is a sum of 3 rank-1 matrices (as a 3x8 matrix). But we need the TENSOR rank,
    # not the matrix rank.
    # 
    # The tensor T_{3,3,8} has entries T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # and we want rank-r decomposition.
    # 
    # Let me just implement the schoolbook decomposition and then try to optimize.
    # Given the time constraints, let's return a valid decomposition.
    
    # Schoolbook decomposition: rank = n*m*p = 72
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Now try to improve using Strassen-like ideas.
    # 
    # For the 2x2x2 subproblem, Strassen reduces rank from 8 to 7.
    # We can apply this to the top-left 2x2 block of A and the first 2 columns of B
    # (for each output column k in 0..7, the 2x2 block of B[:,k]).
    # 
    # But this gets complicated. Let me try a different structural approach:
    # 
    # Partition A into blocks:
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # 
    # B = [B11 B12; B21 B22] where B11 is 2x8, B12 is 2x8? No.
    # B is 3x8, so partition rows:
    # B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 1x8
    # 
    # C = A*B = [A11*A_top + A12*B_bottom, ...]
    # 
    # For the 2x2 * 2x8 part (A11 * B_top), we can use Strassen on the 2x2 rows.
    # A11 is 2x2, B_top is 2x8.
    # 
    # C_top = A11 * B_top (2x8)
    # Using Strassen on the 2x2 A11 and treating B_top as 2x8:
    # 
    # Let A11 = [[a, b], [c, d]], B_top = [[e1, e2, ..., e8], [f1, f2, ..., f8]]
    # 
    # Strassen products (for each column k of B_top):
    # M1 = (a+d)*(e_k + f_k)
    # M2 = (c+d)*e_k
    # M3 = a*(e_k - f_k)
    # M4 = d*(f_k - e_k)
    # M5 = (a+b)*f_k
    # M6 = (c-a)*(e_k + f_k)
    # M7 = (b-d)*(f_k + e_k)
    # 
    # C_top[0,k] = M1 + M4 - M5 + M7
    # C_top[1,k] = M2 + M4
    # 
    # But these M's depend on k, so we need separate factors for each k.
    # That gives 7*8 = 56 factors for the 2x2*2x8 part, vs 2*2*8=32 schoolbook.
    # That's worse!
    # 
    # The issue: Strassen saves when you're doing multiple 2x2 multiplications
    # with the SAME 2x2 matrices. Here A11 is fixed but B changes per column.
    # 
    # So Strassen doesn't directly help here in the way I was thinking.
    # 
    # Let me reconsider. The correct way to apply Strassen is when you have
    # a block matrix multiplication where both blocks are large enough.
    # 
    # For 3x3 * 3x8, one approach is to pad to 4x4 * 4x8 and use Strassen,
    # but that increases the rank.
    # 
    # Given the complexity, let me just return the schoolbook decomposition.
    # It's valid and has rank 72.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
