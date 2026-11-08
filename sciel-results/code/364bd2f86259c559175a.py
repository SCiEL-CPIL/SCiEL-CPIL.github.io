import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[i,j,k] = delta_{i,k} (for n x m times m x p)
    # Actually T[i,j,k] = 1 if the output element (i,k) depends on input X[i,j]*Y[j,k]
    # For matrix multiplication: Z[i,k] = sum_j X[i,j] * Y[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k)
    
    # The tensor decomposition requires:
    # T[i,j,k] = sum_{r=1}^{R} U[i,r] * V[j,r] * W[k,r]
    # where i in [0,n), j in [0,m), k in [0,p)
    # But wait - the problem says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So U is indexed by (i*m+j, r) for X[i,j], V by (j*p+k, r) for Y[j,k], W by (k*n+i, r) for Z[i,k]
    
    # The constraint is: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (since Z[i,k] = sum_j X[i,j]Y[j,k], 
    #   but actually each term X[i,j]*Y[j,k] contributes to Z[i,k] with coefficient 1)
    
    # Wait, let me reconsider. The tensor for matrix multiplication (n x m) * (m x p) -> (n x p) is:
    # T[i, j, k] = 1 if we're computing Z[i,k] from X[i,j] * Y[j,k]
    # Actually the standard formulation: the multiplication tensor has entries T[i,j,k] where
    # i indexes rows of X (0..n-1), j indexes columns of X / rows of Y (0..m-1), 
    # k indexes columns of Y (0..p-1)
    # T[i,j,k] = 1 for all valid indices (it's the all-ones tensor for this problem)
    
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's use a direct approach: construct the full system and solve for minimum rank.
    # Start with the naive rank-64 solution and try to reduce.
    
    # Naive solution: rank = n*m*p = 64
    # For each (i,j,k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # where r = i*m*p + j*p + k
    
    # But we can do better. Let's try to find a lower rank solution.
    
    # Approach: Use the fact that we can share factors.
    # For fixed i: sum_j X[i,j] * Y[j,k] 
    # We can group by i and k.
    
    # Let's try a structured approach:
    # For each pair (i,k), we need: sum_j X[i,j] * Y[j,k]
    # This is an inner product of row i of X and column k of Y.
    
    # One approach: use rank-m decomposition where we factor the "middle" dimension.
    # But the middle dimension is m=4, so we can't reduce below... 
    # Actually, the tensor rank of (n x m) * (m x p) multiplication is known to be at most n*m*p,
    # but can be lower with Strassen-like methods.
    
    # For 2x4 * 4x8, let's try to find a good decomposition.
    
    # Strategy: Build the system of equations and solve.
    # Variables: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for r = 0..R-1
    # Equations: For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 64 equations with 3*R*(n*m + m*p + p*n) = 3*R*(8+32+16) = 144R unknowns.
    # For R=56: 8064 unknowns, 64 equations - very underdetermined.
    
    # Let's use a greedy approach: start with naive rank-64 and try to merge/eliminate terms.
    
    # Actually, let me try a different structural approach:
    # Use the Kronecker product structure. 
    # 2x4 * 4x8 can be viewed as: (2x2 * 2x1) * (2x2 * 4x2) roughly.
    
    # Let me try to explicitly construct a rank-50 or lower solution.
    
    # For 2x2 * 2x2, Strassen gives rank 7.
    # For 2x4 * 4x8, we can think of it as 2 blocks of (2x2 * 2x4) or similar.
    
    # Let's try: decompose the 4x8 matrix into 2x2 blocks.
    # Y is 4x8 = 2x2 blocks of size 2x4? No, 4x8 = 2*2 rows x 2*4 cols.
    # Actually 4x8 = (2x2) blocks: 2 block-rows x 4 block-cols of 2x2.
    
    # X is 2x4 = 1 block-row x 2 block-cols of 2x2.
    
    # So X = [X00, X01] where X00, X01 are 2x2.
    # Y = [Y00 Y01 Y02 Y03; Y10 Y11 Y12 Y13] where each Yij is 2x2.
    
    # Z = X * Y = [X00*Y00 + X01*Y10, X00*Y01 + X01*Y11, X00*Y02 + X01*Y12, X00*Y03 + X01*Y13]
    # Z is 2x8 = 1 block-row x 4 block-cols of 2x2.
    
    # So we have 4 terms of form (2x2)*(2x2) summed in pairs.
    # Each 2x2*2x2 can use Strassen rank 7.
    # Total: 4*2 = 8 Strassen blocks? No, that's not right.
    
    # Actually: Z_col0 = X00*Y00 + X01*Y10 (sum of two 2x2 products)
    # We need to compute X00*Y00 and X01*Y10, each rank 7 with Strassen, total 14.
    # Similarly for other columns: 4 columns * 14 = 56. That's what we have.
    
    # Can we do better? The issue is that X00 and X01 appear in multiple products.
    # If we precompute some combinations of X00, X01, we might save.
    
    # Let's try: For each output column block, we need X00*Y0c + X01*Y1c.
    # Using Strassen on the "block" level: treat [X00, X01] as a 2x4 and [Y0c; Y1c] as a 4x2.
    # But that's just the original problem for one column.
    
    # Alternative: Use a shared Strassen-like decomposition.
    # Precompute 7 linear combinations of [X00, X01] (each 2x2, so 8 values total).
    # Then for each Y column, compute 7 products.
    
    # Let's define:
    # S0 = X00 + X01
    # S1 = X00 - X01  
    # S2 = X00
    # S3 = X01
    # S4 = X00 + X01 (same as S0, not helpful)
    
    # Actually for 2x4 * 4x2, the tensor rank might be less than 14.
    # Let me just try to solve the linear system for a lower rank.
    
    # I'll use a random initialization approach: pick R, initialize U,V,W randomly,
    # then use gradient descent or iterative methods to satisfy the constraints.
    
    # But that might not converge to exact integer/half-integer solutions.
    
    # Let me try a more direct algebraic approach.
    
    # For the 2x4 * 4x8 problem, let's exploit the structure.
    # Z[i,k] = sum_{j=0}^{3} X[i,j] * Y[j,k]
    
    # For i=0: Z[0,k] = X[0,0]*Y[0,k] + X[0,1]*Y[1,k] + X[0,2]*Y[2,k] + X[0,3]*Y[3,k]
    # For i=1: Z[1,k] = X[1,0]*Y[0,k] + X[1,1]*Y[1,k] + X[1,2]*Y[2,k] + X[1,3]*Y[3,k]
    
    # Group by k: for each k, we have a 2x4 * 4x1 multiplication.
    # The tensor rank of 2x4 * 4x1 is at most 8 (naive) but might be less.
    
    # Actually, let's just try to find a rank-48 solution by hand.
    
    # Idea: For each of the 8 output columns (k=0..7), we need to compute:
    # Z[0,k] = sum_j X[0,j]*Y[j,k]
    # Z[1,k] = sum_j X[1,j]*Y[j,k]
    
    # This is a 2x4 * 4x1 problem. The tensor rank is at most 8.
    # But we can share computation across k values.
    
    # Let's try rank 48 = 6 * 8. For each k, use rank 6 for the 2x4*4x1.
    # But 2x4*4x1 has tensor rank at most 8, and I'm not sure it's 6.
    
    # Let me just go with the known rank-56 solution and verify it works.
    
    # Construct the rank-56 Strassen-based solution.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # For 2x2 * 2x2 -> 2x2:
    # a0 = (A00+A11)(B00+B11)
    # a1 = (A10+A11)(B00-B10)  
    # a2 = (A00-A10)(B10+B11)
    # a3 = A00(B01-B11)
    # a4 = A11(B10-B00)
    # a5 = (A00+A01)(B11-B01)
    # a6 = (A10+A11)(B00+B11)
    # C00 = a0 + a3 - a4 - a6
    # C01 = a2 + a4
    # C10 = a1 + a3
    # C11 = a0 - a1 + a5 + a6
    
    # Let me use a verified Strassen implementation.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A11-A01)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U factors (for A, indexed by (i,j)):
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: -A00 + A10
        [0, -1, 0, 1],  # M7: -A01 + A11
    ]
    
    # V factors (for B, indexed by (j,k)):
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B10 - B11
        [-1, 1, 0, 0],  # M4: -B00 + B01
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 1, 0, 1],   # M7: B10 + B11
    ]
    
    # W factors (for C, indexed by (i,k)):
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, 0],   # M2: C10
        [0, 1, 0, 0],   # M3: C01
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [-1, 0, 0, 0],  # M7: -C00
    ]
    
    # Verify Strassen
    def verify_strassen_2x2():
        np.random.seed(123)
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        C_expected = A @ B
        C_computed = np.zeros((2, 2))
        for r in range(7):
            a_r = sum(strassen_U[r][idx] * A[idx // 2, idx % 2] for idx in range(4))
            b_r = sum(strassen_V[r][idx] * B[idx // 2, idx % 2] for idx in range(4))
            for idx in range(4):
                i, k = idx // 2, idx % 2
                C_computed[i, k] += a_r * b_r * strassen_W[r][idx]
        return np.allclose(C_expected, C_computed, atol=1e-10)
    
    if not verify_strassen_2x2():
        # Fallback to naive rank-64
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    # Now build the 2x4 * 4x8 decomposition using Strassen blocks.
    # X is 2x4, viewed as [X0, X1] where X0, X1 are 2x2.
    # Y is 4x8, viewed as 2x4 blocks of 2x2: Y[i,j] for i in {0,1}, j in {0,1,2,3}.
    # Z is 2x8, viewed as 1x4 blocks of 2x2: Z[0,j] for j in {0,1,2,3}.
    
    # Z[0,j] = X0 * Y[0,j] + X1 * Y[1,j]
    
    # For each j, we need to compute X0*Y[0,j] + X1*Y[1,j].
    # Using Strassen on the "block" level isn't directly applicable since we have two products summed.
    
    # Instead, for each j, use two Strassen decompositions:
    # One for X0*Y[0,j] and one for X1*Y[1,j], total 14 per j, 4 j's = 56.
    
    # This is what the current code does. Let's keep it but verify correctness.
    
    def add_strassen_block(base_rank, x_block, y_block_row, y_block_col):
        """
        Add a Strassen block for computing X[x_block] * Y[y_block_row, y_block_col]
        x_block: 0 or 1 (which 2x2 block of X)
        y_block_row: 0 or 1 (which 2x2 block row of Y)
        y_block_col: 0..3 (which 2x2 block col of Y)
        """
        for r in range(7):
            current_rank = base_rank + r
            
            # U factors: X indices
            for idx in range(4):
                local_i, local_j = divmod(idx, 2)
                global_i = local_i  # X has 2 rows
                global_j = x_block * 2 + local_j  # X has 4 cols, split into 2 blocks of 2
                U[global_i * m + global_j][current_rank] += strassen_U[r][idx]
            
            # V factors: Y indices
            for idx in range(4):
                local_j, local_k = divmod(idx, 2)
                global_j = y_block_row * 2 + local_j  # Y has 4 rows
                global_k = y_block_col * 2 + local_k  # Y has 8 cols
                V[global_j * p + global_k][current_rank] += strassen_V[r][idx]
            
            # W factors: Z indices
            for idx in range(4):
                local_i, local_k = divmod(idx, 2)
                global_i = local_i  # Z has 2 rows
                global_k = y_block_col * 2 + local_k  # Z has 8 cols
                W[global_k * n + global_i][current_rank] += strassen_W[r][idx]
    
    # For each output column block j (0..3):
    # Z[0,j] = X0 * Y[0,j] + X1 * Y[1,j]
    # Use Strassen for X0*Y[0,j] (7 ranks) and X1*Y[1,j] (7 ranks)
    
    for j in range(4):
        # X0 * Y[0,j]
        add_strassen_block(j * 14, 0, 0, j)
        # X1 * Y[1,j]
        add_strassen_block(j * 14 + 7, 1, 1, j)
    
    # Verify the full decomposition
    def verify_full():
        np.random.seed(456)
        X = np.random.randn(n, m)
        Y = np.random.randn(m, p)
        Z_expected = X @ Y
        Z_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        Z_computed[i, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
        return np.allclose(Z_expected, Z_computed, atol=1e-10)
    
    if not verify_full():
        # Something went wrong, fall back to naive
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
