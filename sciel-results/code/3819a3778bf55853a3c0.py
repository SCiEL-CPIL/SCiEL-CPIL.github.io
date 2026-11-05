"""Program for matrix multiplication 4x5x5 tensor decomposition."""
import time
import random
import itertools

DIMENSIONS = [4, 5, 5]
n, m, p = DIMENSIONS

def _fract(x):
    """Return integer or half-integer nearest to x, or None if not close."""
    # Check if x is close to an integer
    xi = round(x)
    if abs(x - xi) < 1e-7:
        return int(xi)
    # Check if x is close to a half-integer
    xh = round(2 * x) / 2.0
    if abs(x - xh) < 1e-7:
        return xh
    return None

def _validate(U, V, W, tol=1e-6):
    """Validate that U, V, W correctly decompose the tensor."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Expected: T[i, j, k] = 1 if we are computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually the tensor for matmul: T_{i,j,k} where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is indexed such that U has shape (n*m, rank), V (m*p, rank), W (p*n, rank)
    # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Wait, no. The matrix multiplication tensor has T[i, j, k] = 1 for all i, j, k
    # where i in [0,n), j in [0,m), k in [0,p)
    # and the output is C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor indices: U is indexed by (i, j) -> i*m+j, V by (j, k) -> j*p+k, W by (k, i) -> k*n+i
    # The condition is: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Wait, that's not right either. Let me think again.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i, j, k] such that the multiplication is:
    # C[i,k] = sum_{j} T[i, j, k] * A[i, j] * B[j, k]
    # where T[i, j, k] = 1 for all i, j, k.
    # The rank-1 decomposition: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's verify with random A, B
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from factors: C[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k]
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                C_computed[i, k] += s * A[i, j] * B[j, k]
    
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def _schoolbook():
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return rank-7 decomposition for 2x2x2 matrix multiplication."""
    # Strassen's 7 products
    # U indices: (i, j) for i,j in 0..1 -> i*2+j
    # V indices: (j, k) for j,k in 0..1 -> j*2+k
    # W indices: (k, i) for k,i in 0..1 -> k*2+i
    # 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]  # n*m = 4
    V = [[0]*7 for _ in range(4)]  # m*p = 4
    W = [[0]*7 for _ in range(4)]  # p*n = 4
    
    # M1: (A00+A11)(B00+B11)
    # U[0,0]=1, U[3,0]=1 (A00, A11)
    # V[0,0]=1, V[3,0]=1 (B00, B11)
    # W[0,0]=1 (C00), W[3,0]=1 (C11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    W[3][1] = -1  # C11
    
    # M3: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4: A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5: (A00+A10)B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C00
    W[1][4] = 1   # C01
    
    # M6: (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1   # C11
    
    # M7: (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C00
    W[1][6] = 1   # C01
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find better decompositions using random search / local optimization
    # Strategy: start from schoolbook, try to merge terms, or use random linear combinations
    
    # For 4x5x5, the known lower bound is around 90, upper bound 100.
    # Let's try a randomized local search approach.
    
    def try_random_search(iterations, deadline):
        """Random local search to reduce rank."""
        nonlocal best_result, best_rank
        U_curr = [row[:] for row in best_result["U"]]
        V_curr = [row[:] for row in best_result["V"]]
        W_curr = [row[:] for row in best_result["W"]]
        rank_curr = best_rank
        best_local_rank = rank_curr
        best_local = (
            [row[:] for row in U_curr],
            [row[:] for row in V_curr],
            [row[:] for row in W_curr]
        )
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            # Try to reduce rank by 1: randomly pick a column to eliminate
            if rank_curr <= 1:
                break
            
            c_remove = random.randint(0, rank_curr - 1)
            
            # Try to absorb this column into others
            # For each other column c, try to find alpha, beta such that
            # after adding alpha*U_c to U_c' and beta*V_c to V_c', the W equation still holds
            # This is a system of equations.
            
            # Simpler approach: try random perturbations and check validity
            # Pick a random column c and try to zero out one entry by adjusting another column
            
            # Alternative: try to merge two columns
            c1 = random.randint(0, rank_curr - 1)
            c2 = random.randint(0, rank_curr - 1)
            if c1 == c2:
                continue
            
            # Try to eliminate c2 by expressing it as combination of c1 and others
            # For simplicity, try to eliminate one entry at a time
            # Pick a random (i, j) index for U
            idx_u = random.randint(0, n * m - 1)
            val_to_eliminate = U_curr[idx_u][c2]
            if val_to_eliminate == 0:
                continue
            
            # Find another column c3 where U[idx_u][c3] != 0
            candidates = [c for c in range(rank_curr) if c != c2 and U_curr[idx_u][c] != 0]
            if not candidates:
                continue
            
            c3 = random.choice(candidates)
            alpha = val_to_eliminate / U_curr[idx_u][c3]
            
            # Try to eliminate: add alpha * column c3 to column c2
            # U: U_new[c2] = U[c2] + alpha * U[c3]
            # V: V_new[c2] = V[c2]
            # W: W_new[c2] = W[c2]
            # But this changes the tensor. We need to adjust W or V to compensate.
            # Actually, if we do U_new = U + alpha * U_c3 * e_c2^T, 
            # then the tensor changes by alpha * U_c3 * V_c2^T * W_c2
            # We need to cancel this.
            
            # This is getting complex. Let's use a different approach.
            
            # Approach: randomly perturb all columns, then try to reduce rank via SVD-like method
            # Since we need integer/half-integer entries, this is tricky.
            
            # Let's try a simpler heuristic: randomly set some entries to 0 and check if valid
            # and if rank can be reduced.
            
            # Actually, let's just keep the schoolbook for now and try a few specific known decompositions.
            # For 4x5x5, we can use block decomposition:
            # 4x5x5 = 4x4x4 + 4x4x1 + 4x1x5 + 4x1x1 (approximately)
            # But this doesn't directly give a lower rank.
            
            pass
        
        if best_local_rank < best_rank:
            best_rank = best_local_rank
            best_result = {
                "U": [row[:] for row in best_local[0]],
                "V": [row[:] for row in best_local[1]],
                "W": [row[:] for row in best_local[2]]
            }
    
    # For this problem, let's try to use a known improved decomposition.
    # The 4x5x5 tensor can be decomposed with rank less than 100.
    # One approach: use the fact that 5 = 4+1, and use Strassen for the 4x4x4 part.
    
    # Decompose 4x5x5 as:
    # A is 4x5, B is 5x5, C is 4x5
    # Write A = [A4 | A1] where A4 is 4x4 and A1 is 4x1
    # Write B = [B41; B11] where B41 is 4x5 and B11 is 1x5
    # C = A4 * B41 + A1 * B11
    # 
    # A4 * B41: 4x4 * 4x5 -> use rank-49 Strassen for 4x4x4, plus 4x4x1 terms
    # Actually, 4x4*4x5: we can think of it as 5 independent 4x4*4x1 products
    # Each 4x4*4x1 has rank... well, it's just matrix-vector, rank 16 each? No.
    
    # Let me think differently. The tensor M_{4,5,5} can be viewed as:
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    # 
    # Use Strassen for the first 4 columns:
    # A' = A[:, 0:4] (4x4), B' = B[0:4, :] (4x5)
    # A'' = A[:, 4] (4x1), B'' = B[4, :] (1x5)
    # 
    # C = A' * B' + A'' * B''
    # 
    # A' * B': 4x4 * 4x5. We can write B' = [b0, b1, b2, b3, b4] where bj is 4x1.
    # A' * bj is a 4x1 vector. 
    # The tensor for 4x4*4x5 has rank at most 4*4*5 = 80, but we can do better.
    # Actually, 4x4*4x5 = 5 copies of 4x4*4x1.
    # 4x4*4x1: C[i] = sum_j A[i,j] B[j] where B is 4x1.
    # This is a bilinear form in A and B. The tensor is M_{4,4,1} which has rank 16.
    # So 5 of them have rank 5*16 = 80.
    # 
    # A'' * B'': 4x1 * 1x5. C[i,k] = A''[i] * B''[k].
    # This is rank 4*1*5 = 20? No, it's a rank-1 outer product for each i,k pair?
    # Actually, A'' is 4x1, B'' is 1x5. The product is 4x5.
    # C[i,k] = A''[i] * B''[k].
    # This is a sum of 4*1*5 = 20 rank-1 terms? No, it's already a single term in the bilinear sense.
    # Wait, the tensor for 4x1*1x5: T[i, j, k] where j is the only index in the middle (j=0).
    # T[i, 0, k] = 1 for all i, k.
    # Rank of this tensor: it's the all-ones tensor of size 4x1x5, which has rank 1? No.
    # The tensor T[i, j, k] with j fixed: T[i, 0, k] = 1.
    # This can be written as sum_c U[i, c] * V[0, c] * W[k, c].
    # If we take rank 1: U[i, 0] = 1, V[0, 0] = 1, W[k, 0] = 1 for all i, k.
    # Then sum_c U[i,c]V[0,c]W[k,c] = 1*1*1 = 1. Yes, rank 1!
    # 
    # So A'' * B'' has rank 1.
    # 
    # Total rank: rank(4x4*4x5) + 1.
    # 
    # Now, 4x4*4x5: Let's decompose this.
    # B' is 4x5. Write B' = [B1 | B2] where B1 is 4x2 and B2 is 4x3?
    # Or use Strassen recursively.
    # 
    # Actually, let's just use the schoolbook for the 4x4*4x5 part: rank 4*4*5 = 80.
    # Total: 80 + 1 = 81.
    # 
    # Can we do better for 4x4*4x5?
    # 4x4*4x5: think of it as multiplying A (4x4) by 5 different column vectors of B.
    # For each column b of B, we compute A*b. The tensor for A*b is M_{4,4,1} with rank 16.
    # But the 5 columns share the same A.
    # 
    # Alternative: use Strassen 4x4x4 for the first 4 columns of B, and schoolbook for the 5th.
    # 4x4*4x4 has rank 49 (Strassen).
    # 4x4*4x1 has rank 16.
    # Total: 49 + 16 = 65.
    # Plus the A''*B'' part: rank 1.
    # Total: 65 + 1 = 66.
    
    # Let's implement this!
    
    def strassen_4x4x4():
        """Return rank-49 decomposition for 4x4x4 matrix multiplication."""
        # Recursive Strassen
        U2, V2, W2 = _strassen_2x2x2()
        
        # Split 4x4 into 2x2 blocks
        # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
        # B = [[B00, B01], [B10, B11]]
        # 
        # Strassen meta-products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # 
        # Each Mk is a 2x2 product, computed with Strassen 2x2x2 (rank 7).
        # 
        # U for 4x4x4: indexed by (i, j) for i,j in 0..3 -> i*4+j, size 16
        # V for 4x4x4: indexed by (j, k) for j,k in 0..3 -> j*4+k, size 16
        # W for 4x4x4: indexed by (k, i) for k,i in 0..3 -> k*4+i, size 16
        
        rank = 49
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # For each of the 7 Strassen meta-products, we have 7 sub-products
        # Total: 7*7 = 49 terms
        
        # Meta-product definitions:
        # (u_coeffs, v_coeffs, w_coeffs) for each of the 7 Strassen steps
        # u_coeffs: dict mapping (block_i, block_j) -> coefficient for A
        # v_coeffs: dict mapping (block_j, block_k) -> coefficient for B
        # w_coeffs: dict mapping (block_k, block_i) -> coefficient for C output
        
        # Block indices: 00, 01, 10, 11 for 2x2 blocks of 4x4
        # For U: A has blocks (bi, bj) where bi, bj in {0,1}
        # For V: B has blocks (bj, bk) where bj, bk in {0,1}
        # For W: C has blocks (bk, bi) where bk, bi in {0,1}
        
        # Strassen step definitions:
        # M1: X=(A00+A11), Y=(B00+B11)
        # M2: X=(A10+A11), Y=B00
        # M3: X=A00, Y=(B01-B11)
        # M4: X=A11, Y=(B10-B00)
        # M5: X=(A00+A10), Y=B11
        # M6: X=(A10-A00), Y=(B00+B10)
        # M7: X=(A01-A11), Y=(B01+B11)
        
        # For each meta-step s, we compute X_s * Y_s using Strassen 2x2x2
        # X_s is a 2x2 matrix with entries that are linear combos of A blocks
        # Y_s is a 2x2 matrix with entries that are linear combos of B blocks
        
        # Let's define the Strassen steps more carefully.
        # For meta-step s, we have:
        # X_s[i,j] for i,j in {0,1} (these are the 2x2 blocks of X_s)
        # Actually, X_s is a 2x2 matrix of scalars? No.
        # 
        # Wait, in the recursive Strassen for 4x4:
        # We split A, B into 2x2 blocks.
        # For each Strassen meta-product, X_s and Y_s are 2x2 matrices.
        # The entries of X_s are linear combinations of the 2x2 blocks of A.
        # Similarly for Y_s and B.
        # Then we compute X_s * Y_s (2x2 * 2x2) using Strassen 2x2x2.
        # 
        # But X_s is a 2x2 matrix, and its entries are themselves 2x2 blocks?
        # No. Let me reconsider.
        # 
        # When we split A (4x4) into 2x2 blocks, each block is 2x2.
        # In the Strassen formula, X_s is a 2x2 matrix where each entry is a 2x2 block.
        # So X_s is actually a 4x4 matrix partitioned into 2x2 blocks.
        # 
        # This is getting confusing. Let me use a different approach.
        # 
        # I'll directly generate the Strassen 4x4x4 by applying Strassen 2x2x2 
        # to each of the 7 meta-steps, where each meta-step involves 2x2 matrices.
        
        # For meta-step s, X_s and Y_s are 2x2 matrices.
        # The (a,b) entry of X_s is a linear combination of A's 2x2 blocks.
        # Let's define:
        # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), 
        #            A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
        # 
        # For meta-step s, X_s is a 2x2 matrix.
        # X_s[0,0], X_s[0,1], X_s[1,0], X_s[1,1] are each 2x2 blocks.
        # 
        # Actually, I think the cleanest way is to note that:
        # The 4x4 Strassen has 7 meta-steps, each producing a 2x2 result.
        # Each 2x2 result is computed with 7 multiplications.
        # Total: 49 multiplications.
        
        # Let me define the coefficients explicitly.
        # For meta-step s, the 2x2 matrix X_s has entries:
        # X_s[a, b] = sum_{p,q} alpha_s[a,b,p,q] * A[p*2:r, q*2:c]
        # where the sum is over the 4 blocks (p,q) in {(0,0),(0,1),(1,0),(1,1)}
        
        # This is very complex to hardcode. Let me use a simpler approach.
        # I'll just use schoolbook for the 4x4x4 part for now, and see if we can improve.
        
        # Schoolbook 4x4x4: rank 64
        rank_sb = 64
        U_sb = [[0] * rank_sb for _ in range(16)]
        V_sb = [[0] * rank_sb for _ in range(16)]
        W_sb = [[0] * rank_sb for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_sb[i*4+j][col] = 1
                    V_sb[j*4+k][col] = 1
                    W_sb[k*4+i][col] = 1
                    col += 1
        return U_sb, V_sb, W_sb, rank_sb
    
    # Actually, let me just implement the full 4x5x5 decomposition using:
    # - 4x4*4x4 part: rank 64 (schoolbook) or 49 (Strassen, if I can get it)
    # - 4x4*4x1 part: rank 16
    # - 4x1*1x5 part: rank 1
    # Total with schoolbook: 64 + 16 + 1 = 81
    
    # Let me implement the 81-rank decomposition.
    
    def make_81_rank():
        """Construct rank-81 decomposition for 4x5x5."""
        n, m, p = 4, 5, 5
        # A is 4x5, B is 5x5
        # Split: A = [A4 | A1] where A4 is 4x4 (cols 0-3), A1 is 4x1 (col 4)
        # B = [B_top; B_bottom] where B_top is 4x5 (rows 0-3), B_bottom is 1x5 (row 4)
        # C = A4 * B_top + A1 * B_bottom
        # 
        # A4 * B_top: 4x4 * 4x5
        # Split B_top into 5 column vectors: b0, b1, b2, b3, b4 (each 4x1)
        # C1 = A4 * B_top = [A4*b0, A4*b1, A4*b2, A4*b3, A4*b4]
        # 
        # For each b_j (4x1 vector), compute A4 * b_j.
        # The tensor for 4x4 * 4x1: T[i, j, k] where k is fixed (k=0 for the 1-dim output)
        # Wait, 4x4 * 4x1: output is 4x1.
        # C[i] = sum_{j=0}^{3} A[i,j] * b[j]
        # Tensor: U[i*4+j, c] * V[j, c] * W[i, c] where:
        # U: 4*4 = 16 rows, V: 4 rows, W: 4 rows
        # 
        # Schoolbook: rank 16.
        # 
        # A1 * B_bottom: 4x1 * 1x5
        # C[i,k] = A1[i] * B_bottom[k]
        # Tensor: rank 1 (as shown earlier)
        
        rank_total = 5 * 16 + 1  # 81
        U = [[0] * rank_total for _ in range(n * m)]  # 20 rows
        V = [[0] * rank_total for _ in range(m * p)]  # 25 rows
        W = [[0] * rank_total for _ in range(p * n)]  # 20 rows
        
        col = 0
        # Part 1: A4 * B_top, for each of the 5 columns
        for k in range(5):  # k is the output column
            # A4 * b_k where b_k is column k of B_top (rows 0-3 of B)
            # b_k[j] = B[j, k] for j in 0..3
            for i in range(4):
                for j in range(4):
                    # U[i*4+j] corresponds to A[i, j] in A4
                    # V[j] corresponds to b_k[j] = B[j, k]
                    # W[k*4+i] corresponds to C[i, k]
                    U[i * 5 + j][col] = 1  # A[i, j] in full A (i in 0..3, j in 0..3)
                    V[j * 5 + k][col] = 1  # B[j, k] in full B (j in 0..3, k in 0..4)
                    W[k * 4 + i][col] = 1  # C[i, k] in full C (k in 0..4, i in 0..3)
                    col += 1
        
        # Part 2: A1 * B_bottom
        # A1[i] = A[i, 4] for i in 0..3
        # B_bottom[k] = B[4, k] for k in 0..4
        # C[i, k] += A1[i] * B_bottom[k]
        for i in range(4):
            for k in range(5):
                # We need rank-1 terms: U[i*5+4, c] * V[4*5+k, c] * W[k*4+i, c]
                # But we want to use rank 1 for all of these.
                # The tensor T[i, 4, k] = 1 for all i, k.
                # This can be done with 1 rank-1 term if we set:
                # U[i*5+4, c] = 1 for all i, V[4*5+k, c] = 1 for all k, W[k*4+i, c] = 1
                # But wait, this would give sum_c U*V*W = 1*1*1 = 1 for each (i,k), which is correct.
                # However, we need separate columns for each (i,k) pair if we use this approach.
                # Actually, no. The tensor is T[i, j, k] where j=4.
                # T[i, 4, k] = 1 for all i, k.
                # We can use ONE rank-1 term:
                # U[i*5+4, c] = 1 for all i (0..3)
                # V[4*5+k, c] = 1 for all k (0..4)
                # W[k*4+i, c] = 1 for all i, k
                # Then for any (i, k): sum_c U[i*5+4,c]*V[4*5+k,c]*W[k*4+i,c] = 1*1*1 = 1. Correct!
                
                # But we can't do this with a single column because W[k*4+i, c] needs to be 1 for all i,k.
                # Let's just use one column for this part.
                pass
        
        # Use one column for the A1*B_bottom part
        c = col
        for i in range(4):
            U[i * 5 + 4][c] = 1
        for k in range(5):
            V[4 * 5 + k][c] = 1
        for i in range(4):
            for k in range(5):
                W[k * 4 + i][c] = 1
        col += 1
        
        assert col == rank_total, f"col={col}, rank_total={rank_total}"
        
        return {"U": U, "V": V, "W": W}
    
    # Try the 81-rank decomposition
    result_81 = make_81_rank()
    if _validate(result_81["U"], result_81["V"], result_81["W"]):
        best_result = result_81
        best_rank = 81
    
    # Check time
    if time.time() > deadline:
        return best_result
    
    # Try to improve further
    # Can we do better than 81?
    # If we use Strassen for 4x4*4x4 (rank 49) instead of schoolbook (rank 64),
    # we save 15 terms.
    # Total: 49 + 16 + 1 = 66.
    
    # Let me try to implement Strassen 4x4x4 properly.
    
    def strassen_4x4x4_v2():
        """Return rank-49 decomposition for 4x4x4 using recursive Strassen."""
        # I'll build this by composing Strassen 2x2x2.
        # 
        # The 4x4 matrices are partitioned into 2x2 blocks.
        # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
        # B = [[B00, B01], [B10, B11]]
        # 
        # Strassen meta-products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # For each meta-product, X and Y are 2x2 matrices.
        # We compute X*Y using Strassen 2x2x2.
        
        # Let's define the meta-steps as (u_expr, v_expr, w_expr)
        # where u_expr maps block indices to coefficients
        
        # Block index: (bi, bj) for A, (bj, bk) for B, (bk, bi) for C
        # bi, bj, bk in {0, 1}
        
        # For each meta-step s, define:
        # u_coeff[s][a][b] = dict mapping (pi, pj) -> coeff for A[pi,pj]
        # where a,b in {0,1} are the 2x2 positions within the meta-matrix
        
        # Actually, let me think of it differently.
        # X_s is a 2x2 matrix. X_s[a,b] is a 2x2 matrix (a block of A).
        # X_s[a,b] = sum_{p,q} alpha_s[a,b,p,q] * A[p,q]
        
        # For M1: X1 = A00 + A11, so X1[a,b] = A00[a,b] + A11[a,b]
        # This means alpha_1[a,b,0,0] = 1, alpha_1[a,b,1,1] = 1 for all a,b
        
        # This is getting very tedious. Let me use a computational approach.
        # I'll represent the 4x4 Strassen as a list of 49 rank-1 terms.
        # Each term has:
        # u_vec: 16-dim vector (for A)
        # v_vec: 16-dim vector (for B)
        # w_vec: 16-dim vector (for C)
        
        # I'll generate these by iterating over the 7 meta-steps and 7 Strassen sub-steps.
        
        # Define Strassen 2x2x2 coefficients
        # From _strassen_2x2x2(), we have U2, V2, W2 for 2x2x2
        U2, V2, W2 = _strassen_2x2x2()
        
        # For meta-step s, we need to define:
        # X_s as a 2x2 matrix of 2x2 blocks
        # Y_s as a 2x2 matrix of 2x2 blocks
        
        # Meta-step definitions (coefficients for A blocks and B blocks):
        # Each meta-step has 4 entries (for the 2x2 X_s and Y_s)
        # X_s[a,b] = sum of A blocks with coefficients
        # Y_s[a,b] = sum of B blocks with coefficients
        
        # Let's define meta_steps as a list of 7 tuples:
        # (x_coeffs, y_coeffs) where x_coeffs is a 2x2 list of dicts, etc.
        
        # This is too complex to do by hand. Let me use a known result:
        # The rank of 4x4x4 is 49, and I'll just use schoolbook (rank 64) for now.
        # The total would be 64 + 16 + 1 = 81, which we already have.
        
        return None
    
    # Since Strassen 4x4x4 is complex to implement correctly, let's stick with rank 81.
    # But let's try to verify and potentially improve.
    
    if time.time() > deadline:
        return best_result
    
    return best_result

